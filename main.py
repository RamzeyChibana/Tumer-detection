import numpy as np
import matplotlib.pyplot as plt
import torch 
from utils.DataPre import get_features,get_regions
from utils.load_data import BrainDataset , Transform
from torch.utils.data import DataLoader
from utils.parser import train_parse
import os 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score , accuracy_score , confusion_matrix
from models import MLP
import csv
import json
import argparse
from tqdm import tqdm

from time import time




def save_args(args, filename='args.json'):
    with open(filename, 'w') as f:
        json.dump(vars(args), f, indent=4)

def load_args(filename='args.json'):
    with open(filename, 'r') as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)

parse = train_parse()
args = parse.parse_args()

NUM_EPOCHS = args.epochs
gpu = args.device
verbose = args.verbose
exp = args.exp
device = torch.device("cuda" if gpu=="gpu" else "cpu")
exps = os.listdir("Experiments")
exp_dir = f"exp_{exp}"
if exp_dir in exps:
    print("\t\tContinue Training..")
    args = load_args(os.path.join("Experiments",f"{exp_dir}","args.json"))
    new = False

else :
    exp_dir = f"exp_{len(exps)}"
    print("\t\tStart Training..")
    os.makedirs(os.path.join("Experiments",f"{exp_dir}"))
    save_args(args,os.path.join("Experiments",f"{exp_dir}","args.json"))
    new = True

batch_size = args.batch_size
learning_rate = args.learning_rate
layers = args.layers
dropout = args.dropout
model_name = args.model
rows = args.rows
columns = args.columns
num_bins = args.bins




def get_files(path):
    categories = os.listdir(path)
    categories_map = {category:i for i,category in enumerate(categories)}
   
    files = []
    labels = []
    for category in categories :
        for file in os.listdir(os.path.join(path,category)):
            files.append(os.path.join(path,category,file)) 
            one_hot = np.zeros(shape=(len(categories)))
            one_hot[categories_map[category]]=1
            labels.append(one_hot)
    
    return files,labels,categories


def split_data(data,labels,test_size=0.2,random_state=18):
    train_data, test_data , train_labels , test_labels = train_test_split(data,labels,test_size=test_size,random_state=random_state)
    return train_data, test_data , train_labels , test_labels






training_data,training_labels,classes = get_files("D:\df\\ai\mri\Training")


training_data, val_data , training_labels , val_labels = split_data(training_data,training_labels,test_size=0.1)
num_train = len(training_data)
num_val = len(val_data)
print(f"num_train :{num_train}")
print(f"num_val :{num_val}")


csv_columns = ["Epoch","loss","val_loss","acc","val_acc","f1","val_f1"] 
transform = Transform(rows,columns,num_bins)

train_dataset = BrainDataset(training_data,training_labels,device,transform=transform)
val_dataset = BrainDataset(val_data,val_labels,device,transform=transform)

 
train_loader = DataLoader(train_dataset,batch_size=batch_size)
val_loader = DataLoader(val_dataset,batch_size=batch_size)



if model_name.lower() == "mlp":
    in_features = rows*columns*num_bins
    num_classes = len(classes)
    model = MLP(layers,in_features,num_classes,dropout).to(device)

if model_name.lower() == "cnn":
    in_features = rows*columns*num_bins
    num_classes = len(classes)
    model = MLP(layers,in_features,num_classes,dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(),learning_rate)

if new :
    iepoch = 0
    best_val = np.inf
    csv_file =  open(os.path.join("Experiments",f"{exp_dir}","history.csv"),"w",newline="")
    writer  = csv.writer(csv_file)
    writer.writerow(csv_columns)
else :
    checkpoint = torch.load(f"Experiments/{exp_dir}/last_checkpoint.pth")
    csv_file =  open(os.path.join("Experiments",f"{exp_dir}","history.csv"),"a",newline="")
    model.load_state_dict(torch.load(f"Experiments/{exp_dir}/last_weights.pt",weights_only=True))
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    iepoch = checkpoint["epoch"]+1
    best_val = checkpoint["best_val"]

        
def evaluate(model:torch.nn.Module,loader):
    preds = []
    actual = []
    cuda_time = 0
    feed_time = 0
    max_time = 0
    torch.cuda.synchronize()
    t = time()
    with torch.no_grad():
        for data,labels in loader:
            torch.cuda.synchronize()
            t1 = time()
            data = data.to(device)
            labels = labels.to(device)
            torch.cuda.synchronize()
            t2 = time()
            out = model(data)
            torch.cuda.synchronize()
            t3 = time()
            preds.append(torch.argmax(out,dim=1))
            actual.append(torch.argmax(labels,dim=1))
            torch.cuda.synchronize()
            t4 = time()
            cuda_time +=t2 - t1
            feed_time +=t3 - t2 
            max_time += t4 - t3 

    torch.cuda.synchronize()
    t_end = time()
    cuda_time =cuda_time / t_end - t
    feed_time =feed_time / t_end - t 
    max_time =max_time /  t_end - t 

    preds = torch.cat(preds,dim=0)
    actual = torch.cat(actual,dim=0)
    print(f"testing :cuda :{cuda_time}\tfeed :{feed_time}\targmax :{max_time}")
    preds = list(preds.cpu().numpy())
    actual = list(actual.cpu().numpy())

    acc = accuracy_score(actual,preds)
    f1 = f1_score(actual,preds,average="macro")
 
    return acc, f1

loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(iepoch,NUM_EPOCHS):
    train_loss = []
    val_loss = []
    pbar = tqdm(total=num_train)
    pbar.set_description(f"Epoch {epoch}:")
    for data,labels in train_loader:
        #feed forward
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        loss = loss_fn(out,labels)
        # backward
        train_loss.append(float(loss))
        loss.backward()
        optimizer.step()
        pbar.update(batch_size)

    for data,labels in val_loader:
        #feed forward
        with torch.no_grad() :
            data = data.to(device)
            labels = labels.to(device)
            out = model(data)
            loss = loss_fn(out,labels)
            val_loss.append(float(loss))

        
    pbar.close()
    val_acc , val_f1 = evaluate(model,val_loader)
    train_acc , train_f1 = evaluate(model,train_loader)
    train_loss = np.mean(train_loss)
    val_loss = np.mean(val_loss)
    
    if val_loss < best_val :
        best_val = val_loss
        torch.save(model.state_dict(),f"Experiments/{exp_dir}/best_weights.pt")
    
    checkpoint = {
    "epoch":epoch,
    "optimizer_state":optimizer.state_dict(),
    "best_val":best_val
    }
    torch.save(checkpoint,f"Experiments/{exp_dir}/last_checkpoint.pth")
    torch.save(model.state_dict(),f"Experiments/{exp_dir}/last_weights.pt")
    loss_verbose = f"Epoch {epoch}:loss=[{np.mean(train_loss):.3f}],val loss={np.mean(val_loss):.3f}"
    eval_verbose = f"acc:{train_acc} , f1:{train_f1} | val acc:{val_acc} , val_f1:{val_f1}"

    if verbose > 0:
        print(loss_verbose,end="\t")

    if verbose > 1 :
        print(eval_verbose)
    print()
    writer.writerow([epoch,train_loss,val_loss,train_acc,val_acc,train_f1,val_f1])















