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
import torch.nn as nn
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
    checkpoint = torch.load(f"Experiments/{exp_dir}/last_checkpoint.pth")
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
            labels.append(categories_map[category])
    
    return files,labels


def split_data(data,labels,test_size=0.2,random_state=18):
    train_data, test_data , train_labels , test_labels = train_test_split(data,labels,test_size=test_size,random_state=random_state)
    return train_data, test_data , train_labels , test_labels






training_data,training_labels = get_files("D:\df\\ai\Clean Data\clean_images")


training_data, val_data , training_labels , val_labels = split_data(training_data,training_labels,test_size=0.1)
num_train = len(training_data)
num_val = len(val_data)
print(f"num_train :{num_train}")
print(f"num_val :{num_val}")


csv_columns = ["Epoch","loss","val_loss","val_acc","val_f1"] 
transform = Transform(rows,columns,num_bins)

train_dataset = BrainDataset(training_data,training_labels,device,transform=transform)
val_dataset = BrainDataset(val_data,val_labels,device,transform=transform)

 
train_loader = DataLoader(train_dataset,batch_size=batch_size)
val_loader = DataLoader(val_dataset,batch_size=batch_size)



if model_name.lower() == "mlp":
    in_features = rows*columns*num_bins
    model = MLP(layers,in_features,dropout).to(device)

if model_name.lower() == "cnn":
    in_features = rows*columns*num_bins
    model = MLP(layers,in_features,dropout).to(device)

optimizer = torch.optim.Adam(model.parameters(),learning_rate)

if new :
    iepoch = 0
    best_val = np.inf
    csv_file =  open(os.path.join("Experiments",f"{exp_dir}","history.csv"),"w",newline="")
    writer  = csv.writer(csv_file)
    writer.writerow(csv_columns)
else :
    
    model.load_state_dict(torch.load(f"Experiments/{exp_dir}/last_weights.pt",weights_only=True))
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    csv_file =  open(os.path.join("Experiments",f"{exp_dir}","history.csv"),"a",newline="")
    iepoch = checkpoint["epoch"]+1
    best_val = checkpoint["best_val"]
    writer = csv.writer(csv_file)
        
def evaluate(model:torch.nn.Module,loader):
    model.eval()
    preds = []
    real = []

    with torch.no_grad() :
        for x , y in loader :
            x = x.to(device)
            out = model(x)
            out = out.to("cpu").numpy()

            pred = [0 if i<0.5 else 1 for i in out ]
            preds +=pred
            real += list(y)
    
    f1 = f1_score(preds,real)
    acc = accuracy_score(preds,real)
 
    return acc, f1

loss_fn = nn.BCELoss()
for epoch in range(iepoch,iepoch+NUM_EPOCHS):
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
        loss = loss_fn(out.view(-1),labels)
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
            loss = loss_fn(out.view(-1),labels)
            val_loss.append(float(loss))

        
    pbar.close()
    val_acc , val_f1 = evaluate(model,val_loader)
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
    eval_verbose = f"val acc:{val_acc} , val_f1:{val_f1}"

    if verbose > 0:
        print(loss_verbose,end="\t")

    if verbose > 1 :
        print(eval_verbose)
    print()
    writer.writerow([epoch,train_loss,val_loss,val_acc,val_f1])















