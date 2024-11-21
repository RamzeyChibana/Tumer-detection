import torch 
from sklearn.metrics import accuracy_score,f1_score
from utils.parser import test_parse
import os 
import json 
import argparse
from models import MLP
from sklearn.model_selection import train_test_split
from utils.load_data import Transform,BrainDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
def load_args(filename='args.json'):
    with open(filename, 'r') as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)


parser = test_parse()
args = parser.parse_args()
gpu = args.device
batch_size = args.batch_size
exp = args.exp
device = torch.device("cuda" if gpu=="gpu" else "cpu")
exps = os.listdir("Experiments")
exp_dir = f"exp_{exp}"
if exp_dir not in exps:
    raise ValueError("No such Exp")
args = load_args(os.path.join("Experiments",f"{exp_dir}","args.json"))

layers = args.layers
dropout = args.dropout
rows = args.rows
columns = args.columns
num_bins = args.bins



in_features = rows*columns*num_bins
model = MLP(layers,in_features,dropout).to(device)

model.load_state_dict(torch.load(f"Experiments/{exp_dir}/best_weights.pt",weights_only=True))



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
transform = Transform(rows,columns,num_bins)

train_dataset = BrainDataset(training_data,training_labels,device,transform=transform)
val_dataset = BrainDataset(val_data,val_labels,device,transform=transform)
train_loader = DataLoader(train_dataset,batch_size=batch_size)
val_loader = DataLoader(val_dataset,batch_size=batch_size)

def evaluate(model:torch.nn.Module,loader,train=False):
    model.eval()
    preds = []
    real = []
    pbar = tqdm(total=len(loader)*batch_size)
    pbar.set_description("Testing :")
    with torch.no_grad() :
        for x , y in loader :
            x = x.to(device)
            out = model(x)
            out = out.to("cpu").numpy()

            pred = [0 if i<0.5 else 1 for i in out ]
            preds +=pred
            real += list(y)
            pbar.update(batch_size)

    pbar.close()
    f1 = f1_score(preds,real)
    acc = accuracy_score(preds,real)
 
    return acc, f1

train_acc , train_f1 = evaluate(model,train_loader)
val_acc , val_f1 = evaluate(model,val_loader)
eval_verbose = f"train acc:{train_acc} , train_f1:{train_f1},val acc:{val_acc} , val_f1:{val_f1}"
print(eval_verbose)

