import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from models import RaGrd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from tqdm import tqdm

rows,columns,num_bins = 3,3,255
Path = f"D:\df\\ai\Clean Data\clean_features\{rows}_{columns}_{num_bins}"


def get_features(path):
    categories = os.listdir(path)

    categories_map = {category:i for i,category in enumerate(categories)}
   
    features = []
    labels = []
    for category in categories :
        files = os.listdir(os.path.join(path,category))
        pbar = tqdm(total=len(files))
        pbar.set_description(f"{category} Files")
        for file in os.listdir(os.path.join(path,category)):
            features.append(np.load(os.path.join(path,category,file))) 
            labels.append(categories_map[category])
            pbar.update(1)
        pbar.close()
    return features,labels
print("get features")
features , labels = get_features(Path)

print("get centroids")
features,centroids_features,labels,centroids_labels = train_test_split(features,labels,test_size=100,random_state=18)
features , labels = features[:5000] , labels[:5000] 
print(pd.Series(centroids_labels).value_counts())

print("predict")
model = RaGrd(10)

result = model.predict(features,centroids_features,centroids_labels)
acc = accuracy_score(labels,result)
f1 = f1_score(labels,result)
print(f"accuracy :{acc}")
print(f"f1 score :{f1}")


    






