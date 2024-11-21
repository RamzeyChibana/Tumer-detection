import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from models import RaGrd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

print("split data")
X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=100,random_state=18)
print("training")
model = RandomForestClassifier(n_estimators=100,  max_depth=10)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

