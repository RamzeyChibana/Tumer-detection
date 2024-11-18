import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
from utils.parser import plot_hist
import os



parser = plot_hist()
args = parser.parse_args()

exp = args.exp
print(exp)

exp = f"exp_{exp}"
exps = os.listdir("Experiments")
if exp not in exps :
    raise ValueError(f"There is no Experiement {exp} to test")

history = pd.read_csv(os.path.join("Experiments",exp,"history.csv"))
metrices = list(history.columns)[2:] # remove epoch from columns



x = np.arange(history.shape[0])

fig , axes = plt.subplots(1,2,figsize=(15,10))


axes[0].set_title(f"Loss over epochs")
axes[0].set_xlabel(f"epochs")
axes[0].set_ylabel(f"BCE loss")
axes[0].plot(x,history["loss"].values,label="training loss")
axes[0].plot(x,history["val_loss"].values,label="val loss")
axes[0].legend()

axes[1].set_title(f"Loss over epochs")
axes[1].set_xlabel(f"epochs")
axes[1].set_ylabel(f"acc/f1")
axes[1].plot(x,history["val_f1"].values,label="f1")
axes[1].plot(x,history["val_acc"].values,label="accuracy")
axes[1].legend()

plt.show()