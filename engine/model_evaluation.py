import pickle
import numpy as np
import matplotlib.pyplot as plt 
import os

with open(os.path.join(os.path.dirname(__file__),"__apps_model","train_loss.pkl"),"rb") as loss_t:
    loss_train=pickle.load(loss_t)
with open(os.path.join(os.path.dirname(__file__),"__apps_model","val_loss.pkl") ,"rb") as loss_v:
    loss_val=pickle.load(loss_v)

t=np.array(list(loss_train.values()))
v=np.array(list(loss_val.values()))

e=np.array(np.arange(1,len(loss_train)+1))

plt.plot(e,t,label="Training-loss")
plt.plot(e,v,label="Validation-loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="best")
#add ticks 
plt.xticks(np.arange(1,len(loss_train)+1,1))
plt.show()




