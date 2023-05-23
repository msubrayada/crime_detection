import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.metrics import roc_curve, auc, roc_auc_score
from torch.nn.functional import one_hot


labels = {"Normal": 0,
          "Abuse": 1,
          "Arrest": 2,
          "Arson": 3,
          "Burglary": 4,
          "Explosion": 5,
          "Fighting": 6,
          "RoadAccidents": 7,
          "Shooting": 8,
          "Vandalism": 9
         }


CLASS_LABELS = ["normal", 
                "Abuse",
                "Arrest",
                "Arson",
                "Burglary",
                "Explosion",
                "Fighting",
                "RoadAccidents",
                "Shooting",
                "Vandalism"]

def evaluate(data, model, device, batch_size):
    with torch.no_grad():
        d_pred = torch.zeros([len(data), 10])
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch = batch.to(device)
            pred = model(batch)
            d_pred[i:i+batch_size] = pred 
    return d_pred

def get_acc(pred, target, return_sum=False):
    s = (torch.argmax(pred, dim=1).round() == target).sum()
    if (return_sum == True):
        return s
    return s/len(pred)

def get_data(df, path="d:/data/UCF-crime/Anomaly-Videos-qformer-features", n=768):    
    X = np.zeros((len(df), n))
    c = 0
    for _, f in df.iterrows():
        directory = f["directory"][:-7] 
        X[c] = np.load(f"{path}/{directory}/{f['frame']}.npy")
        c += 1
    X = torch.tensor(X, dtype=torch.float32)
    y = list(df["class"])
    y = [labels[lbl] for lbl in y]
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

def plot_results(y, pred):   
    plt.figure(figsize = (12,3))
    plt.plot(y)
    for i in range(10):
        plt.plot(pred[:,i].numpy(), label=CLASS_LABELS[i])
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()

def show_frames(video, n=10):
    path = "d:/data/UCF-crime/Anomaly-Videos-frames"
    f, axarr = plt.subplots(1, n,  figsize=(15, 2)) 
    sk = len(video) // n
    for i in range(0, n):
        d = list(video["directory"])[i*sk]    
        im = imread(f"{path}/{d}/{list(video['frame'])[i*sk]}")
        axarr[i].imshow(im)
        axarr[i].title.set_text(f"Frame {i*sk}")
        axarr[i].axis('off')
    plt.show()
    
def get_acc_per_lbl(pred, target, return_sum=False):
    for i in range(len(labels)):
        p = (target == i).sum()
        t = (torch.argmax(pred, dim=1) == i).sum()
        s = ((torch.argmax(pred, dim=1) == i) & (target == i)).sum()
        presc = s/t
        recall = s/p
        print(f"{i}, presc {s/t:.2f}, recall {s/p:.2f}, f1 {2*presc*recall/(presc+recall):.2f}\t total {p} {list(labels.keys())[i]}")
    
    return 



def multiclass_roc_auc_score(y_true, y_pred, average="macro"):    
    fig, c_ax = plt.subplots(1,1, figsize = (8,5))
    for (idx, c_label) in enumerate(CLASS_LABELS):
        fpr, tpr, thresholds = roc_curve((y_true == idx).type(torch.int32), y_pred[:,idx])
        c_ax.plot(fpr, tpr,lw=2, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'black',linestyle='dashed', lw=4, label = 'Random Guessing')
    y_true = one_hot(y_true.type(torch.LongTensor))
    plt.xlabel('FALSE POSITIVE RATE', fontsize=18)
    plt.ylabel('TRUE POSITIVE RATE', fontsize=16)
    plt.legend(fontsize = 11.5)
    plt.show()
    return roc_auc_score(y_true, y_pred, average=average)

def put_momentum(pred, rho=0.9):
    v = torch.tensor([0]*10,  dtype=torch.float32)
    pred_momentum = []
    for p in pred:
        p_m = p + v * rho
        v = p_m
        p_m = p_m / (torch.sum(p_m))
        pred_momentum.append(torch.unsqueeze(p_m, 0))    
    pred_momentum = torch.cat(pred_momentum, dim=0)
    return pred_momentum

def put_momentum_per_video(pred, videos, rho=0.9):
    v = torch.tensor([0]*10,  dtype=torch.float32)
    pred_momentum = []
    count = 0
    video = videos[0]
    for p in pred:
        if (videos[count] != video):
            video = videos[count]
            v = torch.tensor([0]*10,  dtype=torch.float32)
        p_m = p + v * rho
        v = p_m
        p_m = p_m / (torch.sum(p_m))
        pred_momentum.append(torch.unsqueeze(p_m, 0))    
        count += 1
    pred_momentum = torch.cat(pred_momentum, dim=0)
    return pred_momentum