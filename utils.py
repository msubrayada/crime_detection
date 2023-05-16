#import pandas as pd
import torch
#import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread


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

def get_data(df, path="d:/data/UCF-crime/Anomaly-Videos-qformer-features"):    
    X = np.zeros((len(df), 768))
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