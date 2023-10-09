import logging
import os
import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval(pro, dataload, num_class):
    y_pred_l = []
    y_true_l = []
    y_pred_auc = []
    with torch.no_grad():
        acc = 0
        for x in dataload:

            x_ = pro[x.T[1].long()]
            x_1 = x_.cpu().numpy()
            y_true = x.T[0]
            y_pred = torch.argmax(x_,-1,False)+1

            y_true = y_true.cpu().numpy()
            y_pred = y_pred.cpu().numpy()
            y_pred_l += list(y_pred.ravel())
            y_true_l += list(y_true.ravel())
            y_pred_auc += list(x_1)

        y_true_l = np.array(y_true_l).ravel()
        y_true_l_auc = np.array([a-1 for a in y_true_l]).ravel()
        y_true_onehot = np.array(label_binarize(y_true_l_auc,classes=np.arange(num_class)))
        y_pred_l = np.array(y_pred_l).ravel()
        y_pred_auc = np.array(y_pred_auc).reshape(y_true_onehot.shape[0],y_true_onehot.shape[1])

        acc = np.sum(y_pred_l==y_true_l)/(len(dataload.dataset))
        f1_score_ = f1_score(y_true_l,y_pred_l,average='weighted')
    

        auc = metrics.roc_auc_score(y_true_onehot,y_pred_auc,average='samples', 
        sample_weight=None, max_fpr=None, multi_class='raise', labels=None)

    return acc, f1_score_ , auc



