import torch
import numpy as np


def pseudo_labels(adj,train_dataset_,val_dataset_,test_dataset_,num_class,k):

    val_dataset = []
    test_dataset = []
    adj = adj.tolist()
    dic={}
    for x in train_dataset_:
        dic[int(x[1].tolist())] =int(x[0].tolist())

    for x in val_dataset_:

        knn_idx = np.zeros(num_class)
        idx = x[1]
        adj_i_row = adj[idx.long()]
        sorted_nums = sorted(enumerate(adj_i_row), key=lambda x: x[1],reverse = True)
        adj_sort = [i[0] for i in sorted_nums]
        
        for i in range(k):
            if adj_sort[i] in list(dic.keys()):
                a = adj_sort[i]
                knn_idx[int(dic[a])-1]+=1
        knn_idx = torch.tensor(knn_idx)
        x[0] = torch.argmax(knn_idx)+1
        val_dataset.append([int(x[0]),int(x[1])])

    for x in test_dataset_:

        knn_idx = np.zeros(num_class)
        idx = x[1]
        adj_i_row = adj[idx.long()]
        sorted_nums = sorted(enumerate(adj_i_row), key=lambda x: x[1], reverse=True)
        adj_sort = [i[0] for i in sorted_nums]  
        
        for i in range(k):
            if adj_sort[i] in list(dic.keys()):  
                a = adj_sort[i]
                knn_idx[int(dic[a]) - 1] += 1
        knn_idx = torch.tensor(knn_idx)
        x[0] = torch.argmax(knn_idx) + 1
        test_dataset.append([int(x[0]),int(x[1])])


    for x in val_dataset:
        dic[int(x[1])] = int(x[0])
    for x in test_dataset:
        dic[int(x[1])] = int(x[0])

    return dic




def knn_similarity(train_idx,labels,test_idx,val_idx,sim_matrix, k):
    if k >= len(train_idx):
        raise ValueError('K is too large!')
    labels = labels.ravel()
    num_class = len(set(labels))
    train_labels = labels[train_idx]
   
    train_matrix = sim_matrix[:,train_idx]
    
    sorted_nums = np.argsort(-train_matrix)
    test_labels_pseudo = np.zeros_like(test_idx)
    test_labels_pseudo_auc = []
    n =-1
    for x in test_idx:
        n+=1
        pseudo_labels_onehot = np.zeros(num_class)
        for i in range(k):
            
            pseudo_labels_onehot[train_labels[sorted_nums[x][i]]-1]+=1
      
        test_labels_pseudo[n] = np.argmax(pseudo_labels_onehot)+1
        test_labels_pseudo_auc.append(pseudo_labels_onehot/k)

    val_labels_pseudo = np.zeros_like(val_idx)
    val_labels_pseudo_auc = []
    n=-1
    for x in val_idx:
        n+=1
        pseudo_labels_onehot = np.zeros(num_class)
        for i in range(k):
            pseudo_labels_onehot[train_labels[sorted_nums[x][i]] - 1] += 1
        val_labels_pseudo[n] = np.argmax(pseudo_labels_onehot) + 1
        val_labels_pseudo_auc.append(pseudo_labels_onehot/k)

    dic = {}
    n=0
    for x in train_idx:
        dic[x] =train_labels[n]
        n+=1
    n=0
    for x in val_idx:
        dic[x] = val_labels_pseudo[n]
        n+=1
    n=0
    for x in test_idx:
        dic[x] = test_labels_pseudo[n]
        n+=1

    return dic

