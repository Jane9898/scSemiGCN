import torch
import numpy as np
import enhancement
import numpy as np


def adj_function(feature,labels):
    adj = np.zeros([len(labels), len(labels)])

    t = 7000
    for i in range(len(labels)):
        for j in range(len(labels)):
            f = (feature[i]-feature[j])
            adj[i,j] = torch.exp(-f@f.T/t)


    b = torch.zeros_like(torch.tensor(labels),dtype=torch.float64)
    #b = torch.zeros_like(torch.tensor(labels))
    for i in range(len(labels)):
        knn_idx = sorted(enumerate(adj[i]),key=lambda x:x[1],reverse=True)
        for (j1,j2) in knn_idx[:20]:

            b[i] += torch.norm(feature[i]-feature[j1],1)
        b[i] = b[i]/20

    for i in range(len(labels)):
        for j in range(len(labels)):
            f = (feature[i]-feature[j])

            adj[i,j] = torch  .exp(-20000*f@f.T/(b[i]+b[j])**2)

    def EN(adj):
        a = enhancement.network_enhancement(adj, 2, 20, 0.5)#20,0.5(蝴蝶)
        return a


    return adj



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
        adj_sort = [i[0] for i in sorted_nums]###返回adj中的行，即样本i与其他样本的邻接值####
        #adj_sort= adj_i_row.sort(reverse=True).indices  ###返回降序排序后的值的样本下标#####
        for i in range(k):
            if adj_sort[i] in list(dic.keys()):####如果第i个近邻在已知标签的样本里面，记下它的标签###
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
        adj_sort = [i[0] for i in sorted_nums]  ###返回adj中的行，即样本i与其他样本的邻接值####
        # adj_sort= adj_i_row.sort(reverse=True).indices  ###返回降序排序后的值的样本下标#####
        for i in range(k):
            if adj_sort[i] in list(dic.keys()):  ####如果第i个近邻在已知标签的样本里面，记下它的标签###
                a = adj_sort[i]
                knn_idx[int(dic[a]) - 1] += 1
        knn_idx = torch.tensor(knn_idx)#####没有，返回【0，0，0】
        x[0] = torch.argmax(knn_idx) + 1
        test_dataset.append([int(x[0]),int(x[1])])


    for x in val_dataset:
        dic[int(x[1])] = int(x[0])
    for x in test_dataset:
        dic[int(x[1])] = int(x[0])

    return dic




def knn_similarity(labels, adj, k):
    if k > len(labels):
        raise ValueError("K is too large!")

    unannotated_idx = np.arange(len(labels), adj.shape[0])
    annotated_idx = np.arange(len(labels))
    
    pseudo_label = []
    for i in range(len(unannotated_idx)):
        this_idx = unannotated_idx[i]
        this_sim = adj[this_idx, ]
        this_sim_ = this_sim[annotated_idx]
        sort_idx = np.argsort(-this_sim_)
        label_collection = labels[sort_idx[0:k]]
        this_pseudo_label = np.argmax(np.bincount(label_collection))
        pseudo_label.append(this_pseudo_label)
    
    pseudo_label = np.array(pseudo_label)
    
    pseudo_label = np.hstack((labels, pseudo_label))


    return pseudo_label

