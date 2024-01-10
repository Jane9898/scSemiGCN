import enhancement
import torch
import numpy as np
from scipy.io import loadmat, savemat
import argparse
from torch.utils.data import DataLoader
from TwoLayerGCN import GCN
import argparse
from data_cut import trian_val_test
from validation import eval
import torch.nn as nn
import enhancement
import random
import  os
# from PseudoLabels import pseudo_labels, knn_similarity
from contrastive_loss import contrastive_loss
from OneLayerGCN import preprocessor
import pandas as pd
import anndata as ad


def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
     

def pretrain(model,feature,pseudo_labels,loss_func,optimizer):

    by_key = sorted(pseudo_labels.items(), key=lambda item: item[0], reverse=False)
    pseudo_labels_list = []
    
    for i in by_key:
        pseudo_labels_list.append(i[1])
    
    pseudo_labels_tensor = torch.tensor(pseudo_labels_list,dtype=torch.int32)
    
    feature = feature.float()
    
    for i in range(10):
        feature_ = model(feature)

        loss = loss_func(feature_, pseudo_labels_tensor, 0.5)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        print('contrastive loss[%d/%d]: %.4f' % (i+1, 10, loss))
    
    return feature_


def train(model, features, loss, optimizer, train_datasest, val_dataset, test_dataset, num_class, epochs):
    best_acc =0
    best_epoch = 0
    for i in range(epochs):
        model.train()
        losses_train = 0
        for x in train_datasest:
            pro = model(features)
            y = x.T[0]-1
            a = pro[x.T[1].long()]

            loss_ = loss(a, y.long())
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()

            losses_train += loss_.cpu().item()

        eval_train = eval(pro, train_dataset, num_class)
        eval_val = eval(pro, val_dataset, num_class)

        if eval_val[0] > best_acc:
            best_acc = eval_val[0]
            best_epoch = i
            best_val_acc = eval_val[0]
            best_val_f1 = eval_val[1]
            best_val_auc = eval_val[2]
            #best_path = save_path
            torch.save(model.state_dict(), "model.pt")
        print(
            'In epoch: %d, losses: %.4f, acc_train:%.4f,acc_val:%.4f,f1_train:%.4f,f1_val:%.4f,auc_train:%.4f,auc_val:%.4f '% (
            i, losses_train, eval_train[0], eval_val[0],  eval_train[1], eval_val[1],eval_train[2], eval_val[2]))

    model.load_state_dict(torch.load("model.pt"))
    pro = model(features)
    eval_test = eval(pro, test_dataset, num_class)
    print('best_epoch:%d,test_acc:%.4f,test_f1:%.4f,test_auc:%.4f,val_acc:%.4f,val_f1:%.4f,val_auc:%.4f'%(best_epoch,eval_test[0],eval_test[1],eval_test[2],best_val_acc,best_val_f1,best_val_auc))
    return eval_test[0]

def params():
    args = argparse.ArgumentParser()
    args.add_argument('--model', default='gcn')
    args.add_argument('--hidden', type=int, default=100)
    args.add_argument('--dropout', type=float, default=0.5)
    args.add_argument('--weight_decay', type=float, default=5e-4)
    args.add_argument('--early_stopping', type=int, default=10)
    args.add_argument('--max_degree', type=int, default=4)
    args.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    args.add_argument('--epoch', type=float, default=100, help='num of epoch')
    args.add_argument('--glr', type=float, default=0.001, help='Initial learning rate for GCN.')
    args.add_argument('--slr', type=float, default=0.05, help='Initial learning rate for supervised contrastive learning.')
    args.add_argument('--t', type=float, default=7000, help='I')
    args.add_argument("--batch_size", type=float, default=100, help="Batch size for training GCN.")
    args = args.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args



if __name__=="__main__":

    seed_torch()
    # args = params()
    # data = loadmat("seqdata.mat")

    # adj = data["similarity"]
    # labels = data["annotation"]
    # feature = data["feature"]
    # print()

    data = loadmat('Buettner.mat')
    feature = torch.from_numpy(data['in_X'])
    labels = data['true_labs']
    feat_dim = feature.shape[1]
    num_class = len(np.unique(labels))

    adj = data['S']
    # # adj = enhancement.network_enhancement(adj, 2, 18, 0.5) 
    # # adj = torch.from_numpy(adj)
    # # adj = adj.to(torch.float32)
    
    # # premodel = preprocessor(feat_dim, args.dropout, adj)
    # # net = GCN(feat_dim, args.hidden, num_class, args.dropout, adj)
    
    fea_lab = [feature, labels]
    generate = trian_val_test(fea_lab,0.95,0.5)
    train_dataset_, val_dataset_, test_dataset_, train_idx, val_idx, test_idx = generate.generate_train_val()
    reorder_idx = np.hstack((train_idx, test_idx, val_idx))

    reorder_adj_ = adj[reorder_idx, :]
    reorder_adj = reorder_adj_[:, reorder_idx]

    feature_ = feature.numpy()
    reorder_feature = feature_[reorder_idx, :]
    seqdata = ad.AnnData(reorder_feature, dtype=reorder_feature.dtype)
    labels_ = np.int32(labels[reorder_idx].ravel())
    for i in range(len(labels)):
        if i>= len(train_idx):
            labels_[i] = -1
        else:
            labels_[i] = labels_[i]
    
    seqdata.obs["cell_type"] = pd.Categorical(labels_)
    seqdata.obsm["similarity"] = reorder_adj
    print(seqdata)
    # seqdata.write("demo_data.h5ad", compression="gzip")
    # seqdata.write_csvs("demo_data")
    mdict = {"feature": reorder_feature, "annotation": labels_, "similarity": reorder_adj}
    savemat("seqdata.mat", mdict)

    # train_dataset = DataLoader(train_dataset_,batch_size=args.batch_size,shuffle=True)
    # val_dataset = DataLoader(val_dataset_,batch_size=args.batch_size,shuffle=True)
    # test_dataset = DataLoader(test_dataset_,batch_size=args.batch_size, shuffle=True)


    # pseudo_labels = knn_similarity(train_idx, labels, test_idx, val_idx, adj, 1)


    # loss = nn.CrossEntropyLoss(reduction="mean")

    # optimizer1 = torch.optim.SGD(params=premodel.parameters(),lr=args.slr, weight_decay=1e-2)
    # optimizer2 = torch.optim.Adam(params=net.parameters(), lr=args.glr)
    
    # print("#"*40, "Feature Refinement", "#"*40)
    # feature_new = pretrain(premodel,feature,pseudo_labels, contrastive_loss, optimizer1).detach()

    # print("#"*40, "Cell Type Annotation", "#"*40)
    # test_acc = train(net, feature_new, loss, optimizer2, train_dataset, val_dataset, test_dataset, num_class, args.epoch)
