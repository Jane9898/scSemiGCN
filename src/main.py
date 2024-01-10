import enhancement
import torch
import numpy as np
from scipy.io import loadmat
import argparse
from torch.utils.data import DataLoader
from TwoLayerGCN import GCN
import argparse
from validation import eval, make_prediction
import torch.nn as nn
import enhancement
import random
import os
from PseudoLabels import knn_similarity
from contrastive_loss import contrastive_loss
from OneLayerGCN import preprocessor
import pandas as pd



def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
     

def pretrain(args, model, feature, pseudo_labels, loss_func, optimizer):
    feature = feature.float()
    for i in range(args.round):
        feature_ = model(feature)
        loss = loss_func(feature_, pseudo_labels, args.tau)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('contrastive loss[%d/%d]: %.4f' % (i+1, args.round, loss))
    
    feature_ = model(feature)
    
    return feature_


def train(model, features, loss, optimizer, train_dataset, num_class, epochs):
    for i in range(epochs):
        model.train()
        losses_train = 0
        for x in train_dataset:
            pro = model(features)
            y = x.T[0]-1
            a = pro[x.T[1].long()]
            loss_ = loss(a, y.long())
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()
            losses_train += loss_.cpu().item()

        eval_train = eval(pro, train_dataset, num_class)
       
        print("In epoch: %d, losses: %.4f, acc_train:%.4f, f1_train:%.4f, ,auc_train:%.4f" 
              % (i+1, losses_train, eval_train[0],  eval_train[1], eval_train[2]))

    return model


def params():
    args = argparse.ArgumentParser()

    # parameters for topological denoising
    args.add_argument("--Nk", type=int, default=18, help="Size of neighborhood in topological denoising.")
    args.add_argument("--alpha", type=int, default=0.5, help="Regularization parameter for restart topological denoising.")

    # parameters for feature refinement
    args.add_argument("--round", type=float, default=10, help="Number of epochs for supervised contrastive learning")
    args.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for GCNs")
    args.add_argument('--slr', type=float, default=0.05, help='Learning rate for supervised contrastive learning.')
    args.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay in supervised contrastive learning.")
    args.add_argument("--tau", type=float, default=0.5, help="Temperature for supervised contrastive loss")

    # paramters for semi-supervised cell-type annotation
    args.add_argument("--hidden", type=int, default=100, help="Dimension of the hidden layer for the two-layer GCN")
    args.add_argument("--glr", type=float, default=0.002, help='Learning rate for training cell-type annotation GCN.')
    args.add_argument("--epoch", type=float, default=100, help='Number of epochs for training cell-type annotation GCN')
    args.add_argument("--batch_size", type=float, default=100, help="Batch size for training cell-type annotation GCN.")
    args.add_argument("--dir", type=str, default="Prediction", help="Directory for output")
    args = args.parse_args()
    return args



if __name__=="__main__":

    seed_torch()
    args = params()
    data = loadmat("seqdata.mat")

    adj = data["similarity"]
    annotation = data["annotation"].ravel()
    feature = data["feature"]

    break_idx = np.where(annotation==-1)[0][0]
    labels = annotation[:break_idx]
    
    feat_dim = feature.shape[1]
    num_class = len(np.unique(labels))

    print("#"*20, "Pseudo-label with Topological Denoising", "#"*20)
    adj = enhancement.network_enhancement(adj, 2, 18, 0.5) 
    pseudo_labels = knn_similarity(labels, adj, 1)

    adj = torch.from_numpy(adj)
    adj = adj.to(torch.float32)
    
    print("#"*40, "Feature Refinement", "#"*40)
    feature = torch.from_numpy(feature)
    pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.int32)
    
    premodel = preprocessor(feat_dim, args.dropout, adj)
    optimizer1 = torch.optim.SGD(params=premodel.parameters(), lr=args.slr, weight_decay=args.weight_decay)
    refined_feature = pretrain(args, premodel, feature, pseudo_labels, contrastive_loss, optimizer1).detach()


    print("#"*40, "Train GCN for Cell-type Annotation", "#"*40)
    train_idx = np.arange(break_idx)
    train_data = np.vstack((labels, train_idx))
    train_data = torch.tensor(train_data.T, dtype=torch.float)
    train_dataset = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    
    net = GCN(feat_dim, args.hidden, num_class, args.dropout, adj)
    optimizer2 = torch.optim.Adam(params=net.parameters(), lr=args.glr)
    loss = nn.CrossEntropyLoss(reduction="mean")
    opt_model = train(net, refined_feature, loss, optimizer2, train_dataset, num_class, args.epoch)


    print("#"*40, "Cell-type Annotation", "#"*40)
    pro = opt_model(refined_feature)
    unannotated_idx = np.arange(len(labels), len(annotation))
    to_anno_data = torch.tensor(unannotated_idx, dtype=torch.float)
    to_anno_dataset = DataLoader(to_anno_data, batch_size=args.batch_size, shuffle=False)
    prediction_ = make_prediction(pro, to_anno_dataset)
    prediction = pd.DataFrame(prediction_, columns=["sampleID", "predictedLabel"])
    
    if not os.path.exists(args.dir):
         os.makedirs(args.dir)
    prediction.to_csv(args.dir + "/" + "make_prediction.csv", index=False)
    torch.save(opt_model.state_dict(), args.dir + "/" + "opt_model.pt")