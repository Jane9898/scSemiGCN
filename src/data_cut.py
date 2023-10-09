import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset
#from PIL import Image
# from torchvision import transforms
import torchvision.transforms as T



class trian_val_test:
    def __init__(self,sample,train_frac,test_frac):
        self.sample = sample
        self.train_frac = train_frac
        self.test_frac = test_frac
        self.label = sample[0]

    @staticmethod
    def generate_trainval_test(sample,train_frac):
        label =sample[1]
        sample_num = len(sample[0])
        idx = np.arange(sample_num)
        sample[1] = sample[1].ravel()
        train_idx,test_val_idx = train_test_split(idx,test_size=train_frac,shuffle=True,stratify = label)

        
        train_idx_ = np.append(train_idx, 206, axis=None)
        test_val_idx_ = np.delete(test_val_idx, np.where(test_val_idx == 206))
      
        train_idx_1 = np.append(train_idx_, 0, axis=None)
        test_val_idx_1 = np.delete(test_val_idx_, np.where(test_val_idx == 0))

        test_val_sample = [sample[0][idx] for idx in test_val_idx]
        test_val_label = [sample[1][idx] for idx in test_val_idx]
        train_sample = [sample[0][idx] for idx in train_idx]
        train_label = [sample[1][idx] for idx in train_idx]

        return train_sample,test_val_sample,train_label,test_val_label,train_idx,test_val_idx

    @staticmethod
    def generate_trainval_test_1(sample,label, test_frac):
        sample_num = len(sample)
        idx = np.arange(sample_num)

        test_idx, val_idx = train_test_split(idx, test_size=test_frac, shuffle=True,stratify = label)
        test_sample = [sample[idx] for idx in test_idx]
        test_label = [label[idx] for idx in test_idx]
        val_sample = [sample[idx] for idx in val_idx]
        val_label = [label[idx] for idx in val_idx]
        return test_sample, val_sample,test_label,val_label,test_idx, val_idx

    def generate_train_val(self):
        sample = self.sample
        train_frac = self.train_frac
        test_frac = self.test_frac
        train_sample,test_val_sample,train_label,test_val_label,train_idx,test_val_idx = self.generate_trainval_test(sample,train_frac)
        train_sample = [ a.tolist() for a in train_sample]
        test_val_sample = [a.tolist() for a in test_val_sample]
        test_sample,val_sample,test_label,val_label,test_idx,val_idx = self.generate_trainval_test_1(test_val_sample,test_val_label,test_frac)

        train_sample = [tuple(a) for a in train_sample]
        val_sample = [tuple(a) for a in val_sample]
        test_sample = [tuple(a) for a in test_sample]

        train_dataset = list(zip(train_label,train_idx))
        val_dataset = list(zip(val_label,test_val_idx[val_idx]))
        test_dataset = list(zip(test_label,test_val_idx[test_idx]))

        #train_sample,val_sample,test_sample,train_label,val_label,test_label,train_val_idx[train_idx],train_val_idx[val_idx],test_idx
        train_dataset = np.array(train_dataset)
        train_dataset = torch.tensor(train_dataset,dtype=torch.float)

        val_dataset = np.array(val_dataset)
        val_dataset = torch.tensor(val_dataset,dtype=torch.float)

        test_dataset = np.array(test_dataset)
        test_dataset = torch.tensor(test_dataset,dtype=torch.float)


        return train_dataset,val_dataset,test_dataset,train_idx,test_val_idx[val_idx],test_val_idx[test_idx]

