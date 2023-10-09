import numpy as np

import torch
import torch.nn.functional as F

# def loss(X, labels, tau):
#     # X should be normalize
#     labels = torch.squeeze(labels, 0)
#     n = len(labels)
#     labels_repeat = labels.repeat(n,1)
#     indicator = torch.eq(labels_repeat, labels_repeat.T).float() - torch.eye(X.size(0))
#     sim = torch.matmul(X, X.T) / tau
#     offd = torch.ones_like(sim) - torch.eye(X.shape[0])
#     sim = sim*offd
#     sim_max = torch.max(sim)
#     sim = sim-sim_max.detach()
#     exp_sim = torch.exp(sim)


#     pos_sim = sim * indicator
#     denominator = torch.sum(exp_sim * offd)
    
#     loss = -((pos_sim - torch.log(denominator))/indicator.sum(dim=1, keepdim=True)).sum()
#     loss = loss/len(labels)
#     return loss

def contrastive_loss(representations,label,T):
    """
    adapted from Chen, L., Wang, F., Yang, R. et al. Representation learning from noisy user-tagged data for 
    sentiment classification.  Int. J. Mach. Learn. & Cyber. 13, 3727–3742 (2022). 
    """
    n = label.shape[0]  
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))
    mask_no_sim = torch.ones_like(mask) - mask


    mask_diag = torch.ones(n, n) - torch.eye(n, n)
    similarity_matrix = torch.exp(similarity_matrix / T)
    similarity_matrix = similarity_matrix * mask_diag
    sim = mask * similarity_matrix
    
    no_sim = similarity_matrix - sim
    no_sim_sum = torch.sum(no_sim, dim=1)


    no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
    sim_sum = sim + no_sim_sum_expend
    loss = torch.div(sim, sim_sum)


    loss = mask_no_sim + loss + torch.eye(n, n)

    loss = -torch.log(loss)  
    loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss))) 

    return loss

'''

if __name__=="__main__":
    X  = np.random.rand(4, 10)
    labs = np.array([1, 1, 2, 2])
    X = torch.tensor(X)
    labs = torch.tensor(labs)
    tau = 0.07
    
    normX = X / torch.norm(X, dim=1, keepdim=True)

    con_loss = contrastive_loss(X, labs, tau)
    print("Contrastive loss: %.3f" % con_loss)
'''