import numpy as np




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
