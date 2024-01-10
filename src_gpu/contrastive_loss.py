import numpy as np
import torch
import torch.nn.functional as F


def block_cosine_similarity(x,batch_size):
    n = len(x)
    m = n//batch_size

    for i in range(1,m+2):

        if i<=m:
            for j in range(1,m+2):

                if j == m+1:
                    cur = F.cosine_similarity(x[(i-1)*batch_size:i*batch_size,:].unsqueeze(1), x[(j-1)*batch_size:,:].unsqueeze(0), dim=2)
                else:
                    cur = F.cosine_similarity(x[(i-1)*batch_size:i*batch_size,:].unsqueeze(1), x[(j-1)*batch_size:j*batch_size,:].unsqueeze(0), dim=2)


                if j ==1:
                    sum = cur
                else:
                    sum =torch.cat((sum, cur), 1)

            if i ==1:
                sum_matrix = sum
            else:
                sum_matrix = torch.cat((sum_matrix, sum), 0)

        if i==m+1:

            for j in range(1, m + 2):

                if j == m + 1:
                    cur = F.cosine_similarity(x[(i-1) * batch_size:, :].unsqueeze(1),
                                              x[(j-1) * batch_size:, :].unsqueeze(0), dim=2)
                else:
                    cur = F.cosine_similarity(x[(i-1) * batch_size:, :].unsqueeze(1),
                                              x[(j - 1) * batch_size:j * batch_size, :].unsqueeze(0), dim=2)
                if j == 1:
                    sum = cur
                else:
                    sum = torch.cat((sum, cur), 1)

            sum_matrix = torch.cat((sum_matrix, sum), 0)

    return sum_matrix


def contrastive_loss(representations, label, T):
    n = label.shape[0]  
    if n >= 500:
        similarity_matrix = block_cosine_similarity(representations, 500)
    else:
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))

    mask_no_sim = torch.ones_like(mask) - mask
    mask_diagonal = torch.ones(n, n) - torch.eye(n, n)
    mask_diagonal = mask_diagonal.cuda()
    similarity_matrix = torch.exp(similarity_matrix / T)
    similarity_matrix = similarity_matrix * mask_diagonal

    sim = mask * similarity_matrix

    no_sim = similarity_matrix - sim

    no_sim_sum = torch.sum(no_sim, dim=1)


    no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
    sim_sum = sim + no_sim_sum_expend
    loss = torch.div(sim, sim_sum)


    loss = mask_no_sim + loss + torch.eye(n, n).cuda()
    loss = -torch.log(loss)  
    loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))

    return loss

