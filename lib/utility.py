import torch
import numpy as np

def diag_to_vector(m):
    assert m.shape[-1] == m.shape[-2] # make sure it's square matrix
    m[m == float("Inf")] = 0
    return m.sum(-1)

def gaussian_log_likelihood(x, mu, sigma):
    k = x.shape[-1]
    diff = x-mu
    mse = 0.5 * (diff.unsqueeze(1).bmm(torch.inverse(sigma)).bmm(diff.unsqueeze(-1))).squeeze()
    const = (k*torch.log(torch.ones(1)*2*np.pi)).to(x.device)
    sigma_det = 0.5 * torch.log(torch.det(sigma))

    return -(mse+const+sigma_det)
