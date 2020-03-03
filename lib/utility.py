import torch

def diag_to_vector(m):
    assert m.shape[-1] == m.shape[-2] # make sure it's square matrix
    m[m == float("Inf")] = 0
    return m.sum(-1)