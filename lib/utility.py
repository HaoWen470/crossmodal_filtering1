import torch

def diag_to_vector(m):
    assert m.shape[-1] == m.shape[-2] # make sure it's square matrix
    assert len(m.shape) == 3 or len(m.shape) == 2 # make sure it's N, *, *

    if len(m.shape) == 3:
        N, i, _ = m.shape
        vector = [m[:, i, i] for i in range(m.shape[-1])]
        vector_torch = torch.stack(vector).transpose(0, 1)
        assert vector_torch.shape == (N, i)
    else:
        i, _ = m.shape
        vector = [m[i, i] for i in range(m.shape[-1])]
        vector_torch = torch.stack(vector)

        assert vector_torch.shape == (i, )

    return vector_torch.to(m.device)