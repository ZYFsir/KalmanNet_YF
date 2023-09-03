import torch

def expand_to_batch(tensor_2d, batch_size):
    # tensor_2d: 2D input tensor of shape (N, d)
    # batch_size: Desired batch size

    # Expand the tensor along the first dimension
    tensor_3d = torch.unsqueeze(tensor_2d, dim=0).expand(batch_size, -1, -1)

    return tensor_3d

def move_to_cuda(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {key: move_to_cuda(value, device) for key, value in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [move_to_cuda(item, device) for item in batch]
    else:
        return batch

def state_detach(state):
    state_after_detach = {}
    for k, v in state.items():
        state_after_detach[k] = v.detach()
        state_after_detach[k].requires_grad = True
    return state_after_detach