import torch

# Original tensor
tensor = torch.tensor([[1, 2],
                       [3, 4]])

# Repeat each element along both dimensions
upsampled_tensor = tensor.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

print(upsampled_tensor)