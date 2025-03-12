import torch
from moe import MoE

tensor = torch.randn(1, 4, 20, 30)
flatten_tensor = tensor.view(1, 4, 20*30)
print(flatten_tensor.shape)

model = MoE(input_size=20*30, num_experts=4, k=2, noisy_gating=True)
loss, top_k_indices = model(flatten_tensor)
print(loss)
print(top_k_indices)
