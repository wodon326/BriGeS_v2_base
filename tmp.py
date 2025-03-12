import torch
from moe import MoE
import numpy as np
# tensor = torch.randn(1, 100, 20, 30)
tensor = torch.randn(4, 8, 20, 30).flatten(start_dim=1)
print(tensor.shape)

print(tensor)

model = MoE(input_size=8*20*30, num_experts=8, k=4, noisy_gating=True)
loss_1, top_k_indices_1 = model(tensor)
print(top_k_indices_1)
sorted_tensor = np.sort(top_k_indices_1.numpy(), axis=1)
print(sorted_tensor)
for i in sorted_tensor:
    for j in i:
        print(j,end=' ')

    print()