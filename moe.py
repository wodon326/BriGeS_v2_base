import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class MoE(nn.Module):
    def __init__(self, input_size, num_experts, moe_num, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.k = k
        self.moe_num = moe_num
        self.flatten = nn.Flatten(start_dim=1)
        self.w_gate = nn.Parameter(torch.zeros(moe_num, input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(moe_num, input_size, num_experts), requires_grad=True)
        self.noise_epsilon = 1e-2
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=2)
        
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def forward(self, x, absolute_feature, loss_coef=1e-2):
        B, C, H, W = x.shape
        x_flatten = self.flatten(x)

        # 텐서 초기화
        #new_absolute_index = torch.zeros((B, self.moe_num * self.k), dtype=torch.int32, device=self.DEVICE)

        # x_flatten을 moe_num 개로 분할하여 벡터화된 연산 준비
        select_x_splits = x_flatten.view(B, self.moe_num, self.input_size)  # (B, moe_num, input_size)

        # 벡터화된 연산 수행
        clean_logits = torch.einsum('bmi,min->bmn', select_x_splits, self.w_gate)

        if self.noisy_gating and self.training:
            raw_noise_stddev = torch.einsum('bmi,min->bmn', select_x_splits, self.w_noise)
            noise_stddev = self.softplus(raw_noise_stddev) + self.noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits, device=self.DEVICE) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=2)
        top_k_logits = top_logits[:, :, :self.k]
        top_k_indices = top_indices[:, :, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(2, keepdim=True) + 1e-6)  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True, device=self.DEVICE)
        gates = zeros.scatter(2, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and self.training:
            load = self._prob_in_top_k(clean_logits.flatten(start_dim=1), noisy_logits.flatten(start_dim=1), noise_stddev.flatten(start_dim=1), top_logits.flatten(start_dim=1)).sum(0)
        else:
            load = self._gates_to_load(gates)

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        # new_absolute_index와 top_k_channel 계산
        top_k_indices_flat = top_k_indices.reshape(B, -1)
        top_k_absolute_feature = torch.stack([absolute_feature[b, top_k_indices_flat[b] + torch.arange(self.moe_num, device=self.DEVICE).repeat_interleave(self.k) * self.num_experts] for b in range(B)], dim=0)
        top_k_adaptive_channel = torch.stack([x[b, top_k_indices_flat[b] + torch.arange(self.moe_num, device=self.DEVICE).repeat_interleave(self.k) * self.num_experts] for b in range(B)], dim=0)
        
        gates_expanded = top_k_gates.reshape(B, -1).unsqueeze(2).unsqueeze(3)
        weighted_top_k_adaptive_channel = top_k_adaptive_channel * gates_expanded
        weighted_top_k_absolute_feature = top_k_absolute_feature * gates_expanded
        return loss, weighted_top_k_absolute_feature, weighted_top_k_adaptive_channel