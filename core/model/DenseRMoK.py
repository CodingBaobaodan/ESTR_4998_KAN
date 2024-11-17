import torch
import torch.nn as nn
import torch.nn.functional as F

from core.layer.kanlayer import ChebyKANLayer, JacobiKANLayer, MoKLayer, RBFKANLayer, TaylorKANLayer, WaveKANLayer
from core.layer.embedding import PatchEmbedding



class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)

        elif mode == 'denorm':
            x = self._denormalize(x)

        else:
            raise NotImplementedError

        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean

        return x
    
    # Modified code
    def set_statistics(self, mean, stdev):
        """
        Manually set the mean and standard deviation for denormalization.

        Args:
            mean (torch.Tensor): Mean tensor of shape [1, 1, 1].
            stdev (torch.Tensor): Standard deviation tensor of shape [1, 1, 1].
        """
        self.mean = mean
        self.stdev = stdev


class DenseRMoK(nn.Module):
    def __init__(self, hist_len, pred_len, var_num, num_experts=4, drop=0.1, revin_affine=True):
        super(DenseRMoK, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.var_num = var_num
        self.num_experts = num_experts
        self.drop = drop
        self.revin_affine = revin_affine

        #self.gate = nn.Linear(hist_len, num_experts)
        # Modified code
        self.gate = nn.Linear(hist_len, num_experts)

        self.softmax = nn.Softmax(dim=-1)
        self.experts = nn.ModuleList([
            TaylorKANLayer(hist_len, pred_len, order=3, addbias=True),
            TaylorKANLayer(hist_len, pred_len, order=3, addbias=True),
            WaveKANLayer(hist_len, pred_len, wavelet_type="mexican_hat", device="cuda"),
            WaveKANLayer(hist_len, pred_len, wavelet_type="mexican_hat", device="cuda"),
        ])
        # Modified code

        # self.experts = nn.ModuleList([
        #     TaylorKANLayer(hist_len * var_num, pred_len, order=3, addbias=True),
        #     TaylorKANLayer(hist_len * var_num, pred_len, order=3, addbias=True),
        #     WaveKANLayer(hist_len * var_num, pred_len, wavelet_type="mexican_hat", device="cuda"),
        #     WaveKANLayer(hist_len * var_num, pred_len, wavelet_type="mexican_hat", device="cuda"),
        # ])

        # self.experts = nn.ModuleList([
        #     TaylorKANLayer(hist_len * var_num, pred_len, order=3, addbias=True),
        #     JacobiKANLayer(hist_len * var_num, pred_len, degree=6),  # Replaced WaveKANLayer with JacobiKANLayer
        #     WaveKANLayer(hist_len * var_num, pred_len, wavelet_type="mexican_hat", device="cuda"),
        #     JacobiKANLayer(hist_len * var_num, pred_len, degree=6),  # Another JacobiKANLayer
        # ])


        # self.experts = nn.ModuleList([
        #     TaylorKANLayer(hist_len * var_num, pred_len, order=3, addbias=True),
        #     ChebyKANLayer(hist_len * var_num, pred_len, degree=4),      # Added ChebyKANLayer
        #     nn.Linear(in_features=hist_len * var_num, out_features=pred_len),  # Added Linear Layer
        #     RBFKANLayer(hist_len * var_num, pred_len, num_centers=10)   # Added RBFKANLayer
        # ])

        # self.experts = nn.ModuleList([
        #     nn.Linear(hist_len * var_num, pred_len),
        #     nn.Linear(hist_len * var_num, pred_len),
        #     nn.Linear(hist_len * var_num, pred_len),
        #     nn.Linear(hist_len * var_num, pred_len),
        # ])


        self.dropout = nn.Dropout(drop)
        self.rev = RevIN(var_num, affine=revin_affine)
        # Modified code
        self.rev_output = RevIN(num_features=1, affine=revin_affine)  # For output denormalization

    def forward(self, var_x, marker_x):
        #print("input paramter : ", var_x.shape)
        var_x = var_x[..., 0]  # x: [B, Li, N]
        #print("input paramter : ", var_x.shape)

        B, L, N = var_x.shape

        var_x = self.rev(var_x, 'norm') if self.rev else var_x

        var_x = self.dropout(var_x).transpose(1, 2).reshape(B * N, L)

        #print("input paramter : ", var_x.shape)

        score = F.softmax(self.gate(var_x), dim=-1)  # (BxN, E)


        expert_outputs = torch.stack([self.experts[i](var_x) for i in range(self.num_experts)], dim=-1)  # (BxN, Lo, E)

        prediction = torch.einsum("BLE,BE->BL", expert_outputs, score).reshape(B, N, -1).permute(0, 2, 1)
        prediction = self.rev(prediction, 'denorm')
        #print(prediction)

        return prediction
    
    # def forward(self, var_x, marker_x):
    #     # var_x: [B, L, N=4, C=1]
    #     var_x = var_x[..., 0]  # Remove the last dimension C=1
    #     # Now var_x is [B, L, N=4]

    #     B, L, N = var_x.shape  # N=4 (OHLC features)
    #     #print("Before RevIN normalization:", var_x[:, :, 3])
    #     var_x = self.rev(var_x, 'norm') if self.rev else var_x  # Apply RevIN
    #     #print("After RevIN normalization:", var_x[:, :, 3])

    #     # Reshape var_x to combine features and sequence length
    #     var_x = var_x.permute(0, 2, 1)  # [B, N=4, L]
    #     var_x = var_x.reshape(B, N * L)  # [B, N*L]

    #     var_x = self.dropout(var_x)

    #     # Update the gate and experts to accept input of dimension N*L
    #     score = F.softmax(self.gate(var_x), dim=-1)  # (B, num_experts)

    #     # Each expert processes input of size N*L and outputs pred_len predictions
    #     expert_outputs = torch.stack([self.experts[i](var_x) for i in range(self.num_experts)], dim=-1)  # (B, pred_len, num_experts)

    #     # Combine the experts' outputs using the gate scores
    #     prediction = torch.einsum("BLE,BE->BL", expert_outputs, score)  # (B, pred_len)

    #     # Since we're predicting only the Close price
    #     prediction = prediction.unsqueeze(-1)  # (B, pred_len, N=1, C=1)

    #     #print(prediction)
    #     #prediction = self.rev(prediction, 'denorm')
    #     #Modified Code
    #     prediction = self.rev_output(prediction, 'denorm')  # [B, pred_len, 1, 1]
    #     #print("asdsadasdas")
    #     #print("\n\n", prediction)

    #     return prediction  # (B, pred_len, N=1, C=1)

# class DenseRMoK(nn.Module):
#     def __init__(self, hist_len, pred_len, var_num, d_model=64, patch_len=4, stride=8, padding=4, dropout=0.1, embed_type='fixed', freq='h', device='cuda'):
#         super(DenseRMoK, self).__init__()
#         self.hist_len = hist_len
#         self.pred_len = pred_len
#         self.var_num = var_num
#         self.device = device

#         #modified code:
#         self.patch_len = patch_len
#         self.padding = padding
#         self.stride = stride

#         # Patch Embedding Layer
#         self.patch_embedding = PatchEmbedding(
#             d_model=d_model,
#             patch_len=patch_len,
#             stride=stride,
#             padding=padding,
#             dropout=dropout
#         )

#         # Gate to produce expert scores
#         self.gate = nn.Linear(hist_len * d_model, 4)
#         self.softmax = nn.Softmax(dim=-1)

#         # Define experts
#         self.experts = nn.ModuleList([
#             MoKLayer(in_features=hist_len * d_model, out_features=pred_len, experts_type="A", gate_type="Linear"),
#             MoKLayer(in_features=hist_len * d_model, out_features=pred_len, experts_type="B", gate_type="Linear"),
#             MoKLayer(in_features=hist_len * d_model, out_features=pred_len, experts_type="C", gate_type="Linear"),
#             MoKLayer(in_features=hist_len * d_model, out_features=pred_len, experts_type="V", gate_type="Linear"),
#         ])

#         self.dropout = nn.Dropout(dropout)
#         self.rev = RevIN(var_num, affine=True)
#         self.rev_output = RevIN(pred_len, affine=True)

#     def forward(self, var_x, x_mark):
#         # var_x: [B, L, N, C=1]
#         var_x = var_x[..., 0]  # Shape: [B, L, N]
#         print("Input shape before patch embedding:", var_x.shape)

#         # Ensure compatible dimensions for patch embedding
#         # Padding dynamically to ensure compatibility with patch_len and stride
#         _, seq_len, _ = var_x.shape  # seq_len = L (history length)
#         padding_needed = max(0, self.patch_len - (seq_len + self.padding) % self.stride)

#         if padding_needed > 0:
#             var_x = torch.nn.functional.pad(var_x, (0, 0, 0, padding_needed))

#         patches, n_vars = self.patch_embedding(var_x)  # [B * num_patches, patch_len, d_model], n_vars
#         print("Patches shape:", patches.shape)

#         # Reshape patches to [B, num_patches, patch_len, d_model]
#         B = var_x.shape[0]
#         num_patches = patches.shape[0] // B
#         patches = patches.view(B, num_patches, patches.shape[-2], patches.shape[-1])

#         # Aggregate patches (e.g., average or concatenate)
#         aggregated = patches.mean(dim=1).reshape(B, -1)  # [B, num_patches * patch_len * d_model]

#         # Apply dropout
#         aggregated = self.dropout(aggregated)

#         # Compute gate scores
#         score = self.softmax(self.gate(aggregated))  # [B, E]

#         # Get expert outputs
#         expert_outputs = torch.stack([expert(aggregated) for expert in self.experts], dim=-1)  # [B, pred_len, E]

#         # Combine expert outputs
#         prediction = torch.einsum("BLE,BE->BL", expert_outputs, score)  # [B, pred_len]

#         # Denormalize
#         prediction = self.rev_output(prediction.unsqueeze(-1), 'denorm')  # [B, pred_len, 1]

#         return prediction  # [B, pred_len, 1]

        
