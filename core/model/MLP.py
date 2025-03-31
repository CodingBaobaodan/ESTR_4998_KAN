import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, indicators_list_01, num_features: int, eps=1e-5, affine=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.indicators_list_01 = indicators_list_01

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
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.min_val = torch.amin(x, dim=dim2reduce, keepdim=True).detach()
        self.max_val = torch.amax(x, dim=dim2reduce, keepdim=True).detach()

    def _normalize(self, x):
        x = (x - self.min_val) / (self.max_val - self.min_val + self.eps)
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x
    
    def _denormalize(self, x):
        col_index = sum(self.indicators_list_01[:-14]) + 4 - 1

        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)

        closing_min = self.min_val[..., col_index:col_index+1]
        closing_max = self.max_val[..., col_index:col_index+1]
        
        # Reverse the normalization: x = normalized_value * (max - min) + min.
        x = x * (closing_max - closing_min) + closing_min

        # If the last dimension is larger than 1, slice to keep only the closing price column.
        if x.shape[-1] > 1:
            x = x[..., col_index:col_index+1]

        return x

    def set_statistics(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val


class MLP(nn.Module):
    def __init__(self,pred_len, var_num, drop, revin_affine, indicators_list_01):
        super(MLP, self).__init__()
        self.pred_len = pred_len
        self.var_num = var_num
        self.drop = drop
        self.indicators_list_01 = indicators_list_01
        
        self.fc1 = nn.Linear(self.var_num, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 8)
        self.fc4 = nn.Linear(8, 6)
        self.fc5 = nn.Linear(6, 5)
        self.fc6 = nn.Linear(5, 3)
        self.final_layer = nn.Linear(3, 1)
        
        self.dropout = nn.Dropout(self.drop)
        self.rev = RevIN(self.indicators_list_01, self.var_num, affine=revin_affine)

    def forward(self, var_x, marker_x):
        var_x = var_x[..., 0]  # x: [B, Li, N]
        B, L, N = var_x.shape
        var_x = self.rev(var_x, 'norm') if self.rev else var_x
        
        #var_x = var_x.view(B * L, -1)  # Shape: [B * L, input_dim]

        x = F.relu(self.fc1(var_x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.fc6(x)

        #x = x.view(B, L, -1)  # Shape: [B, L, 3]
        final_out = x[:, -1, :]  # Take the last time step for prediction

        prediction = self.final_layer(final_out)  # Shape: [B, 1]
        prediction = self.rev(prediction, 'denorm')

        confidence = torch.ones(size=(B, self.pred_len, 1))

        return prediction, confidence