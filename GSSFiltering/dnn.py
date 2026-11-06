import torch
import torch.nn as nn


def _module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _module_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


class OnlineKoopmanKNet(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, n_features: int):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_features = n_features
        self.mlp_hidden = 4 * n_features

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, self.mlp_hidden),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden, n_features),
        )

        self.k_input_dim = n_features + y_dim
        self.k_token_dim = self.k_input_dim
        self.k_hidden_dim = 4 * self.k_token_dim
        self.k_num_layers = 1

        self.K_in = nn.Sequential(
            nn.Linear(self.k_input_dim, self.mlp_hidden // 2),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden // 2, self.k_token_dim),
        )
        self.K_gru = nn.GRU(
            input_size=self.k_token_dim,
            hidden_size=self.k_hidden_dim,
            num_layers=self.k_num_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(self.k_hidden_dim, n_features * y_dim),
        )
        self.U = nn.Parameter(torch.eye(n_features, n_features))
        self.C = nn.Parameter(torch.eye(n_features, y_dim))
        self.B = nn.Parameter(torch.eye(x_dim, n_features))
        self.k_h = None

    def initialize_hidden(self, batch_size: int = 1, device=None):
        if device is None:
            device = _module_device(self)
        self.k_h = torch.zeros(
            self.k_num_layers,
            batch_size,
            self.k_hidden_dim,
            device=device,
            dtype=_module_dtype(self),
        )

    def forward(self, z_post: torch.Tensor, y_obs: torch.Tensor):
        device = _module_device(self)
        dtype = _module_dtype(self)
        z_post = z_post.to(device=device, dtype=dtype)
        y_obs = y_obs.to(device=device, dtype=dtype)
        
        if self.k_h is not None:
            self.k_h = self.k_h.to(device=device, dtype=dtype)

        z_pred = self.U @ z_post
        state_inova = z_post - z_pred
        y_pred = (self.C.T @ z_pred).view(-1, 1)
        innovation = y_obs - y_pred

        feat = torch.cat([state_inova, innovation], dim=0).view(1, -1)  # [1, k_input_dim]
        k_token = self.K_in(feat).unsqueeze(1)                           # [1, 1, k_token_dim]
        out, self.k_h = self.K_gru(k_token, self.k_h)
        k_last = out[:, -1, :]                                           # [1, k_hidden_dim]

        self.k_h = self.k_h.detach()

        K_gain = self.fc(k_last).view(self.n_features, self.y_dim)
        return K_gain, z_pred, y_pred