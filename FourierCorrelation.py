# coding=utf-8
# author: maziqing
# email: maziqing.mzq@alibaba-inc.com

import numpy as np
import torch
import torch.nn as nn


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    Select modes in the frequency domain:
    'random' - randomly samples the modes.
    'else' - selects the lowest modes.
    """
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        indices = list(range(seq_len // 2))
        np.random.shuffle(indices)
        indices = indices[:modes]
    else:
        indices = list(range(modes))
    indices.sort()
    return indices


# ########## Fourier Block #############
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        print('Fourier enhanced block used!')

        # Get modes in the frequency domain
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        print(f'modes={modes}, index={self.index}')

        self.n_heads = n_heads
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(n_heads, in_channels // n_heads, out_channels // n_heads, len(self.index), dtype=torch.float)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(n_heads, in_channels // n_heads, out_channels // n_heads, len(self.index), dtype=torch.float)
        )

    # Complex multiplication
    def compl_mul1d(self, order, x, weights):
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x).to(x.device))
        if not torch.is_complex(weights):
            weights = torch.complex(weights, torch.zeros_like(weights).to(weights.device))
        return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag),
                             torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))

    def forward(self, q, k, v, mask):
        # Input size: [B, L, H, E]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)

        # Perform Fourier operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            if i >= x_ft.shape[3] or wi >= out_ft.shape[3]:
                continue
            out_ft[:, :, :, wi] = self.compl_mul1d("bhi,hio->bho", x_ft[:, :, :, i], torch.complex(self.weights1, self.weights2)[:, :, :, wi])

        # Return to the time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x, None


# ########## Fourier Cross Attention ####################
class FourierCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random',
                 activation='tanh', policy=0, num_heads=8):
        super(FourierCrossAttention, self).__init__()
        print('Fourier enhanced cross attention used!')

        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Get modes for queries and keys/values in the frequency domain
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        print(f'modes_q={len(self.index_q)}, index_q={self.index_q}')
        print(f'modes_kv={len(self.index_kv)}, index_kv={self.index_kv}')

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(num_heads, in_channels // num_heads, out_channels // num_heads, len(self.index_q), dtype=torch.float)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(num_heads, in_channels // num_heads, out_channels // num_heads, len(self.index_q), dtype=torch.float)
        )

    # Complex multiplication
    def compl_mul1d(self, order, x, weights):
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x).to(x.device))
        if not torch.is_complex(weights):
            weights = torch.complex(weights, torch.zeros_like(weights).to(weights.device))
        return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag),
                             torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))

    def forward(self, q, k, v, mask):
        # Input size: [B, L, H, E]
        B, L, H, E = q.shape

        xq, xk, xv = q.permute(0, 2, 3, 1), k.permute(0, 2, 3, 1), v.permute(0, 2, 3, 1)

        # Compute Fourier coefficients for queries
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            if j < xq_ft.shape[3]:
                xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]

        # Compute Fourier coefficients for keys
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            if j < xk_ft.shape[3]:
                xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # Perform attention in frequency domain
        xqk_ft = self.compl_mul1d("bhex,bhey->bhxy", xq_ft_, xk_ft_)
        if self.activation == 'tanh':
            xqk_ft = torch.complex(xqk_ft.real.tanh(), xqk_ft.imag.tanh())
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise ValueError(f"{self.activation} activation function is not implemented")

        xqkv_ft = self.compl_mul1d("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = self.compl_mul1d("bhex,heox->bhox", xqkv_ft, torch.complex(self.weights1, self.weights2))

        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            if i < xqkvw.shape[3] and j < out_ft.shape[3]:
                out_ft[:, :, :, j] = xqkvw[:, :, :, i]

        # Return to the time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        return out, None
