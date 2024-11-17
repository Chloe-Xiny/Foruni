import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
import math
from functools import partial
from torch import nn, einsum
from sympy import Poly, legendre, Symbol, chebyshevt
from scipy.special import eval_legendre


# Helper function to compute Legendre derivative
def legendre_derivative(order, x):
    def compute_legendre(order, x):
        return (2 * order + 1) * eval_legendre(order, x)

    result = 0
    for i in range(order - 1, -1, -2):
        result += compute_legendre(i, x)
    return result


# Polynomial evaluation with boundary masking
def evaluate_polynomial(coeffs, x, lower_bound=0, upper_bound=1):
    mask = np.logical_or(x < lower_bound, x > upper_bound) * 1.0
    return np.polynomial.polynomial.Polynomial(coeffs)(x) * (1 - mask)


# Generate scaling and wavelet functions
def get_scaling_and_wavelets(order, base):
    x = Symbol('x')
    phi_coeffs, phi_2x_coeffs = np.zeros((order, order)), np.zeros((order, order))

    if base == 'legendre':
        for i in range(order):
            coeffs = Poly(legendre(i, 2 * x - 1), x).all_coeffs()
            phi_coeffs[i, :i + 1] = np.flip(np.sqrt(2 * i + 1) * np.array(coeffs, dtype=np.float64))
            coeffs = Poly(legendre(i, 4 * x - 1), x).all_coeffs()
            phi_2x_coeffs[i, :i + 1] = np.flip(np.sqrt(2) * np.sqrt(2 * i + 1) * np.array(coeffs, dtype=np.float64))

        psi1_coeffs, psi2_coeffs = np.zeros((order, order)), np.zeros((order, order))
        # Compute orthogonal wavelets
        for i in range(order):
            psi1_coeffs[i] = phi_2x_coeffs[i]
            for j in range(order):
                overlap = np.convolve(phi_2x_coeffs[i, :i + 1], phi_coeffs[j, :j + 1])
                projection = np.dot(overlap, 1 / (np.arange(len(overlap)) + 1) * np.power(0.5, 1 + np.arange(len(overlap))))
                psi1_coeffs[i] -= projection * phi_coeffs[j]
                psi2_coeffs[i] -= projection * phi_coeffs[j]
        phi = [np.poly1d(np.flip(phi_coeffs[i])) for i in range(order)]
        psi1 = [np.poly1d(np.flip(psi1_coeffs[i])) for i in range(order)]
        psi2 = [np.poly1d(np.flip(psi2_coeffs[i])) for i in range(order)]

    elif base == 'chebyshev':
        # Similar process for Chebyshev polynomials
        # Initialization omitted for brevity
        pass

    return phi, psi1, psi2


# Construct filter coefficients
def construct_filters(base, order):
    phi, psi1, psi2 = get_scaling_and_wavelets(order, base)
    H0, H1, G0, G1 = np.zeros((order, order)), np.zeros((order, order)), np.zeros((order, order)), np.zeros((order, order))
    PHI0, PHI1 = np.eye(order), np.eye(order)

    # Generate filter banks for Legendre or Chebyshev bases
    if base == 'legendre':
        roots = Poly(legendre(order, 2 * Symbol('x') - 1)).all_roots()
        nodes = np.array([root.evalf() for root in roots], dtype=np.float64)
        weights = 1 / order / legendre_derivative(order, 2 * nodes - 1) / eval_legendre(order - 1, 2 * nodes - 1)
        # Compute H0, H1, G0, G1 here (omitted for brevity)

    return H0, H1, G0, G1, PHI0, PHI1


# MultiWavelet Transform class
class MultiWaveletTransform(nn.Module):
    def __init__(self, input_channels=1, order=8, alpha=16, hidden_dim=128, num_cz_layers=1, L=0, base='legendre'):
        super().__init__()
        self.order = order
        self.hidden_dim = hidden_dim
        self.num_cz_layers = num_cz_layers
        self.linear_in = nn.Linear(input_channels, hidden_dim * order)
        self.linear_out = nn.Linear(hidden_dim * order, input_channels)
        self.cz_layers = nn.ModuleList(MWT_CZ1d(order, alpha, L, hidden_dim, base) for _ in range(num_cz_layers))

    def forward(self, queries, keys, values, mask):
        # Implementation omitted for brevity
        pass


# MultiWavelet Cross Attention class
class MultiWaveletCrossAttention(nn.Module):
    # Implementation details omitted for brevity
    pass


# Sparse Kernel Fourier Transform class
class SparseKernelFourierTransform1d(nn.Module):
    # Implementation details omitted for brevity
    pass


# MultiWavelet Coarse-to-Zero Layer class
class MWT_CZ1d(nn.Module):
    # Implementation details omitted for brevity
    pass
