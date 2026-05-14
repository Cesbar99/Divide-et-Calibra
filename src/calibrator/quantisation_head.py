import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from omegaconf import ListConfig

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQHeadEMA(nn.Module):
    """
    Hard VQ per slot with STE in backward pass and EMA codebook updates.

    Input:  z        (B, S, d)
    Output: z_q_st   (B, S, d)   # quantized in forward, identity in backward
            indices  (B, S)
    """
    def __init__(self, K: int, d: int, L1: bool = False, decay: float = 0.99, eps: float = 1e-8):
        super().__init__()
        self.K = K
        self.d = d
        self.decay = decay
        self.eps = eps
        self.L1 = L1        

        # Codebook embeddings (updated by EMA, not gradients)
        embed = torch.randn(K, d)
        self.register_buffer("codebook", embed)

        # EMA state
        self.register_buffer("cluster_size", torch.zeros(K))
        self.register_buffer("embed_avg", torch.zeros(K, d))        

    @torch.no_grad()
    def init_codebook_from_samples(self, samples: torch.Tensor):
        """
        samples: (N, d) slot vectors used to initialize codebook
        Strategy: random subset of samples (simple baseline).
        You can replace with k-means if you want.
        """
        assert samples.dim() == 2 and samples.size(1) == self.d
        N = samples.size(0)
        idx = torch.randperm(N, device=samples.device)[: self.K]
        self.codebook.copy_(samples[idx])
        self.cluster_size.zero_()
        self.embed_avg.zero_()
    
    @torch.no_grad()
    def init_codebook_from_samples_std(self, samples: torch.Tensor):
        """
        samples: (N, d) slot vectors used to initialize codebook

        Standardizes samples before initializing codebook.
        Stores mean/std for later use in forward.
        """
        assert samples.dim() == 2 and samples.size(1) == self.d

        # ---- compute dataset statistics ----
        self.register_buffer("feat_mean", samples.mean(dim=0))
        self.register_buffer("feat_std", samples.std(dim=0).clamp_min(1e-6))

        # ---- standardize ----
        samples_std = (samples - self.feat_mean) / self.feat_std

        # ---- random subset initialization ----
        N = samples_std.size(0)
        idx = torch.randperm(N, device=samples.device)[: self.K]
        self.codebook.copy_(samples_std[idx])

        # ---- reset EMA state ----
        self.cluster_size.zero_()
        self.embed_avg.zero_()
            
    # @torch.no_grad()
    # def init_codebook_kmeanspp(self, samples: torch.Tensor, max_samples: int = 200000):
    #     """
    #     k-means++ initialization for codebook.
    #     samples: (N, d)
    #     """
    #     assert samples.dim() == 2 and samples.size(1) == self.d

    #     # optionally subsample for speed
    #     if samples.size(0) > max_samples:
    #         idx = torch.randperm(samples.size(0), device=samples.device)[:max_samples]
    #         x = samples[idx]
    #     else:
    #         x = samples

    #     N, d = x.shape
    #     K = self.K

    #     # pick first center uniformly
    #     centers = torch.empty((K, d), device=x.device, dtype=x.dtype)
    #     first = torch.randint(0, N, (1,), device=x.device).item()
    #     centers[0] = x[first]

    #     # distances to nearest chosen center
    #     dist2 = torch.cdist(x, centers[0:1]).squeeze(1).pow(2)  # (N,)

    #     for k in range(1, K):
    #         probs = dist2 / dist2.sum().clamp_min(1e-12)
    #         idx_k = torch.multinomial(probs, 1).item()
    #         centers[k] = x[idx_k]
    #         dist2 = torch.minimum(dist2, torch.cdist(x, centers[k:k+1]).squeeze(1).pow(2))

    #     self.codebook.copy_(centers)
    #     self.cluster_size.zero_()
    #     self.embed_avg.zero_()  
    
    # @torch.no_grad()
    # def init_codebook_from_samples(self, samples: torch.Tensor, iters: int = 25, batch_size: int = 8192):
    #     """
    #     Mini-batch k-means initialization. Fast and good.
    #     samples: (N,d)
    #     """
    #     assert samples.dim() == 2 and samples.size(1) == self.d
    #     x = samples
    #     N, d = x.shape
    #     K = self.K

    #     # start from kmeans++ for stability
    #     self.init_codebook_kmeanspp(x)
    #     centers = self.codebook.clone()

    #     counts = torch.zeros(K, device=x.device, dtype=torch.float32)

    #     for _ in range(iters):
    #         idx = torch.randint(0, N, (min(batch_size, N),), device=x.device)
    #         xb = x[idx]  # (B,d)

    #         # assign
    #         dist = torch.cdist(xb, centers)  # (B,K)
    #         assign = dist.argmin(dim=1)      # (B,)

    #         # update centers with running mean
    #         for k in range(K):
    #             mask = (assign == k)
    #             if mask.any():
    #                 ck = xb[mask].mean(dim=0)
    #                 counts[k] += mask.sum()
    #                 # move center slightly toward batch mean
    #                 eta = 1.0 / counts[k].clamp_min(1.0)
    #                 centers[k] = (1 - eta) * centers[k] + eta * ck

    #     self.codebook.copy_(centers)
    #     self.cluster_size.zero_()
    #     self.embed_avg.zero_()
  

    def forward(self, z: torch.Tensor, return_entropy: bool = False, tau: float = 1.0):
        assert z.dim() == 3 and z.size(-1) == self.d, f"Expected (B,S,{self.d}), got {tuple(z.shape)}"
        B, S, d = z.shape

        # Flatten slots: (B*S, d)
        z_flat = z.reshape(B * S, d)

        # Compute distances to codewords:
        # L1: |z - c|        
        if self.L1:
            dist = torch.abs(
                z_flat.unsqueeze(1) - self.codebook.unsqueeze(0)).sum(dim=2)   # (BS, K)
        else:
            # L2: ||z - c||^2 = ||z||^2 + ||c||^2 - 2 z·c
            z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)            # (BS, 1)
            c_sq = (self.codebook ** 2).sum(dim=1).unsqueeze(0)      # (1, K)
            zc = z_flat @ self.codebook.t()                          # (BS, K)
            dist = z_sq + c_sq - 2 * zc                              # (BS, K)

        # Hard assignment
        indices = torch.argmin(dist, dim=1)                      # (BS,)
        z_q = self.codebook[indices]                             # (BS, d)
        z_q = z_q.view(B, S, d)

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        # EMA update (only when training)
        if self.training:
            self._ema_update(z_flat, indices)
            
        if return_entropy:
            # soft assignment probabilities over codewords
            probs = F.softmax(-dist / tau, dim=1)      # (BS, K)

            # entropy per slot
            entropy = -(probs * torch.log(probs + self.eps)).sum(dim=1)   # (BS,)
            entropy = entropy.view(B, S)
            
            return z_q_st, indices.view(B, S), entropy
        else:
            # Return indices as (B, S) for downstream calibration usage
            return z_q_st, indices.view(B, S)

    @torch.no_grad()
    def _ema_update(self, z_flat: torch.Tensor, indices: torch.Tensor):
        # One-hot assignments: (BS, K)
        one_hot = F.one_hot(indices, num_classes=self.K).type_as(z_flat)

        # Per-codeword counts and sums
        counts = one_hot.sum(dim=0)                               # (K,)
        sums = one_hot.t() @ z_flat                               # (K, d)

        # EMA
        self.cluster_size.mul_(self.decay).add_(counts * (1 - self.decay))
        self.embed_avg.mul_(self.decay).add_(sums * (1 - self.decay))

        # Normalize to get new codebook
        denom = (self.cluster_size + self.eps).unsqueeze(1)
        self.codebook.copy_(self.embed_avg / denom)

    @torch.no_grad()
    def codeword_usage(self, indices_bs: torch.Tensor):
        """indices_bs: (B,S) or (BS,)"""
        idx = indices_bs.reshape(-1)
        return torch.bincount(idx, minlength=self.K)


class QuantizedClassifierHead(nn.Module):
    def __init__(self, S: int, d: int, num_classes: int, hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        in_dim = S * d
        layers = []
        if hidden and hidden > 0:
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            layers += [nn.Linear(hidden, num_classes)]
        else:
            layers += [nn.Linear(in_dim, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, z_q_st: torch.Tensor):
        B, S, d = z_q_st.shape
        return self.net(z_q_st.reshape(B, S * d))

