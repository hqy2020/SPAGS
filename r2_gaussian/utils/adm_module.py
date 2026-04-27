"""
ADM (Adaptive Density Modulation) Module for SPAGS.

Implements the orthogonal plane feature network + dual-head MLP decoder
for position-dependent density modulation as described in the SPAGS thesis (Chapter 4).

Key components:
1. TriPlaneFeatureNetwork: Three orthogonal 2D feature planes (XY, XZ, YZ)
2. DualHeadMLPDecoder: Shared backbone + offset/confidence heads
3. ADM modulation: rho_final = rho_base * (1 + o_offset * o_conf * r_max * s)
4. TV regularization on feature planes
5. Three-stage modulation strength scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TriPlaneFeatureNetwork(nn.Module):
    """
    Orthogonal plane feature network that encodes 3D space.
    
    Three learnable 2D feature grids (XY, XZ, YZ planes).
    For any 3D position, features are extracted via bilinear interpolation
    from each plane and concatenated.
    """
    
    def __init__(self, grid_size: int = 256, feat_dim: int = 32):
        """
        Args:
            grid_size: Spatial resolution of each feature plane (H=W=grid_size)
            feat_dim: Number of feature channels per plane
        """
        super().__init__()
        self.grid_size = grid_size
        self.feat_dim = feat_dim
        
        # Three orthogonal feature planes: (1, C, H, W)
        # Using nn.Parameter so they are learnable
        self.grid_xy = nn.Parameter(
            torch.randn(1, feat_dim, grid_size, grid_size) * 0.05
        )
        self.grid_xz = nn.Parameter(
            torch.randn(1, feat_dim, grid_size, grid_size) * 0.05
        )
        self.grid_yz = nn.Parameter(
            torch.randn(1, feat_dim, grid_size, grid_size) * 0.05
        )
    
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Extract features for 3D positions.
        
        Args:
            xyz: (N, 3) tensor of 3D positions in [-1, 1] normalized space
            
        Returns:
            features: (N, feat_dim * 3) concatenated tri-plane features
        """
        N = xyz.shape[0]
        
        # Reshape grid_sample expects (N, C, H, W) input and (N, H, W, 2) grid
        # We use a batch of 1 (same grid for all positions)
        
        # Prepare sampling coordinates: grid_sample expects [-1, 1] range
        # xyz is already in [-1, 1], reshape to (1, 1, N, 2) for grid_sample
        # grid_sample format: (x, y) where x is width (last dim), y is height (first dim)
        
        # XY plane: sample at (x, y)
        grid_xy = xyz[:, [0, 1]].view(1, 1, N, 2)  # (1, 1, N, 2) where grid is (x, y)
        feat_xy = F.grid_sample(
            self.grid_xy, grid_xy, mode='bilinear', padding_mode='border', align_corners=True
        )  # (1, C, 1, N)
        feat_xy = feat_xy.squeeze(2).squeeze(0).t()  # (N, C)
        
        # XZ plane: sample at (x, z)  (x=width, z=height in sampling coords)
        grid_xz = xyz[:, [0, 2]].view(1, 1, N, 2)
        feat_xz = F.grid_sample(
            self.grid_xz, grid_xz, mode='bilinear', padding_mode='border', align_corners=True
        )
        feat_xz = feat_xz.squeeze(2).squeeze(0).t()  # (N, C)
        
        # YZ plane: sample at (y, z)  (y=width, z=height in sampling coords)
        grid_yz = xyz[:, [1, 2]].view(1, 1, N, 2)
        feat_yz = F.grid_sample(
            self.grid_yz, grid_yz, mode='bilinear', padding_mode='border', align_corners=True
        )
        feat_yz = feat_yz.squeeze(2).squeeze(0).t()  # (N, C)
        
        # Concatenate: (N, 3*C)
        features = torch.cat([feat_xy, feat_xz, feat_yz], dim=-1)
        
        return features


class DualHeadMLPDecoder(nn.Module):
    """
    Dual-head MLP decoder that produces modulation signals.
    
    From tri-plane features, outputs:
    - o_offset ∈ [-1, 1]: direction of modulation (tanh)
    - o_conf ∈ [0, 1]: confidence/strength of modulation (sigmoid)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Args:
            input_dim: Input feature dimension (feat_dim * 3)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        
        # Dual heads
        self.offset_head = nn.Linear(hidden_dim // 2, 1)  # direction
        self.conf_head = nn.Linear(hidden_dim // 2, 1)    # confidence
    
    def forward(self, features: torch.Tensor) -> tuple:
        """
        Args:
            features: (N, input_dim) tri-plane features
            
        Returns:
            offset: (N, 1) in [-1, 1]
            confidence: (N, 1) in [0, 1]
        """
        h = self.backbone(features)  # (N, hidden_dim // 2)
        
        offset = torch.tanh(self.offset_head(h))  # [-1, 1]
        confidence = torch.sigmoid(self.conf_head(h))  # [0, 1]
        
        return offset, confidence


class ADMModule(nn.Module):
    """
    Adaptive Density Modulation module.
    
    Combines TriPlaneFeatureNetwork + DualHeadMLPDecoder to produce
    position-dependent density modulation for each Gaussian.
    """
    
    def __init__(
        self,
        grid_size: int = 256,
        feat_dim: int = 32,
        r_max: float = 0.5,
        bbox: torch.Tensor = None,
    ):
        """
        Args:
            grid_size: Spatial resolution of feature planes
            feat_dim: Feature dimension per plane
            r_max: Maximum modulation range (paper: controls amplitude)
            bbox: (2, 3) tensor with [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        """
        super().__init__()
        self.grid_size = grid_size
        self.feat_dim = feat_dim
        self.r_max = r_max
        
        # Bbox for normalizing positions to [-1, 1]
        # Will be set later when scene info is available
        if bbox is not None:
            self.register_buffer('bbox_min', bbox[0].clone())
            self.register_buffer('bbox_max', bbox[1].clone())
        else:
            self.bbox_min = None
            self.bbox_max = None
        
        # Tri-plane feature network
        self.tri_plane = TriPlaneFeatureNetwork(grid_size, feat_dim)
        
        # Dual-head MLP decoder
        self.decoder = DualHeadMLPDecoder(input_dim=feat_dim * 3, hidden_dim=64)
        
        # Per-Gaussian modulation bias (learnable per-Gaussian offset)
        # This allows each Gaussian to have its own baseline modulation
        self.modulation_bias = None  # Will be created lazily
    
    def set_bbox(self, bbox_min: torch.Tensor, bbox_max: torch.Tensor):
        """Set bounding box for position normalization."""
        if self.bbox_min is None:
            self.register_buffer('bbox_min', bbox_min.clone())
            self.register_buffer('bbox_max', bbox_max.clone())
        else:
            self.bbox_min.copy_(bbox_min)
            self.bbox_max.copy_(bbox_max)
    
    def normalize_positions(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Normalize world coordinates to [-1, 1] using bounding box.
        
        Args:
            xyz: (N, 3) world coordinates
            
        Returns:
            normalized: (N, 3) in [-1, 1]
        """
        if self.bbox_min is None or self.bbox_max is None:
            # Fallback: use min/max of positions
            bbox_min = xyz.min(dim=0)[0]
            bbox_max = xyz.max(dim=0)[0]
            center = (bbox_min + bbox_max) / 2
            scale = (bbox_max - bbox_min).max() / 2 + 1e-8
            return (xyz - center) / scale
        
        center = (self.bbox_min + self.bbox_max) / 2
        scale = (self.bbox_max - self.bbox_min).max() / 2 + 1e-8
        return (xyz - center) / scale
    
    def get_modulation(
        self, xyz: torch.Tensor, schedule_s: float = 1.0,
        view_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Compute density modulation factors for given positions.
        
        rho_final = rho_base * (1 + o_offset * o_conf * r_max * s * view_scale)
        
        Args:
            xyz: (N, 3) world coordinates of Gaussians
            schedule_s: Three-stage schedule strength [0, 1]
            view_scale: View-adaptive scaling factor
            
        Returns:
            modulation_factor: (N, 1) multiply with base density
        """
        # Normalize positions
        xyz_norm = self.normalize_positions(xyz)
        
        # Extract tri-plane features
        features = self.tri_plane(xyz_norm)  # (N, 3*C)
        
        # Decode to modulation signals
        offset, confidence = self.decoder(features)  # (N, 1), (N, 1)
        
        # Compute modulation factor
        modulation = 1.0 + offset * confidence * self.r_max * schedule_s * view_scale
        
        return modulation, offset, confidence
    
    def compute_tv_loss(self) -> torch.Tensor:
        """
        Compute TV regularization on all three feature planes.
        
        L_ADM-tv = TV(P_xy) + TV(P_xz) + TV(P_yz)
        
        Returns:
            tv_loss: scalar
        """
        def _tv_loss(grid: torch.Tensor) -> torch.Tensor:
            # grid: (1, C, H, W)
            diff_h = torch.abs(grid[:, :, 1:, :] - grid[:, :, :-1, :]).mean()
            diff_w = torch.abs(grid[:, :, :, 1:] - grid[:, :, :, :-1]).mean()
            return diff_h + diff_w
        
        loss_xy = _tv_loss(self.tri_plane.grid_xy)
        loss_xz = _tv_loss(self.tri_plane.grid_xz)
        loss_yz = _tv_loss(self.tri_plane.grid_yz)
        
        return loss_xy + loss_xz + loss_yz


def compute_adm_schedule(iteration: int, total_iters: int = 30000) -> float:
    """
    Three-stage modulation strength schedule.
    
    Stage 1 (Warmup, 0-20%): s linear 0 → 1.0
    Stage 2 (Normal, 20-70%): s = 1.0
    Stage 3 (Decay, 70-100%): s linear 1.0 → 0.5
    
    Args:
        iteration: Current training iteration
        total_iters: Total training iterations
        
    Returns:
        s: modulation strength in [0, 1]
    """
    warmup_end = int(total_iters * 0.20)   # 0-6000
    decay_start = int(total_iters * 0.70)  # 6000-21000
    
    if iteration < warmup_end:
        # Stage 1: Linear warmup
        s = iteration / warmup_end
    elif iteration < decay_start:
        # Stage 2: Full strength
        s = 1.0
    else:
        # Stage 3: Gradual decay
        decay_progress = (iteration - decay_start) / (total_iters - decay_start)
        s = 1.0 - 0.5 * decay_progress  # Decay to 0.5
    
    return max(0.0, min(1.0, s))


def compute_view_scale(n_views: int) -> float:
    """
    View-adaptive scaling factor.
    
    lambda_view = 1 / sqrt(n_views / 3)
    
    Fewer views → stronger modulation (more compensation needed)
    More views → weaker modulation (supervision is sufficient)
    """
    return 1.0 / np.sqrt(max(n_views, 1) / 3.0)


def create_adm_module(
    grid_size: int = 256,
    feat_dim: int = 32,
    r_max: float = 0.5,
    bbox: torch.Tensor = None,
    device: str = 'cuda'
) -> ADMModule:
    """
    Factory function to create an ADM module.
    """
    module = ADMModule(grid_size=grid_size, feat_dim=feat_dim, r_max=r_max, bbox=bbox)
    module = module.to(device)
    return module
