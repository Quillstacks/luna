"""Utilities for extracting and visualizing attention maps from DINOv3."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


class AttentionExtractor:
    """Hook to extract attention maps from DINOv3 transformer blocks."""
    
    def __init__(self, model: torch.nn.Module) -> None:
        self.attention_maps: list[torch.Tensor] = []
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks(model)
    
    def _register_hooks(self, model: torch.nn.Module) -> None:
        """Register forward hooks on all attention layers."""
        def hook_fn(module, input, output):
            # For DINOv3, attention output is typically (B, num_heads, N, N)
            # We average over heads: (B, N, N)
            if isinstance(output, torch.Tensor) and output.dim() == 4:
                attn = output.mean(dim=1)  # (B, N, N)
                self.attention_maps.append(attn.detach())
        
        for name, module in model.named_modules():
            if "attn" in name.lower() or "self_attention" in name.lower():
                try:
                    handle = module.register_forward_hook(hook_fn)
                    self.hooks.append(handle)
                    log.debug(f"Registered attention hook on {name}")
                except Exception as e:
                    log.debug(f"Could not register hook on {name}: {e}")
    
    def clear(self) -> None:
        """Clear stored attention maps."""
        self.attention_maps = []
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_attention_rollout(self) -> Optional[torch.Tensor]:
        """
        Compute attention rollout from all stored attention maps.
        
        Returns:
            Tensor of shape (B, N, N) where N is the number of patches + 1 (CLS token)
            or None if no attention maps were captured.
        """
        if not self.attention_maps:
            return None
        
        # Start with identity matrix
        batch_size = self.attention_maps[0].shape[0]
        num_tokens = self.attention_maps[0].shape[-1]
        rollout = torch.eye(num_tokens, device=self.attention_maps[0].device).unsqueeze(0).expand(batch_size, -1, -1)
        
        for attn in self.attention_maps:
            # Reshape if needed and ensure same device
            if attn.dim() == 3:
                attn = attn.unsqueeze(1)
            # Average over heads if needed
            if attn.dim() == 4:
                attn = attn.mean(dim=1)
            
            # Normalize rows to sum to 1 (stochastic matrix)
            attn = attn + torch.eye(attn.shape[-1], device=attn.device) * 1e-6
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            rollout = torch.bmm(attn, rollout)
        
        return rollout
    
    def __del__(self) -> None:
        self.remove_hooks()


def create_attention_overlay(
    image: np.ndarray,
    attention: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create an attention overlay on the input image.
    
    Args:
        image: Input image as numpy array (H, W) or (H, W, C), dtype uint8
        attention: Attention map as numpy array (H, W) or resized to match image
        alpha: Blending factor for overlay (0 = only image, 1 = only attention)
    
    Returns:
        RGB image with attention overlay, dtype uint8
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for attention visualization")
    
    # Ensure image is 3-channel
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Normalize attention to [0, 255]
    attention = attention.astype(np.float32)
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8) * 255
    attention = attention.astype(np.uint8)
    
    # Apply colormap (red/yellow heatmap)
    heatmap = np.zeros((*attention.shape, 3), dtype=np.uint8)
    heatmap[..., 0] = np.clip(attention * 2, 0, 255)  # Red channel
    heatmap[..., 1] = np.clip(attention, 0, 255)      # Green channel
    heatmap[..., 2] = 0                             # Blue channel
    
    # Blend with original image
    image_f = image.astype(np.float32)
    heatmap_f = heatmap.astype(np.float32)
    overlay = (image_f * (1 - alpha) + heatmap_f * alpha)
    
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_attention_overlay(
    image: np.ndarray,
    attention: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.5,
) -> None:
    """Save image with attention overlay to file."""
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for attention visualization")
    
    overlay = create_attention_overlay(image, attention, alpha)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    log.info(f"Saved attention overlay to {output_path}")
