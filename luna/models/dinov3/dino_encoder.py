import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from huggingface_hub import hf_hub_download

log = logging.getLogger("luna.models.dinov3")


class DINOEncoder:
    """
    DINOv3 (vits16) wrapper with LVD pretrain weights and optional LoRA adapters.
    Compatible with the EmbeddingModel protocol for CandidateScreener.
    """

    def __init__(
        self,
        lora_dir: str | Path | None = None,
        base_weights_path: str = "dinov3_vits16_pretrain_lvd.pth",
        model_size: str = "vits16",
        matryoshka_dim: int = 384,
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        base_weights_filename: str | None = None,
    ):
        self.device = torch.device(device)
        self.matryoshka_dim = matryoshka_dim

        log.info(f"Loading DINO backbone: dinov3_{model_size}...")

        try:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov3",
                f"dinov3_{model_size}",
                pretrained=False,
            )
        except Exception as e:
            log.error(f"Failed to load DINOv3 from hub: {e}")
            raise

        resolved_base_path = str(base_weights_path)
        filename = base_weights_filename

        if not filename:
            path_obj = Path(resolved_base_path)
            if path_obj.suffix in [".pth", ".pt", ".safetensors", ".bin"]:
                filename = path_obj.name
                if len(path_obj.parts) > 1 and not path_obj.exists():
                    repo_id = str(Path(*path_obj.parts[:-1]))
                    resolved_base_path = repo_id
            if not filename:
                filename = "dinov3_vits16_pretrain_lvd.pth"

        if not os.path.exists(resolved_base_path):
            log.info(f"Trying to fetch base weights from Hugging Face: {resolved_base_path}...")
            filenames_to_try = [filename]
            if filename == "dinov3_vits16_pretrain_lvd.pth":
                filenames_to_try.append("dinov3_vits16_pretrain_lvd.safetensors")

            success = False
            for fname in filenames_to_try:
                try:
                    resolved_base_path = hf_hub_download(
                        repo_id=resolved_base_path, 
                        filename=fname
                    )
                    success = True
                    break
                except Exception as e:
                    log.warning(f"Could not download {fname} from HF: {e}")

            if not success:
                # Fall back to local HF snapshot cache
                from pathlib import Path as _Path
                _repo_slug = base_weights_path.replace("/", "--")
                _hf_cache = _Path.home() / ".cache" / "huggingface" / "hub"
                _candidates = []
                for fname in filenames_to_try:
                    _candidates.extend(
                        list((_hf_cache / f"models--{_repo_slug}").rglob(fname))
                    )
                if _candidates:
                    _candidates = [
                        p for p in _candidates
                        if ".no_exist" not in p.parts and p.stat().st_size > 1024
                    ]
                if _candidates:
                    resolved_base_path = str(_candidates[0])
                    log.info(f"Found base weights in HF snapshot cache: {resolved_base_path}")

        if os.path.exists(resolved_base_path):
            log.info(f"Loading base weights from {resolved_base_path}...")
            if resolved_base_path.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(resolved_base_path, device="cpu")
                except ImportError:
                    log.error("Please install 'safetensors' to load .safetensors files.")
                    raise
            else:
                state_dict = torch.load(resolved_base_path, map_location="cpu", weights_only=True)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.backbone.load_state_dict(state_dict, strict=True)
        else:
            log.warning(
                "Base weights not found. Model will use random weights."
            )

        self.model = self._attach_lora(lora_dir)
        self.model.to(self.device)
        self.model.eval()
        if self.device.type in ["mps", "cuda"]:
            log.info(f"Switching DINOv3 to FP16 precision on {self.device.type}...")
            self.model = self.model.half()

        # Keep a direct reference to the raw backbone for inference.
        # Newer peft versions route model(x) through a text-model forward that
        # injects `input_ids`, which DINOv3's forward_features() doesn't accept.
        # Calling backbone.forward_features() directly avoids this entirely while
        # still benefiting from LoRA adapter weights applied in-place.
        try:
            self._backbone_module = self.model.base_model.model
        except AttributeError:
            self._backbone_module = self.backbone

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _attach_lora(self, lora_dir: str | Path | None) -> torch.nn.Module:
        if not lora_dir:
            return self.backbone

        try:
            from peft import PeftModel
        except ImportError:
            log.error("Run 'pip install peft' to use LoRA adapters.")
            raise

        lora_path_str = str(lora_dir)
        
        if os.path.isdir(lora_path_str) and os.path.exists(os.path.join(lora_path_str, "weights")):
            lora_path_str = os.path.join(lora_path_str, "weights")

        log.info(f"Loading LoRA adapters from {lora_path_str}...")
        
        try:
            return PeftModel.from_pretrained(self.backbone, lora_path_str)
        except Exception as e:
            log.error(f"Failed to load LoRA from {lora_path_str}: {e}. Falling back to base model.")
            return self.backbone

    @torch.inference_mode()
    def encode(self, batch: np.ndarray, return_attention: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Encode a batch of grayscale or RGB images into L2-normalized embeddings.
        
        Args:
            batch: Input images as numpy array, shape (B, H, W) or (B, C, H, W), dtype uint8
            return_attention: If True, also return attention rollout map
            
        Returns:
            If return_attention is False: embeddings array of shape (B, matryoshka_dim)
            If return_attention is True: tuple of (embeddings, attention_map) where attention_map
                is shape (B, num_patches+1, num_patches+1) resized to (B, H, W)
        """
        if self.device.type in ["mps", "cuda"]:
            tensor_dtype = torch.float16
        else:
            tensor_dtype = torch.float32

        if batch.ndim == 3:
            tensor = torch.from_numpy(batch).unsqueeze(1).to(device=self.device, dtype=tensor_dtype)
        elif batch.ndim == 4:
            tensor = torch.from_numpy(batch).to(device=self.device, dtype=tensor_dtype)
        else:
            raise ValueError(f"Expected batch shape (B, H, W) or (B, C, H, W), got {batch.shape}")
        
        if batch.dtype == np.uint8:
            tensor = tensor / 255.0

        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)

        tensor = self.normalize(tensor)
        
        if return_attention:
            return self._encode_with_attention(tensor)
        
        features = self._backbone_module.forward_features(tensor)["x_norm_clstoken"]

        if self.matryoshka_dim < features.shape[1]:
            features = features[:, : self.matryoshka_dim]

        return F.normalize(features, p=2, dim=-1).cpu().float().numpy()
    
    @torch.inference_mode()
    def _encode_with_attention(self, tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Encode with attention map extraction.
        
        For DINOv3, instead of trying to hook into attention layers (which is
        complex and unreliable), we create an attention map based on the L2 norm
        of patch tokens. This gives a reasonable approximation of where the
        model is "paying attention".
        """
        import torch.nn.functional as F
        
        # Forward pass
        features_dict = self._backbone_module.forward_features(tensor)
        features = features_dict["x_norm_clstoken"]
        
        if self.matryoshka_dim < features.shape[1]:
            features = features[:, : self.matryoshka_dim]
        
        embeddings = F.normalize(features, p=2, dim=-1).cpu().float().numpy()
        
        # Create attention map from patch token norms
        # x_norm_patchtokens has shape (B, N_patches, D) where N_patches = (H/patch_size) * (W/patch_size)
        if "x_norm_patchtokens" in features_dict:
            patch_tokens = features_dict["x_norm_patchtokens"]  # (B, N_patches, D)
            
            # Compute L2 norm of each patch token: (B, N_patches)
            patch_norms = torch.norm(patch_tokens, p=2, dim=-1)
            
            # Get spatial dimensions
            img_h, img_w = tensor.shape[-2], tensor.shape[-1]
            n_patches = patch_norms.shape[-1]
            
            # Calculate patch grid size
            # DINOv3 vits16: patch_size = 16, so for 256x256 we have 16x16 = 256 patches
            # But the actual number depends on the model configuration
            n_spatial = int(n_patches ** 0.5)
            
            # Check if it's a perfect square
            if n_spatial * n_spatial == n_patches:
                # Reshape to 2D
                attention_spatial = patch_norms.reshape(tensor.shape[0], n_spatial, n_spatial)
            else:
                # Not a perfect square, create a 2D map by finding the closest square
                # Use the first n_spatial*n_spatial patches
                import math
                n_spatial = int(math.sqrt(n_patches))
                n_use = n_spatial * n_spatial
                
                if n_patches > n_use:
                    # Truncate to square number
                    patch_norms = patch_norms[:, :n_use]
                elif n_patches < n_use:
                    # Pad with minimum values
                    min_val = patch_norms.min()
                    padding = torch.full((tensor.shape[0], n_use - n_patches), 
                                       min_val, device=patch_norms.device)
                    patch_norms = torch.cat([patch_norms, padding], dim=-1)
                
                attention_spatial = patch_norms.reshape(tensor.shape[0], n_spatial, n_spatial)
            
            # Normalize attention map to [0, 1]
            attention_spatial = (attention_spatial - attention_spatial.min()) / \
                                (attention_spatial.max() - attention_spatial.min() + 1e-8)
            
            # Resize to match input image size
            attention_resized = F.interpolate(
                attention_spatial.unsqueeze(1),  # (B, 1, n_spatial, n_spatial)
                size=(img_h, img_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # (B, H, W)
            
            attention_map = attention_resized.cpu().float().numpy()
        else:
            # Fallback: create a center-focused attention map
            # If x_norm_patchtokens is not available
            h, w = tensor.shape[-2], tensor.shape[-1]
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            r2 = (x - center_x)**2 + (y - center_y)**2
            sigma = min(h, w) / 4
            attention_map = np.exp(-r2 / (2 * sigma**2)).astype(np.float32)
            # Add batch dimension
            attention_map = attention_map[np.newaxis, ...]
            # Expand to match batch size
            attention_map = np.tile(attention_map, (tensor.shape[0], 1, 1))
        
        return embeddings, attention_map