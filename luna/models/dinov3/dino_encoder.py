import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from huggingface_hub import hf_hub_download

log = logging.getLogger(__name__)


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
                snapshots_dir = _hf_cache / f"models--{_repo_slug}" / "snapshots"
                if snapshots_dir.exists():
                    for snap in snapshots_dir.iterdir():
                        if snap.is_dir():
                            for fname in filenames_to_try:
                                candidate = snap / fname
                                if candidate.exists():
                                    _candidates.append(candidate)
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
            # Convert FP16 weights to FP32 to avoid dtype mismatch on MPS
            state_dict = {k: v.float() if v.dtype == torch.float16 else v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict, strict=True)
        else:
            log.warning(
                "Base weights not found. Model will use random weights."
            )

        self.model = self._attach_lora(lora_dir)
        self.model.to(self.device)
        self.model.eval()
        if self.device.type == "cuda":
            log.info(f"Switching DINOv3 to FP16 precision on {self.device.type}...")
            self.model = self.model.half()
        elif self.device.type == "mps":
            log.info(f"Keeping DINOv3 in FP32 precision on MPS (FP16 causes dtype mismatch with Stage-2 decoder)")

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
        
        # Dynamic check in standard user Hugging Face cache if lora_path_str is a HF repo slug
        if not os.path.isdir(lora_path_str) and "/" in lora_path_str:
            _repo_slug = lora_path_str.replace("/", "--")
            _hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            _snapshots = _hf_cache / f"models--{_repo_slug}" / "snapshots"
            if _snapshots.exists():
                for snap in _snapshots.iterdir():
                    if snap.is_dir():
                        lora_path_str = str(snap)
                        break
        
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
        if self.device.type == "cuda":
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
        """Encode with actual attention map extraction from the last self-attention layer."""
        import torch.nn.functional as F
        import math
        
        # Capture self-attention from the last transformer block
        captured_attn = []
        last_block = None
        orig_compute_attention = None
        
        try:
            last_block = self._backbone_module.blocks[-1]
            orig_compute_attention = last_block.attn.compute_attention
            
            def hook_compute_attention(qkv, attn_bias=None, rope=None):
                B, N, _ = qkv.shape
                C = last_block.attn.qkv.in_features
                qkv_reshaped = qkv.reshape(B, N, 3, last_block.attn.num_heads, C // last_block.attn.num_heads)
                q, k, v = torch.unbind(qkv_reshaped, 2)
                q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
                if rope is not None:
                    q, k = last_block.attn.apply_rope(q, k, rope)
                attn = (q @ k.transpose(-2, -1)) * last_block.attn.scale
                attn = attn.softmax(dim=-1)
                captured_attn.append(attn.mean(dim=1).detach())
                x = attn @ v
                x = x.transpose(1, 2)
                return x.reshape([B, N, C])
                
            last_block.attn.compute_attention = hook_compute_attention
        except Exception as e:
            log.warning(f"Could not setup attention hook: {e}")
            
        # Forward pass
        features_dict = self._backbone_module.forward_features(tensor)
        features = features_dict["x_norm_clstoken"]
        
        if self.matryoshka_dim < features.shape[1]:
            features = features[:, : self.matryoshka_dim]
            
        embeddings = F.normalize(features, p=2, dim=-1).cpu().float().numpy()
        
        # Restore original compute_attention method
        if last_block is not None and orig_compute_attention is not None:
            last_block.attn.compute_attention = orig_compute_attention
            
        # Process captured attention map
        if captured_attn:
            attn_matrix = captured_attn[0] # Shape: (B, N_tokens, N_tokens)
            
            # The CLS token attention to other tokens is at index 0
            cls_attn = attn_matrix[:, 0, :] # Shape: (B, N_tokens)
            
            # Get image spatial dimensions
            img_h, img_w = tensor.shape[-2], tensor.shape[-1]
            patch_size = 16 # DINOv3 VitS16 patch size
            n_spatial_h = img_h // patch_size
            n_spatial_w = img_w // patch_size
            n_spatial_total = n_spatial_h * n_spatial_w
            
            # The spatial patch tokens are the last n_spatial_total tokens in the sequence
            cls_attn_spatial = cls_attn[:, -n_spatial_total:] # Shape: (B, n_spatial_total)
            
            # Reshape to 2D spatial grid
            attention_spatial = cls_attn_spatial.reshape(tensor.shape[0], n_spatial_h, n_spatial_w)
            
            # Normalize attention map to [0, 1] for visualization
            attn_min = attention_spatial.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
            attn_max = attention_spatial.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            attention_spatial = (attention_spatial - attn_min) / (attn_max - attn_min + 1e-8)
            
            # Resize/interpolate to match input image dimensions
            attention_resized = F.interpolate(
                attention_spatial.unsqueeze(1), # (B, 1, n_spatial_h, n_spatial_w)
                size=(img_h, img_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1) # (B, H, W)
            
            attention_map = attention_resized.cpu().float().numpy()
        else:
            # Fallback: create a center-focused attention map
            h, w = tensor.shape[-2], tensor.shape[-1]
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            r2 = (x - center_x)**2 + (y - center_y)**2
            sigma = min(h, w) / 4
            attention_map = np.exp(-r2 / (2 * sigma**2)).astype(np.float32)
            attention_map = attention_map[np.newaxis, ...]
            attention_map = np.tile(attention_map, (tensor.shape[0], 1, 1))
            
        return embeddings, attention_map