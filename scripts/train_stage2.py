import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to python path to allow internal imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from luna.models.dinov3 import DINOEncoder
from luna.models.stage2_decoder import build_stage2_decoder, Stage2DenseDecoder


class PitDataset(Dataset):
    """Simple dataset wrapper for cached spatial tokens and targets."""
    def __init__(self, tokens: list[torch.Tensor], masks: list[torch.Tensor]):
        self.tokens = tokens
        self.masks = masks

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[idx], self.masks[idx]


class FocalLoss(nn.Module):
    """Multi-class Focal Loss implementation."""
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha  # class weights tensor, shape (C,)
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs shape: (B, C, H, W), targets shape: (B, H, W)
        log_p = torch.log_softmax(inputs, dim=1)
        ce_loss = F.nll_loss(log_p, targets, weight=self.alpha, reduction='none')
        
        # Get target probability p_t
        p = torch.exp(log_p)
        target_p = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal scaling term
        focal_term = (1.0 - target_p) ** self.gamma
        loss = ce_loss * focal_term
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def augment_batch(tokens: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies random horizontal and vertical flips aligned across tokens and pixel masks."""
    B, num_patches, C = tokens.shape
    H_p, W_p = 16, 16
    tokens_grid = tokens.view(B, H_p, W_p, C)
    
    # Random horizontal flip
    if np.random.rand() > 0.5:
        tokens_grid = torch.flip(tokens_grid, dims=[2])
        masks = torch.flip(masks, dims=[2])
        
    # Random vertical flip
    if np.random.rand() > 0.5:
        tokens_grid = torch.flip(tokens_grid, dims=[1])
        masks = torch.flip(masks, dims=[1])
        
    tokens = tokens_grid.view(B, num_patches, C)
    return tokens, masks


def apply_strict_threshold(logits: torch.Tensor, threshold: float = 0.85) -> np.ndarray:
    """Classifies pixels using a strict threshold for True Void (Class 1)."""
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    B, _, H, W = probs.shape
    preds = np.zeros((B, H, W), dtype=np.int64)
    
    # Class 1: True Void (Strict threshold)
    class_1_mask = probs[:, 1, :, :] > threshold
    preds[class_1_mask] = 1
    
    # Class 2: Obstacle (Standard threshold)
    class_2_mask = (probs[:, 2, :, :] > 0.5) & (~class_1_mask)
    preds[class_2_mask] = 2
    
    return preds


def compute_metrics(preds: np.ndarray, targets: np.ndarray, num_classes: int = 3) -> tuple[float, list[float], list[float], list[float]]:
    """Calculate mIoU, class-wise IoU, Precision, and Recall."""
    ious = []
    precisions = []
    recalls = []
    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)
        
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        
        iou = float(intersection) / float(union) if union > 0 else 1.0
        ious.append(iou)
        
        tp = float(intersection)
        fp = float((pred_c & ~target_c).sum())
        fn = float((~pred_c & target_c).sum())
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        precisions.append(prec)
        recalls.append(rec)
        
    mIoU = float(np.mean(ious))
    return mIoU, ious, precisions, recalls


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Stage2DenseDecoder on verified pit patches.")
    parser.add_argument("--pits-dir", type=Path, default=ROOT / "data" / "_scratch" / "pits", help="Directory containing npy pit patches")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "weights" / "stage2_decoder_finetuned.pt", help="Finetuned checkpoint save path")
    parser.add_argument("--epochs", type=int, default=35, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--threshold", type=float, default=0.75, help="Strict threshold for Class 1 (True Void) prediction")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    
    if not args.pits_dir.exists():
        print(f"Error: Pits directory {args.pits_dir} does not exist.")
        return 1
        
    pit_files = sorted(args.pits_dir.glob("*.npy"))
    if not pit_files:
        print(f"Error: No .npy files found in {args.pits_dir}")
        return 1
        
    print(f"Found {len(pit_files)} pit patches.")

    # 1. Initialize Stage2DenseDecoder from baseline checkpoint
    baseline_weights = ROOT / "data" / "weights" / "stage2_decoder_best.pt"
    print(f"Loading decoder baseline from {baseline_weights}...")
    decoder = build_stage2_decoder(checkpoint_path=baseline_weights, device=args.device)
    
    # 2. Extract DINOv3 backbone features & generate targets
    print("Initializing frozen DINOv3 encoder for feature extraction...")
    from luna.config import HF_REPO_ID
    encoder = DINOEncoder(
        lora_dir=HF_REPO_ID,
        base_weights_path=HF_REPO_ID,
        device=args.device
    )

    from torchvision import transforms
    from PIL import Image
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    pre_extracted_tokens = []
    generated_masks = []
    
    # Generate 256x256 spatial circular center-bias distance mask (R <= 40 px)
    Y, X = np.ogrid[:256, :256]
    dist_from_center = np.sqrt((X - 128)**2 + (Y - 128)**2)
    center_mask = dist_from_center <= 40.0
    
    print("Pre-extracting features and creating spatially-biased target masks...")
    for f in tqdm(pit_files, desc="Encoding progress"):
        img = np.load(f)
        if img.ndim != 2:
            continue
        if img.shape != (256, 256):
            img = np.array(Image.fromarray(img).resize((256, 256), Image.Resampling.BILINEAR))
            
        # LROC-specific valid range normalization
        valid_pixels = img[img > -32752]
        lo, hi = (valid_pixels.min(), valid_pixels.max()) if valid_pixels.size > 0 else (0.0, 1.0)
        norm = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
        
        # Spatially biased procedural masks: Pits/shadows and obstacles/boulders must be close to the center
        target_mask = np.zeros((256, 256), dtype=np.int64)
        target_mask[(norm < 0.20) & center_mask] = 1  # Class 1: True Void
        target_mask[(norm > 0.80) & center_mask] = 2  # Class 2: Obstacle
        
        generated_masks.append(torch.from_numpy(target_mask))
        
        # Prepare tensor for backbone feed
        img_t = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)
        img_t = img_t.repeat(1, 3, 1, 1)
        if device.type == "cuda":
            img_t = img_t.half()
            
        img_norm = normalize_transform(img_t)
        
        with torch.no_grad():
            features = encoder._backbone_module.forward_features(img_norm)
            spatial_tokens = decoder.token_extractor.extract_spatial_tokens(features).squeeze(0)  # (num_patches, hidden_dim)
            pre_extracted_tokens.append(spatial_tokens.cpu())
            
    # Clean up encoder to reclaim GPU memory
    print("Pre-extraction complete. Releasing DINOv3 backbone VRAM...")
    del encoder
    torch.cuda.empty_cache()

    # 3. Deterministic Splitting (70% Train, 15% Val, 15% Test)
    n_samples = len(pre_extracted_tokens)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    n_train = int(0.70 * n_samples)
    n_val = int(0.15 * n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    
    train_tokens = [pre_extracted_tokens[i] for i in train_idx]
    train_masks = [generated_masks[i] for i in train_idx]
    
    val_tokens = [pre_extracted_tokens[i] for i in val_idx]
    val_masks = [generated_masks[i] for i in val_idx]
    
    test_tokens = [pre_extracted_tokens[i] for i in test_idx]
    test_masks = [generated_masks[i] for i in test_idx]
    
    # 4. Data Loaders & Optimizer
    train_loader = DataLoader(PitDataset(train_tokens, train_masks), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(PitDataset(val_tokens, val_masks), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(PitDataset(test_tokens, test_masks), batch_size=args.batch_size, shuffle=False)
    
    # Use mild class weights in CrossEntropyLoss to balance classes without over-optimizing recall
    class_weights = torch.tensor([1.0, 1.5, 1.5], device=device)
    print(f"Using class weights: {class_weights.tolist()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Target learnable decoder layers and train in float32 for gradient stability
    decoder.float()
    for p in decoder.parameters():
        p.requires_grad = True
        
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 5. Training Loop
    print(f"\nStarting Stage-2 Decoder Few-Shot Retraining Loop (CrossEntropy + Precision threshold={args.threshold})...")
    best_val_f05 = -1.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        decoder.train()
        train_loss = 0.0
        for tokens, masks in train_loader:
            tokens = tokens.to(device).float()
            masks = masks.to(device).long()
            
            # Apply random flips to spatial tokens and targets
            tokens, masks = augment_batch(tokens, masks)
            
            optimizer.zero_grad()
            logits = decoder(tokens)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * tokens.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation evaluation
        decoder.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for tokens, masks in val_loader:
                tokens = tokens.to(device).float()
                masks = masks.to(device).long()
                logits = decoder(tokens)
                loss = criterion(logits, masks)
                val_loss += loss.item() * tokens.size(0)
                
                # Apply strict threshold to predict Class 1 (True Void)
                preds = apply_strict_threshold(logits, threshold=args.threshold)
                val_preds.append(preds)
                val_targets.append(masks.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        # Compute validation split metrics
        _, _, precisions, recalls = compute_metrics(val_preds, val_targets)
        val_prec = precisions[1]  # True Void precision
        val_rec = recalls[1]      # True Void recall
        
        # Calculate F0.5 score: (1 + 0.25) * P * R / (0.25 * P + R)
        val_f05 = (1.25 * val_prec * val_rec) / (0.25 * val_prec + val_rec + 1e-8) if (val_prec + val_rec) > 0 else 0.0
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Void Prec: {val_prec:.4f} | Val Void Rec: {val_rec:.4f} | Val F0.5: {val_f05:.4f}")
        
        # Save checkpoints based on maximum Validation F0.5 (favoring Precision) & Early Stopping
        if val_f05 > best_val_f05:
            best_val_f05 = val_f05
            patience_counter = 0
            decoder.save_checkpoint(args.out)
            print(f"  --> Saved new best checkpoint (F0.5={val_f05:.4f}) to {args.out}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered. Best Val F0.5: {best_val_f05:.4f}")
                break
                
    # 6. Evaluation on Test Split
    print("\nEvaluating best checkpoint on isolated Test Split...")
    best_decoder = Stage2DenseDecoder.from_checkpoint(args.out, device=args.device)
    best_decoder.float()
    best_decoder.eval()
    
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for tokens, masks in test_loader:
            tokens = tokens.to(device).float()
            masks = masks.to(device).long()
            logits = best_decoder(tokens)
            loss = criterion(logits, masks)
            test_loss += loss.item() * tokens.size(0)
            
            # Apply strict threshold to predict Class 1 (True Void)
            preds = apply_strict_threshold(logits, threshold=args.threshold)
            all_preds.append(preds)
            all_targets.append(masks.cpu().numpy())
            
    test_loss /= len(test_loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    mIoU, ious, precisions, recalls = compute_metrics(all_preds, all_targets)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test mIoU: {mIoU:.4f}")
    class_names = ["Regolith", "True Void", "Obstacle"]
    for c in range(3):
        print(f"Class '{class_names[c]}':")
        print(f"  IoU:       {ious[c]:.4f}")
        print(f"  Precision: {precisions[c]:.4f}")
        print(f"  Recall:    {recalls[c]:.4f}")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
