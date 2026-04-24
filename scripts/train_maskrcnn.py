"""Train Mask R-CNN on ellipse-approximated pit masks from LPA/Wagner.

Single-GPU training loop. Minimal by design — one COCO JSON in, one
checkpoint out. No Lightning, no distributed. If you need those, wrap
this file rather than growing it.

Usage:
    python scripts/train_maskrcnn.py \\
        --coco data/ellipse_ds/pits.json \\
        --image-dir data/ellipse_ds/images \\
        --out checkpoints/maskrcnn_pit.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from luna.models import CocoMaskDataset, build_maskrcnn
from luna.models.maskrcnn import collate_fn

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--coco", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "maskrcnn_pit.pt")
    p.add_argument("--num-classes", type=int, default=2, help="background + pit = 2")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("train")

    dataset = CocoMaskDataset(args.coco, args.image_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    model = build_maskrcnn(num_classes=args.num_classes)
    model.to(args.device)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=max(1, args.epochs // 3), gamma=0.1)

    start_epoch = 0
    if args.resume and args.resume.exists():
        state = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        sched.load_state_dict(state["sched"])
        start_epoch = int(state.get("epoch", 0))
        log.info("resumed from %s (epoch=%d)", args.resume, start_epoch)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = {}
        for i, (images, targets) in enumerate(loader):
            images = [img.to(args.device) for img in images]
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optim.zero_grad()
            loss.backward()
            optim.step()
            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            if i % 10 == 0:
                log.info("epoch %d step %d loss=%.3f", epoch, i, float(loss.detach()))
        sched.step()
        avg = {k: v / max(1, len(loader)) for k, v in running.items()}
        log.info("epoch %d done in %.1fs — %s", epoch, time.time() - t0, avg)

        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "epoch": epoch + 1,
            "num_classes": args.num_classes,
        }, args.out)
        log.info("saved %s", args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
