"""List newly-released LROC CDR volumes since a prior sweep.

Step 4 of the roadmap: when the LROC archive publishes a new monthly release
(volumes ``LROLRC_1045``, ``_1046``, ...), re-run inference on the new frames.
This script characterizes the current archive via :class:`PDSIndex`, compares
to a stored snapshot, and reports the delta. Intended as a monthly cron:

    python scripts/monthly_pds_sweep.py --state catalogs/pds_state.json

On first run the state file is created and no volumes are "new". On each
subsequent run, newly-listed volumes are printed (and returned via exit code
0 / 1 based on ``--fail-on-new`` for cron alerting).

Once a new volume is detected, the downstream step is to run
``scripts/predict_nac.py`` across its frames — not done here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from luna.io import PDSIndex

log = logging.getLogger("monthly_pds_sweep")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=Path,
                   default=Path(__file__).resolve().parent.parent / "catalogs" / "pds_state.json",
                   help="snapshot of volumes seen on previous runs")
    p.add_argument("--archive", choices=["CDR", "EDR"], default="CDR")
    p.add_argument("--fail-on-new", action="store_true",
                   help="exit 1 if any new volumes are found (for cron alerts)")
    p.add_argument("--dry-run", action="store_true",
                   help="list current archive but do not persist state")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    idx = PDSIndex(archive=args.archive)
    current = [v.volume_id for v in idx.volumes()]
    log.info("%s archive: %d volumes (latest=%s)",
             args.archive, len(current), current[-1] if current else "?")

    prior: list[str] = []
    if args.state.exists():
        prior = json.loads(args.state.read_text()).get("volumes", [])

    new = sorted(set(current) - set(prior))
    if new:
        log.info("NEW volumes since last sweep: %s", ", ".join(new))
    else:
        log.info("no new volumes")

    if not args.dry_run:
        args.state.parent.mkdir(parents=True, exist_ok=True)
        args.state.write_text(json.dumps({"volumes": current}, indent=2))

    return 1 if (new and args.fail_on_new) else 0


if __name__ == "__main__":
    sys.exit(main())
