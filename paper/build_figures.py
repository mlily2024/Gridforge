"""
Stage paper figures from `scripts/output/` into `paper/figures/`.

The Day 1–6 demo scripts generate PNGs under `scripts/output/`. This
helper copies the ones the paper references into a stable path and
rejects the build if any are missing — surfacing a regenerate hint
to the author rather than producing a paper with broken figure links.

Run:
    python paper/build_figures.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_OUT = REPO_ROOT / "scripts" / "output"
FIG_DIR = REPO_ROOT / "paper" / "figures"


# Map paper-figure name -> source PNG produced by a demo script. Any
# new figure the paper references is added here.
FIGURE_MAP = {
    "fig01_iec_validation.png": SCRIPTS_OUT / "01_load_vs_temperature.png",
    "fig02_diurnal_response.png": SCRIPTS_OUT / "02_diurnal_load.png",
    "fig03_lifetime_curves.png": SCRIPTS_OUT / "03_lifetime_curves.png",
    "fig04_pinn_training_curves.png": SCRIPTS_OUT / "05_pinn_training" / "training_curves.png",
    "fig05_pinn_validation_scatter.png": SCRIPTS_OUT
    / "05_pinn_training"
    / "validation_scatter.png",
}


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    missing = []
    copied = []
    for dst_name, src in FIGURE_MAP.items():
        if not src.exists():
            missing.append((dst_name, src))
            continue
        dst = FIG_DIR / dst_name
        shutil.copyfile(src, dst)
        copied.append((dst_name, src))

    print()
    print(f"Staged {len(copied)} figures into {FIG_DIR}")
    for name, src in copied:
        size_kb = src.stat().st_size / 1024
        print(f"  {name:<40} ({size_kb:.1f} KB)  <- {src.relative_to(REPO_ROOT)}")

    if missing:
        print()
        print("Missing source figures (regenerate via the demo scripts):")
        for name, src in missing:
            print(f"  {name:<40} expected at {src.relative_to(REPO_ROOT)}")
        print()
        print("Suggested regenerate sequence:")
        print("  python scripts/01_validate_iec60287.py")
        print("  python scripts/02_transient_diurnal_load.py")
        print("  python scripts/03_lifetime_estimate.py")
        print("  python scripts/05_train_pinn_iec_oracle.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
