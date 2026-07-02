"""Generate the GridForge methodology-pipeline diagram (Figure, Section 3).

Produces scripts/output/00_methodology_pipeline.png. Registered in
paper/build_figures.py so it stages into paper/figures/ like the others.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "scripts" / "output" / "00_methodology_pipeline.png"

BOX_FC = "#eaf0f7"; BOX_EC = "#33608f"; TXT = "#1b2a3a"; ACCENT = "#33608f"; SUP = "#8a5a2b"
stages = [
    ("1  Physics oracle",    "IEC 60287 / 60853\ncoaxial E-field\nCrine ageing",               "§3"),
    ("2  Synthetic dataset", "64 cables (320 cable-years)\n4 archetypes,\n4 condition modes\nsealed labels", "§4"),
    ("3  PINN surrogate",    "4x64 MLP + Fourier\ntrained vs. oracle\n0.064 °C RMSE",           "§5"),
    ("4  Benchmark",         "5 sealed tasks\n3 baselines\nfixed protocol",                     "§6"),
]

fig, ax = plt.subplots(figsize=(11.8, 3.7))
ax.set_xlim(0, 12.0); ax.set_ylim(-0.55, 3.4); ax.axis("off")
bw, bh, y0 = 2.6, 1.95, 0.95
xs = [0.20 + i * 3.0 for i in range(4)]

for x, (title, body, sec) in zip(xs, stages):
    ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0.03,rounding_size=0.10",
                                fc=BOX_FC, ec=BOX_EC, lw=1.6))
    ax.text(x + bw/2, y0 + bh - 0.28, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=ACCENT)
    ax.text(x + bw/2, y0 + bh/2 - 0.20, body, ha="center", va="center",
            fontsize=8.4, color=TXT, linespacing=1.4)
    ax.text(x + bw/2, y0 - 0.26, sec, ha="center", va="center",
            fontsize=9, color="#6b7785", style="italic")

for i in range(3):
    ax.add_patch(FancyArrowPatch((xs[i]+bw, y0+bh/2), (xs[i+1], y0+bh/2),
                                 arrowstyle="-|>", mutation_scale=18, lw=1.8, color=ACCENT))

# curved arrow BELOW the boxes: oracle supervises surrogate training (1 -> 3)
ax.add_patch(FancyArrowPatch((xs[0]+bw/2, y0), (xs[2]+bw/2, y0),
             connectionstyle="arc3,rad=0.28", arrowstyle="-|>", mutation_scale=16,
             lw=1.5, ls=(0,(5,3)), color=SUP))
ax.text((xs[0]+xs[2])/2 + bw/2, -0.34, "oracle supervises PINN training",
        ha="center", va="center", fontsize=8.4, color=SUP, style="italic")

ax.text(6.0, 3.25, "GridForge methodology pipeline", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color=TXT)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
