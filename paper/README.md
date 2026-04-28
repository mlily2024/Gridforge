# GridForge preprint

This folder holds the source for the GridForge preprint:

```
paper/
├── paper.md              Markdown source (Pandoc-compilable)
├── references.bib        Bibliography
├── build_figures.py      Stages figures from scripts/output/
├── build.sh              Pandoc build helper (Unix / Git Bash)
├── build.bat             Pandoc build helper (Windows)
└── figures/              Staged copies of demo-script PNGs (gitignored)
```

## Build

The build is two steps. The first regenerates demo-script figures
(only required if `scripts/output/` is empty or stale):

```bash
python scripts/01_validate_iec60287.py
python scripts/02_transient_diurnal_load.py
python scripts/03_lifetime_estimate.py
python scripts/05_train_pinn_iec_oracle.py    # ~2 min on CPU
```

The second compiles the paper. From the `gridforge` subpackage root:

```bash
./paper/build.sh              # both PDF and HTML
./paper/build.sh --html       # HTML only
./paper/build.sh --pdf        # PDF only
```

On Windows:

```cmd
paper\build.bat
```

Outputs land at `scripts/output/07_paper/paper.pdf` and
`scripts/output/07_paper/paper.html`. Both are gitignored and
regenerable.

## Dependencies

- **Pandoc** ≥ 2.19 with the `--citeproc` filter
- **XeLaTeX** (only for PDF output; install via TeX Live or MiKTeX)
- Python ≥ 3.10 with the GridForge package on the path (already
  satisfied if you can run the demo scripts)

## Author

Single-author preprint. No AI / Anthropic / Claude attribution
appears in the source, the bibliography, or the build artefacts.
