# GridForge

[![CI](https://github.com/mlily2024/Gridforge/actions/workflows/ci.yml/badge.svg)](https://github.com/mlily2024/Gridforge/actions/workflows/ci.yml)

**Open-source physics-informed digital twin and benchmark suite for UK 11kV underground distribution cables.**

GridForge is a research-grade reference implementation that bundles:

1. A **physics-informed neural network (PINN)** trained against the coupled heat, electric-field, and dielectric-ageing equations governing distribution cables — a model that satisfies first principles by construction.
2. A **calibrated synthetic UK 11kV cable dataset** large enough to train and benchmark on, with every parameter cited to a public source (IEC, CIGRE, BS EN, peer-reviewed literature).
3. A **benchmark suite** with five sealed-test tasks: 60-day failure prediction, remaining-useful-life regression, anomaly detection, virtual-sensor temperature reconstruction, and counterfactual fault attribution.

The goal is to give the UK distribution-grid community a common reference model and dataset against which any tool — vendor-built, academic, or in-house — can be measured.

## Project status

**v0.0.1 — alpha.** Day-1 foundation in place: IEC 60287 steady-state thermal solver and a canonical UK 11kV 240 mm² XLPE 3-core cable archetype, with a verification suite that tests internal physical consistency.

PINN, dataset, benchmark, and paper are on the 12-week roadmap (see `docs/`).

## Quickstart

```bash
git clone https://github.com/<user>/gridforge.git
cd gridforge
pip install -e ".[dev]"
pytest tests/
python scripts/01_validate_iec60287.py
```

The validation script prints a load-vs-conductor-temperature table for the canonical 11kV 240 mm² XLPE cable buried at 0.8 m in soil with thermal resistivity 1.0 K·m/W.

## Why physics-informed

Pure-ML cable models overfit to one operator's load patterns and can violate conservation laws (e.g. predicting a conductor cooler than its surroundings). Pure-physics models need parameters that are never measured in the field. A PINN combines both: data fits the network where sensors exist, while the physics residual constrains its behaviour everywhere else. The result is a model engineers can trust because it satisfies Maxwell's equations and energy conservation by construction.

References:

- Raissi, Perdikaris, Karniadakis (2019), *Physics-informed neural networks*, J. Computational Physics
- Cuomo et al. (2022), *Scientific machine learning through physics-informed neural networks: where we are and what's next*, J. Scientific Computing
- IEC 60287 — Calculation of the current rating, 2006/2015
- IEC 60228 — Conductors of insulated cables, 2004
- Crine (2005), *On the interpretation of some electrical-ageing and life-test results*, IEEE Trans. Dielectr. Electr. Insul.

## Layout

```
gridforge/
├── gridforge/
│   ├── physics/        steady-state thermal model, electric field, ageing kinetics
│   ├── models/         PINN architectures and classical baselines
│   ├── training/       loss functions, optimisation, scheduling
│   ├── inference/      virtual sensors, RUL estimation
│   ├── bench/          benchmark suite + leaderboard harness
│   └── data/           synthetic dataset generator
├── tests/              unit + verification tests
├── scripts/            CLI entry points
├── docs/adr/           architecture decision records
└── notebooks/          worked examples
```

## License

MIT — see `LICENSE`.

## Citation

If you use GridForge in your work, please cite it via the **"Cite this repository"** button (backed by [`CITATION.cff`](CITATION.cff)).
