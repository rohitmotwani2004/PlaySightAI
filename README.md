# Badminton Analysis System

This repository provides a starter scaffold for a badminton video analysis system.

Structure:
- `data/` — input / processed videos and datasets
- `models/` — trained model weights
- `src/` — source modules (court detection, player detection, shuttle tracker, analysis)
- `utils/` — helper functions for video I/O and drawing
- `notebooks/` — quick testing notebooks

Quick start

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Place raw videos in `data/raw/` and update paths in `config.yaml`.
3. Run analysis:

```bash
python main.py --input data/raw/match_01.mp4
```
