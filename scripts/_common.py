"""Utilitaires partagés par les scripts d'expérimentation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from app.preprocessing import load_data, preprocess_data, split_data  # noqa: E402

# Palette de référence (validée daltonisme) : slots catégoriels 1 et 2.
SERIES = {"logreg": "#2a78d6", "tree": "#eb6834"}
LABELS = {"logreg": "Régression logistique", "tree": "Arbre de décision"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)


def load_splits(data_path: str | Path):
    df = load_data(data_path)
    train, val, test = split_data(df)
    parts = {"train": train, "val": val, "test": test}
    return df, {k: preprocess_data(v) for k, v in parts.items()}


def dump_json(obj, path: Path) -> None:
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=default) + "\n")
