"""Analyse Sentinel-2 et cartographie la severite des feux.

Dependances : pip install sentinelhub numpy matplotlib pandas jenkspy
Configurer SH_CLIENT_ID et SH_CLIENT_SECRET dans l'environnement.
"""

from __future__ import annotations

import argparse
import os

import jenkspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SHConfig,
    SentinelHubRequest,
    bbox_to_dimensions,
)


DEFAULT_BBOX = (-90.692139, 50.291944, -88.62, 51.170176)
DEFAULT_BEFORE_DATE = "2026-06-01"
DEFAULT_AFTER_DATE = "2026-09-22"
SEVERITY_LABELS = ["Non brule", "Faible", "Modere", "Severe", "Tres fort"]
SEVERITY_COLORS = ["#2864dc", "#43a047", "#f5d547", "#ef8c32", "#d9443f"]

EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{bands: ["B02", "B03", "B04", "B08", "B12"]}],
        output: {bands: 4, sampleType: "FLOAT32"}
    };
}
function evaluatePixel(sample) {
    let nbr = (sample.B08 - sample.B12) / (sample.B08 + sample.B12 + 0.0001);
    return [sample.B04, sample.B03, sample.B02, nbr];
}
"""


def create_config() -> SHConfig:
    """Configure Sentinel Hub avec des identifiants fournis par l'environnement."""
    client_id = os.environ.get("SH_CLIENT_ID")
    client_secret = os.environ.get("SH_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Configurer SH_CLIENT_ID et SH_CLIENT_SECRET dans l'environnement."
        )

    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    return config


def create_bbox(
    coordinates: tuple[float, float, float, float] = DEFAULT_BBOX,
    resolution: int = 60,
) -> tuple[BBox, tuple[int, int]]:
    """Creer la bbox WGS84 et ses dimensions en pixels."""
    bbox = BBox(bbox=coordinates, crs=CRS.WGS84)
    return bbox, bbox_to_dimensions(bbox, resolution=resolution)


def fetch_rgb_nbr(
    bbox: BBox,
    size: tuple[int, int],
    time_interval: tuple[str, str],
    config: SHConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Telecharger une image Sentinel-2 et retourner RGB et NBR."""
    request = SentinelHubRequest(
        evalscript=EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
                mosaicking_order="leastCC",
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    images = request.get_data()
    if not images:
        raise RuntimeError(f"Aucune image Sentinel-2 pour {time_interval}.")

    image = images[0]
    return image[:, :, :3], image[:, :, 3]


def stretch(band: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Etirer le contraste d'une bande avec des percentiles."""
    lower, upper = np.nanpercentile(band, [low, high])
    return np.clip((band - lower) / (upper - lower + 1e-6), 0, 1)


def calculate_rdnbr(nbr_before: np.ndarray, nbr_after: np.ndarray) -> np.ndarray:
    """Calculer le rdNBR a partir des images avant et apres."""
    if nbr_before.shape != nbr_after.shape:
        raise ValueError("Les images NBR avant et apres doivent avoir la meme forme.")
    difference = nbr_before - nbr_after
    return difference / np.sqrt(np.abs(nbr_before) + 1e-6)


def calculate_jenks_breaks(
    values: np.ndarray,
    n_classes: int = 5,
    max_samples: int = 500_000,
    seed: int = 0,
) -> list[float]:
    """Calculer des seuils Jenks sur un echantillon reproductible de pixels valides."""
    finite_values = np.asarray(values).ravel()
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("Impossible de classer une image sans pixel valide.")

    sample_size = min(max_samples, finite_values.size)
    rng = np.random.default_rng(seed)
    sample = rng.choice(finite_values, sample_size, replace=False)
    if np.unique(sample).size < n_classes:
        raise ValueError("Pas assez de valeurs distinctes pour calculer les classes Jenks.")
    return [float(value) for value in jenkspy.jenks_breaks(sample, n_classes=n_classes)]


def classify_pixels(nbr: np.ndarray, rdnbr: np.ndarray) -> pd.DataFrame:
    """Classer chaque pixel par NBR/rdNBR et comparer les deux resultats."""
    if nbr.shape != rdnbr.shape:
        raise ValueError("Les images NBR et rdNBR doivent avoir la meme forme.")

    nbr_breaks = calculate_jenks_breaks(nbr)
    rdnbr_breaks = calculate_jenks_breaks(rdnbr)
    nbr_classes = pd.cut(
        nbr.ravel(), bins=nbr_breaks, labels=SEVERITY_LABELS, include_lowest=True
    )
    rdnbr_classes = pd.cut(
        rdnbr.ravel(), bins=rdnbr_breaks, labels=SEVERITY_LABELS, include_lowest=True
    )

    result = pd.DataFrame(
        {
            "pixel_id": np.arange(1, nbr.size + 1),
            "nbr": nbr.ravel(),
            "rdnbr": rdnbr.ravel(),
            "Classe_NBR": nbr_classes,
            "Classe_RDNBR": rdnbr_classes,
        }
    )
    both_valid = result["Classe_NBR"].notna() & result["Classe_RDNBR"].notna()
    result["Severite_Finale"] = "Hors limites"
    result.loc[both_valid, "Severite_Finale"] = "Indetermine (Mixte)"
    same_class = both_valid & (result["Classe_NBR"] == result["Classe_RDNBR"])
    result.loc[same_class, "Severite_Finale"] = result.loc[same_class, "Classe_NBR"]
    return result


def make_classification_image(
    rgb_after: np.ndarray, rdnbr_classes: pd.Series
) -> np.ndarray:
    """Superposer les classes de severite rdNBR sur l'image RGB apres incendie."""
    image = np.dstack([stretch(rgb_after[:, :, band]) for band in range(3)])
    class_ids = rdnbr_classes.cat.codes.to_numpy().reshape(rgb_after.shape[:2])
    color_table = np.asarray(
        [[int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)] for color in SEVERITY_COLORS]
    )
    valid = class_ids >= 0
    image[valid] = color_table[class_ids[valid]]
    return image


def plot_results(
    rgb_after: np.ndarray,
    nbr_after: np.ndarray,
    rdnbr: np.ndarray,
    classified_image: np.ndarray,
) -> None:
    """Afficher les indices et la carte de classes de severite."""
    figure, axes = plt.subplots(2, 2, figsize=(16, 12))
    rgb_stretched = np.dstack([stretch(rgb_after[:, :, band]) for band in range(3)])
    axes[0, 0].imshow(rgb_stretched)
    axes[0, 0].set_title("Image RGB apres incendie")
    axes[0, 1].imshow(nbr_after, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[0, 1].set_title("NBR apres incendie")
    axes[1, 0].imshow(rdnbr, cmap="RdYlGn_r", vmin=-1, vmax=1)
    axes[1, 0].set_title("rdNBR")
    axes[1, 1].imshow(classified_image)
    axes[1, 1].set_title("Classes de severite rdNBR")

    for axis in axes.flat:
        axis.axis("off")

    legend = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=color, label=label)
        for color, label in zip(SEVERITY_COLORS, SEVERITY_LABELS)
    ]
    axes[1, 1].legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3)
    figure.tight_layout()
    plt.show()


def run_analysis(
    before_date: str = DEFAULT_BEFORE_DATE,
    after_date: str = DEFAULT_AFTER_DATE,
    resolution: int = 60,
) -> pd.DataFrame:
    """Telecharger les images, calculer les indices et afficher les resultats."""
    config = create_config()
    bbox, size = create_bbox(resolution=resolution)
    print(f"Taille demandee : {size}")

    rgb_before, nbr_before = fetch_rgb_nbr(
        bbox, size, (before_date, before_date), config
    )
    rgb_after, nbr_after = fetch_rgb_nbr(bbox, size, (after_date, after_date), config)
    rdnbr = calculate_rdnbr(nbr_before, nbr_after)
    results = classify_pixels(nbr_after, rdnbr)
    classified_image = make_classification_image(rgb_after, results["Classe_RDNBR"])

    print(f"Forme avant : {rgb_before.shape} | apres : {rgb_after.shape}")
    print(f"NBR min/max/moyen : {np.nanmin(nbr_after):.4f} / {np.nanmax(nbr_after):.4f} / {np.nanmean(nbr_after):.4f}")
    print(f"rdNBR min/max/moyen : {np.nanmin(rdnbr):.4f} / {np.nanmax(rdnbr):.4f} / {np.nanmean(rdnbr):.4f}")
    print(results.head(10).to_string(index=False))
    plot_results(rgb_after, nbr_after, rdnbr, classified_image)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-date", default=DEFAULT_BEFORE_DATE)
    parser.add_argument("--after-date", default=DEFAULT_AFTER_DATE)
    parser.add_argument("--resolution", type=int, default=60)
    arguments = parser.parse_args()
    run_analysis(arguments.before_date, arguments.after_date, arguments.resolution)


if __name__ == "__main__":
    main()