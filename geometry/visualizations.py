"""Plot helpers for hierarchy metrics and concept geometry views."""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import plotting as rplot

def safe_norm(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = v.norm()
    return v / (n + eps)

def plot_heatmaps(
    out_path: Path,
    dist_prox: np.ndarray,
    cos_original: np.ndarray,
    cos_shuffled: np.ndarray,
    title_prefix: str,
) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    mats = [dist_prox, cos_original, cos_shuffled]
    titles = [
        f"{title_prefix}: shortest-path proximity",
        f"{title_prefix}: cosine original",
        f"{title_prefix}: cosine shuffled",
    ]

    for ax, mat, title in zip(axs, mats, titles):
        if mat.size == 0:
            ax.set_title(title + " (empty)")
            ax.axis("off")
            continue
        im = ax.imshow(mat, aspect="auto", cmap="icefire", vmin=-1.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_orthogonality_curves(out_path: Path, metrics: Dict[str, Dict[str, List[float]]], key: str, title: str) -> None:
    plt.figure(figsize=(12, 4))

    def rolling_mean(vals: np.ndarray, win: int = 61) -> np.ndarray:
        if vals.size == 0:
            return vals
        if vals.size < win:
            win = max(3, vals.size // 3)
        if win % 2 == 0:
            win += 1
        if win <= 1:
            return vals
        kernel = np.ones(win, dtype=np.float32) / float(win)
        return np.convolve(vals, kernel, mode="same")

    curves = [
        ("original_parent", "Original", "blue"),
        ("original_random_parent", "Original + Random Parent", "orange"),
        ("shuffled_parent", "Shuffled", "green"),
    ]

    for name, label, color in curves:
        vals = metrics[key][name]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float32)
        x = np.arange(len(arr))
        plt.plot(x, arr, color=color, linewidth=1.0, alpha=0.2)
        plt.plot(x, rolling_mean(arr), color=color, linewidth=2.0, alpha=0.95, label=label)

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.ylim(-1.0, 1.0)
    plt.title(title)
    plt.xlabel("Hierarchy edges")
    plt.ylabel("Cosine")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


def plot_projection_feature_figure(out_path: Path, proj_stats: Dict[str, dict]) -> None:
    def draw_panel(ax, data: dict, title: str):
        n = len(data["train_mean"])
        if n == 0:
            ax.set_title(title + " (empty)")
            ax.axis("off")
            return

        x = np.arange(n)
        colors = {
            "train": "#7ac77b",
            "test": "#2938ff",
            "random": "#f6b43f",
        }

        for split in ["train", "test", "random"]:
            mean = np.asarray(data[f"{split}_mean"], dtype=np.float32)
            std = np.asarray(data[f"{split}_std"], dtype=np.float32)
            ax.plot(x, mean, color=colors[split], linewidth=1.2, label=split)
            ax.errorbar(x, mean, yerr=std, fmt="none", ecolor=colors[split], alpha=0.18, capsize=0)

        ax.set_ylim(-1.0, 2.0)
        ax.set_title(title)
        ax.set_xlabel("Binary Features in Hierarchy")

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    draw_panel(axs[0], proj_stats["original"], "Original Unembeddings")
    draw_panel(axs[1], proj_stats["shuffled"], "Shuffled Unembeddings")
    axs[0].set_ylabel(r"$(g(y)^\top \bar{\ell}_w) / \|\bar{\ell}_w\|^2$")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

def run_visual_2d(
    out_path: Path,
    g_whitened: torch.Tensor,
    vocab_list: List[str],
    idx_sets: Dict[str, List[int]],
    dirs: Dict[int, Dict[str, torch.Tensor]],
    ids: Dict[str, int],
) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(24, 7))

    animal = ids["animal"]
    plant = ids["plant"]
    mammal = ids["mammal"]
    bird = ids["bird"]

    colors = {
        "animal": "#f64369",
        "mammal": "#2aab8c",
        "bird": "#48d1e8",
        "plant": "#5170ff",
    }

    inds0 = {"animal": idx_sets["animal"], "mammal": idx_sets["mammal"]}
    inds1 = {"animal": idx_sets["animal"], "mammal": idx_sets["mammal"], "bird": idx_sets["bird"]}
    inds2 = {
        "plant": idx_sets["plant"],
        "animal": idx_sets["animal"],
        "mammal": idx_sets["mammal"],
        "bird": idx_sets["bird"],
    }

    rplot.proj_2d(
        dirs[animal]["lda"],
        dirs[mammal]["lda"],
        g_whitened,
        vocab_list,
        axs[0],
        is_plain=True,
        double=False,
        higher1=None,
        subcat1=None,
        normalize=True,
        orthogonal=True,
        added_inds=inds0,
        category_colors=colors,
        k=50,
        fontsize=11,
        draw_arrows=True,
        arrow1_name="animal",
        arrow2_name="mammal",
        alpha=0.03,
        s=0.05,
        target_alpha=0.65,
        target_s=4,
        left_topk=False,
        right_topk=False,
        top_topk=False,
        bottom_topk=False,
        xlabel="",
        ylabel="",
        title="animal vs mammal",
    )

    rplot.proj_2d_single_diff(
        dirs[animal]["lda"],
        dirs[mammal]["lda"],
        dirs[bird]["lda"],
        g_whitened,
        vocab_list,
        axs[1],
        normalize=True,
        orthogonal=True,
        added_inds=inds1,
        category_colors=colors,
        k=50,
        fontsize=11,
        draw_arrows=True,
        arrow1_name="animal",
        arrow2_name="bird - mammal",
        alpha=0.03,
        s=0.05,
        target_alpha=0.65,
        target_s=4,
        left_topk=False,
        right_topk=False,
        top_topk=False,
        bottom_topk=False,
        xlabel="",
        ylabel="",
        title="animal vs mammal -> bird",
    )

    rplot.proj_2d_double_diff(
        dirs[plant]["lda"],
        dirs[animal]["lda"],
        dirs[mammal]["lda"],
        dirs[bird]["lda"],
        g_whitened,
        vocab_list,
        axs[2],
        normalize=True,
        orthogonal=True,
        added_inds=inds2,
        category_colors=colors,
        k=50,
        fontsize=11,
        draw_arrows=True,
        arrow1_name="animal - plant",
        arrow2_name="bird - mammal",
        alpha=0.03,
        s=0.05,
        target_alpha=0.65,
        target_s=4,
        left_topk=False,
        right_topk=False,
        top_topk=False,
        bottom_topk=False,
        xlabel="",
        ylabel="",
        title="plant -> animal vs mammal -> bird",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def run_visual_3d(
    out_path: Path,
    g_whitened: torch.Tensor,
    idx_sets: Dict[str, List[int]],
    dirs: Dict[int, Dict[str, torch.Tensor]],
    ids: Dict[str, int],
) -> None:
    mammal = ids["mammal"]
    bird = ids["bird"]
    fish = ids["fish"]
    reptile = ids["reptile"]
    animal = ids["animal"]

    fig = plt.figure(figsize=(20, 8))

    ax = fig.add_subplot(121, projection="3d")

    dir1 = dirs[mammal]["lda"]
    dir2 = dirs[bird]["lda"]
    dir3 = dirs[fish]["lda"]
    higher_dir = dirs[animal]["lda"]

    xaxis = safe_norm(dir1)
    yaxis = safe_norm(dir2 - (dir2 @ xaxis) * xaxis)
    zaxis = safe_norm(dir3 - (dir3 @ xaxis) * xaxis - (dir3 @ yaxis) * yaxis)
    axes = torch.stack([xaxis, yaxis, zaxis], dim=1)

    g1 = g_whitened[idx_sets["mammal"]]
    g2 = g_whitened[idx_sets["bird"]]
    g3 = g_whitened[idx_sets["fish"]]

    proj1 = (g1 @ axes).cpu().numpy()
    proj2 = (g2 @ axes).cpu().numpy()
    proj3 = (g3 @ axes).cpu().numpy()
    proj = (g_whitened @ axes).cpu().numpy()

    P1 = (dir1 @ axes).cpu().numpy()
    P2 = (dir2 @ axes).cpu().numpy()
    P3 = (dir3 @ axes).cpu().numpy()
    P4 = (higher_dir @ axes).cpu().numpy()

    ax.scatter(P1[0], P1[1], P1[2], color="#f64369", s=100)
    ax.scatter(P2[0], P2[1], P2[2], color="#2aab8c", s=100)
    ax.scatter(P3[0], P3[1], P3[2], color="#48d1e8", s=100)

    verts = [list(zip([P1[0], P2[0], P3[0]], [P1[1], P2[1], P3[1]], [P1[2], P2[2], P3[2]]))]
    triangle = Poly3DCollection(verts, alpha=0.2, linewidths=1, linestyle="--", edgecolors="#5170ff")
    triangle.set_facecolor("#ffb700")
    ax.add_collection3d(triangle)

    ax.quiver(0, 0, 0, P1[0], P1[1], P1[2], color="#f64369", arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P2[0], P2[1], P2[2], color="#2aab8c", arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P3[0], P3[1], P3[2], color="#48d1e8", arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P4[0], P4[1], P4[2], color="#5170ff", arrow_length_ratio=0.1, linewidth=2)

    ax.scatter(proj1[:, 0], proj1[:, 1], proj1[:, 2], c="#f64369", label="mammal")
    ax.scatter(proj2[:, 0], proj2[:, 1], proj2[:, 2], c="#2aab8c", label="bird")
    ax.scatter(proj3[:, 0], proj3[:, 1], proj3[:, 2], c="#48d1e8", label="fish")
    ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c="grey", s=0.05, alpha=0.03)

    scale = 1.2
    ax.text(P1[0] * scale + 2, P1[1] * scale, P1[2] * scale, "mammal", bbox=dict(facecolor="#f64369", alpha=0.2))
    ax.text(P2[0] * scale + 0.5, P2[1] * scale + 0.5, P2[2] * scale, "bird", bbox=dict(facecolor="#2aab8c", alpha=0.2))
    ax.text(P3[0] * scale, P3[1] * scale, P3[2] * scale, "fish", bbox=dict(facecolor="#48d1e8", alpha=0.2))
    ax.text(P4[0] - 0.6, P4[1] - 0.6, P4[2], "animal", bbox=dict(facecolor="#5170ff", alpha=0.2))
    ax.view_init(elev=20, azim=75)

    ax = fig.add_subplot(122, projection="3d")

    dir4 = dirs[reptile]["lda"]

    xaxis = safe_norm(dir2 - dir1)
    yaxis = safe_norm((dir3 - dir1) - ((dir3 - dir1) @ xaxis) * xaxis)
    zaxis = safe_norm((dir4 - dir1) - ((dir4 - dir1) @ xaxis) * xaxis - ((dir4 - dir1) @ yaxis) * yaxis)
    axes = torch.stack([xaxis, yaxis, zaxis], dim=1)

    g4 = g_whitened[idx_sets["reptile"]]

    proj1 = (g_whitened[idx_sets["mammal"]] @ axes).cpu().numpy()
    proj2 = (g_whitened[idx_sets["bird"]] @ axes).cpu().numpy()
    proj3 = (g_whitened[idx_sets["fish"]] @ axes).cpu().numpy()
    proj4 = (g4 @ axes).cpu().numpy()
    proj = (g_whitened @ axes).cpu().numpy()

    P1 = (dir1 @ axes).cpu().numpy()
    P2 = (dir2 @ axes).cpu().numpy()
    P3 = (dir3 @ axes).cpu().numpy()
    P4 = (dir4 @ axes).cpu().numpy()

    ax.scatter(P1[0], P1[1], P1[2], color="#f64369", s=100)
    ax.scatter(P2[0], P2[1], P2[2], color="#2aab8c", s=100)
    ax.scatter(P3[0], P3[1], P3[2], color="#48d1e8", s=100)
    ax.scatter(P4[0], P4[1], P4[2], color="#5170ff", s=100)

    triangles = [
        [P1, P2, P3],
        [P1, P2, P4],
        [P1, P3, P4],
        [P2, P3, P4],
    ]
    for tri, alpha in zip(triangles, [0.1, 0.2, 0.1, 0.1]):
        verts = [list(zip([tri[0][0], tri[1][0], tri[2][0]],
                          [tri[0][1], tri[1][1], tri[2][1]],
                          [tri[0][2], tri[1][2], tri[2][2]]))]
        poly = Poly3DCollection(verts, alpha=alpha, linewidths=1, linestyle="--", edgecolors="#5170ff")
        poly.set_facecolor("#ffb700")
        ax.add_collection3d(poly)

    ax.quiver(0, 0, 0, P1[0], P1[1], P1[2], color="#f64369", arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P2[0], P2[1], P2[2], color="#2aab8c", arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P3[0], P3[1], P3[2], color="#48d1e8", arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P4[0], P4[1], P4[2], color="#5170ff", arrow_length_ratio=0.01)

    ax.scatter(proj1[:, 0], proj1[:, 1], proj1[:, 2], c="#f64369", label="mammal")
    ax.scatter(proj2[:, 0], proj2[:, 1], proj2[:, 2], c="#2aab8c", label="bird")
    ax.scatter(proj3[:, 0], proj3[:, 1], proj3[:, 2], c="#48d1e8", label="fish")
    ax.scatter(proj4[:, 0], proj4[:, 1], proj4[:, 2], c="#5170ff", label="reptile")
    ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c="gray", s=0.05, alpha=0.01)

    scale = 1.4
    scale2 = 1.2
    ax.text(P1[0] * scale - 1, P1[1] * scale, P1[2] * scale, "mammal", bbox=dict(facecolor="#f64369", alpha=0.2))
    ax.text(P2[0] * scale + 1, P2[1] * scale, P2[2] * scale, "bird", bbox=dict(facecolor="#2aab8c", alpha=0.2))
    ax.text(P3[0] * scale - 1, P3[1] * scale, P3[2] * scale, "fish", bbox=dict(facecolor="#48d1e8", alpha=0.2))
    ax.text(P4[0] * scale2 + 2, P4[1] * scale2, P4[2] * scale2 - 1, "reptile", bbox=dict(facecolor="#5170ff", alpha=0.2))

    plt.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
