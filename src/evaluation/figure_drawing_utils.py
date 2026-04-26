import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def draw_rounded_box(ax, xy, width, height, box_color, text, text_color="black", fontsize=10, alpha=1.0, edgecolor=None):
    rect = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.1",
        facecolor=box_color,
        edgecolor=edgecolor if edgecolor else box_color,
        alpha=alpha,
        zorder=2
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2, xy[1] + height / 2,
        text,
        color=text_color,
        ha="center", va="center",
        fontsize=fontsize,
        zorder=3,
        wrap=True
    )
    return rect

def draw_3d_block(ax, xy, width, height, depth, color, text, fontsize=9):
    # Front face
    front = patches.Rectangle(xy, width, height, facecolor=color, edgecolor="black", zorder=2)
    ax.add_patch(front)
    # Top face
    top_points = [
        [xy[0], xy[1] + height],
        [xy[0] + depth, xy[1] + height + depth],
        [xy[0] + width + depth, xy[1] + height + depth],
        [xy[0] + width, xy[1] + height]
    ]
    top = patches.Polygon(top_points, facecolor=color, edgecolor="black", alpha=0.8, zorder=1)
    ax.add_patch(top)
    # Right face
    right_points = [
        [xy[0] + width, xy[1]],
        [xy[0] + width + depth, xy[1] + depth],
        [xy[0] + width + depth, xy[1] + height + depth],
        [xy[0] + width, xy[1] + height]
    ]
    right = patches.Polygon(right_points, facecolor=color, edgecolor="black", alpha=0.9, zorder=1)
    ax.add_patch(right)
    
    ax.text(
        xy[0] + width / 2, xy[1] + height / 2,
        text,
        color="black",
        ha="center", va="center",
        fontsize=fontsize,
        zorder=3
    )

def draw_arrow(ax, xy_from, xy_to, text="", color="black"):
    ax.annotate(
        text,
        xy=xy_to, xycoords='data',
        xytext=xy_from, textcoords='data',
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5, shrinkA=5, shrinkB=5),
        ha='center', va='center',
        fontsize=9,
        zorder=1
    )

def add_image_thumbnail(ax, image_path, xy, width, height, title=None):
    try:
        img = Image.open(image_path).convert("RGB")
        ax.imshow(img, extent=[xy[0], xy[0]+width, xy[1], xy[1]+height], zorder=2)
        ax.add_patch(patches.Rectangle(xy, width, height, fill=False, edgecolor="black", lw=1, zorder=3))
        if title:
            ax.text(xy[0]+width/2, xy[1]-height*0.1, title, ha="center", va="top", fontsize=8)
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")

def save_figure_all_formats(fig, output_path, name):
    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / f"{name}.png", dpi=600, bbox_inches='tight')
    fig.savefig(output_path / f"{name}.pdf", bbox_inches='tight')
    fig.savefig(output_path / f"{name}.svg", bbox_inches='tight')
