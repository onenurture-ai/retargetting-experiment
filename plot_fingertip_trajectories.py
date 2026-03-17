"""
Reads fingertip_coords.csv and plots the 3D spatial trajectory curves for each fingertip of
the Allegro and Shadow hands in two separate 3D plots (coordinates are in each hand's own base frame).
Usage: python plot_fingertip_trajectories.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "fingertip_coords.csv")

# Common CJK font paths on the system (tried in priority order); register and set as default once found
_CJK_FONT_PATHS = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    os.path.expanduser("~/.local/share/fonts/NotoSansCJK-Regular.ttc"),
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]

def _get_chinese_font():
    """Returns a FontProperties object for a CJK-capable font, or None if no CJK font is found (English titles will be used instead)."""
    for path in _CJK_FONT_PATHS:
        if os.path.isfile(path):
            try:
                return fm.FontProperties(fname=path)
            except Exception:
                pass
    for f in fm.fontManager.ttflist:
        try:
            name = (f.name or "") + (getattr(f, "family", "") or "")
            if any(k in name for k in ("Noto Sans CJK", "WenQuanYi", "WQY", "SimHei", "YaHei", "PingFang", "Heiti", "Micro Hei", "Zen Hei")):
                if getattr(f, "fname", None) and os.path.isfile(f.fname):
                    return fm.FontProperties(fname=f.fname)
        except Exception:
            pass
    return None

_CHINESE_FONT = _get_chinese_font()
plt.rcParams["axes.unicode_minus"] = False


def main():
    if not os.path.isfile(CSV_PATH):
        print(f"File not found: {CSV_PATH}")
        return

    data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_frames = data.shape[0]
    # Column indices: frame=0, allegro thumb(1,2,3), index(4,5,6), middle(7,8,9), ring(10,11,12),
    #        shadow thtip(13,14,15), fftip(16,17,18), mftip(19,20,21), rftip(22,23,24), lftip(25,26,27)
    allegro_tips = [
        (data[:, 1], data[:, 2], data[:, 3], "thumb"),
        (data[:, 4], data[:, 5], data[:, 6], "index"),
        (data[:, 7], data[:, 8], data[:, 9], "middle"),
        (data[:, 10], data[:, 11], data[:, 12], "ring"),
    ]
    shadow_tips = [
        (data[:, 13], data[:, 14], data[:, 15], "thtip"),
        (data[:, 16], data[:, 17], data[:, 18], "fftip"),
        (data[:, 19], data[:, 20], data[:, 21], "mftip"),
        (data[:, 22], data[:, 23], data[:, 24], "rftip"),
        (data[:, 25], data[:, 26], data[:, 27], "lftip"),
    ]

    fig = plt.figure(figsize=(14, 6))

    # Mark frame 120 (the last data point) with a star to help distinguish trajectory order
    idx_end = min(119, n_frames - 1)

    # Allegro 3D curves
    ax1 = fig.add_subplot(121, projection="3d")
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    for (x, y, z, name), c in zip(allegro_tips, colors):
        ax1.plot(x, y, z, label=name, color=c, linewidth=1.5)
        ax1.scatter(x[idx_end], y[idx_end], z[idx_end], color=c, marker="*", s=120, edgecolors="k", linewidths=0.5, zorder=5)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    if _CHINESE_FONT is not None:
        ax1.set_title("Allegro fingertip trajectories (4 fingers, Allegro base frame)", fontproperties=_CHINESE_FONT)
    else:
        ax1.set_title("Allegro fingertip trajectories (4 fingers, Allegro base frame)")
    ax1.legend(prop=_CHINESE_FONT if _CHINESE_FONT is not None else None)
    ax1.set_box_aspect([1, 1, 1])

    # Shadow 3D curves
    ax2 = fig.add_subplot(122, projection="3d")
    colors2 = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    for (x, y, z, name), c in zip(shadow_tips, colors2):
        ax2.plot(x, y, z, label=name, color=c, linewidth=1.5)
        ax2.scatter(x[idx_end], y[idx_end], z[idx_end], color=c, marker="*", s=120, edgecolors="k", linewidths=0.5, zorder=5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    if _CHINESE_FONT is not None:
        ax2.set_title("Shadow fingertip trajectories (5 fingers, Shadow base frame)", fontproperties=_CHINESE_FONT)
    else:
        ax2.set_title("Shadow fingertip trajectories (5 fingers, Shadow base frame)")
    ax2.legend(prop=_CHINESE_FONT if _CHINESE_FONT is not None else None)
    ax2.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "fingertip_trajectories.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
