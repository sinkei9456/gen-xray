"""새 형상의 흐림 재기: 레이블 500장의 실제 점 이물에서 "가장 어두운 픽셀 대비 바로 옆·대각선·두 칸 옆 픽셀의 어두움 비율"을 재고,
작은 원을 그려 가우시안으로 흐렸을 때 같은 비율이 나오는 흐림 정도(σ)를 찾는다. 새 형상 그리기는 synth/shapes.py.
사용: python develop/measure_shapes.py → data/material_500/shapes_500.json(흐림 측정값), shapes_examples.png(예시)
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np, cv2
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (공통 함수)
sys.path.insert(0, str(Path(__file__).resolve().parent))      # src/develop/
from paths import MATERIAL
from synth.shapes import KINDS, S, _disc_ratios, draw_shape

OUT = Path(os.environ.get("XRAY_MATERIAL", str(MATERIAL)))


def measure_blur():
    """500장의 실제 점 이물(사람 레이블 상자 안 가장 어두운 픽셀)에서 어두움 비율을 재고, 같은 비율이 나오는 원 지름·흐림 σ를 찾는다."""
    from splits import all_paths, test_stems
    from synth.render import load_raw
    from prepare_material import human_boxes, separate
    P = all_paths(); rng = np.random.default_rng(0); q = []
    for s in [s for s in test_stems() if s in P]:
        g, mk = load_raw(P[s]); H, W = g.shape; hb = human_boxes(s, W, H); _, _, mless = separate(g, mk, hb, rng)
        for cx, cy, w, h in hb:
            icx, icy = int(round(cx)), int(round(cy)); r = 10
            if icy - r < 0 or icx - r < 0 or icy + r + 1 > H or icx + r + 1 > W: continue
            big = mless[icy - r:icy + r + 1, icx - r:icx + r + 1]; d = 1 - big / np.maximum(ndimage.median_filter(big, size=7), 1)   # 어두움(1 - 투과율)
            hw, hh = int(np.ceil(w / 2)), int(np.ceil(h / 2)); win = np.full_like(d, -9); win[r - hh:r + hh + 1, r - hw:r + hw + 1] = d[r - hh:r + hh + 1, r - hw:r + hw + 1]
            y0, x0 = np.unravel_index(np.argmax(win), win.shape)
            if not (2 <= y0 <= 2 * r - 2 and 2 <= x0 <= 2 * r - 2) or d[y0, x0] < 0.15: continue
            v = d[y0, x0]
            q.append([(d[y0 - 1, x0] + d[y0 + 1, x0] + d[y0, x0 - 1] + d[y0, x0 + 1]) / 4 / v, (d[y0 - 1, x0 - 1] + d[y0 + 1, x0 + 1] + d[y0 - 1, x0 + 1] + d[y0 + 1, x0 - 1]) / 4 / v,
                      (d[y0 - 2, x0] + d[y0 + 2, x0] + d[y0, x0 - 2] + d[y0, x0 + 2]) / 4 / v])
    real = np.median(np.array(q), axis=0); best = None
    for D in np.arange(1.5, 3.01, 0.25):
        for sig in np.arange(0.0, 1.01, 0.05):
            e = float(np.sum((_disc_ratios(D, sig) - real) ** 2))
            if best is None or e < best[0]: best = (e, float(D), float(sig))
    res = dict(n_dots=len(q), real_ratio_side_diag_two=[round(float(x), 4) for x in real], disc_diameter=best[1], blur_sigma=round(best[2], 2),
               disc_ratio=[round(float(x), 4) for x in _disc_ratios(best[1], best[2])],
               rule="실제 점 이물의 (옆, 대각선, 두 칸 옆)/가장 어두운 픽셀 어두움 비율 중간값에 가장 가까운 원 지름·가우시안 σ")
    return res


def main():
    res = measure_blur(); OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT / "shapes_500.json", "w"), ensure_ascii=False, indent=1); print(res)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(0); fig, ax = plt.subplots(4, 6, figsize=(12, 8))
    for i, kind in enumerate(KINDS):
        for j in range(6):
            p, w, h = draw_shape(kind, rng, res["blur_sigma"], 0.5); r = p.shape[0] // 2
            ax[i, j].imshow(p * 120, cmap="gray", vmin=0, vmax=130, interpolation="nearest")
            ax[i, j].add_patch(plt.Rectangle((r - w / 2, r - h / 2), w, h, fill=False, ec="red", lw=1)); ax[i, j].set_title(f"{kind} {w:.0f}x{h:.0f}", fontsize=8); ax[i, j].axis("off")
    plt.tight_layout(); plt.savefig(OUT / "shapes_examples.png", dpi=90); print("예시:", OUT / "shapes_examples.png")


if __name__ == "__main__":
    main()
