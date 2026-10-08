"""정답 있는 500장 → (봉지·막대만 남은 사진, 이물 패치, 생성기 설정값, Diffusion 학습 캐시).

- 봉지·막대 사진: 색 사각형과 그 안의 이물(사람 정답 상자)을 지우고 주변 픽셀로 메운다.
- 이물 패치: 이물 위치의 투과율 = (색 선만 지운 사진) ÷ (그 사진의 국소 밝기 = 7×7 가운데값)(cut_patch). X-ray 감쇠는 곱셈이므로(I = I0·exp(-μt))
  새 사진에 넣을 때는 그 위치의 밝기에 이 투과율을 곱한다.
- 생성기 설정값(장비별 사진 크기·밝기·잡음·봉지 크기·막대 폭·봉지 중심 범위)도 이 500장에서만 잰다.
사용: python develop/prepare_material.py  → data/material_500/ (bags/, patches.npz, profiles_500.json, levels_500.json, diffusion_cache_500.npz, prompt_stats.json)
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np, cv2
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (공통 함수)
sys.path.insert(0, str(Path(__file__).resolve().parent))      # src/develop/
from splits import all_paths, test_stems, machine_of, LABEL_DIR
from synth.render import load_raw, marker_boxes, product_masks, MACHINES
from sd.data import square_pad, layout_map

from paths import MATERIAL
OUT = Path(os.environ.get("XRAY_MATERIAL", str(MATERIAL)))   # 재현 점검 때는 다른 폴더로


def human_boxes(stem, W, H):
    out = []
    for line in (LABEL_DIR / f"{stem}.txt").read_text().strip().splitlines():
        c, cx, cy, w, h = map(float, line.split()); out.append((cx * W, cy * H, w * W, h * H))
    return out


def stick_angle(img):
    """막대(없으면 봉지)의 길이 방향 각도(라디안)"""
    pm = product_masks(img)
    if pm is None: return 0.0
    prod, stick = pm; msk = stick if stick.sum() > 300 else prod
    ys, xs = np.where(msk); (_, _), (w, h), ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
    return float(np.deg2rad(ang if w >= h else ang + 90))


def fill_across(img, m, ang, max_step=30):
    """지운 픽셀을 막대와 직각 방향으로 양쪽의 안 지운 픽셀을 이어서 메운다.
    이물이 막대 끝에 있어 주변 픽셀로 메우면 막대 끝 모서리가 둥글게 무너지는데, 직각 방향으로 이으면 모서리가 그대로 남는다."""
    out = img.copy(); H, W = img.shape; nx, ny = -np.sin(ang), np.cos(ang)
    for y, x in np.argwhere(m):
        vals = []
        for sgn in (1, -1):
            for k in range(1, max_step):
                yy, xx = int(round(y + sgn * k * ny)), int(round(x + sgn * k * nx))
                if not (0 <= yy < H and 0 <= xx < W): break
                if not m[yy, xx]: vals.append((k, img[yy, xx])); break
        if len(vals) == 2: (k1, v1), (k2, v2) = vals; out[y, x] = (v1 * k2 + v2 * k1) / (k1 + k2)
        elif vals: out[y, x] = vals[0][1]
    return out


def separate(g, mk, hb, rng):
    """→ (봉지·막대 사진, 이물까지 지운 매끈한 사진, 색 선만 지운 사진)"""
    line = ndimage.binary_dilation(mk, iterations=1)
    markerless = cv2.inpaint(g.astype(np.uint8), line.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA).astype(np.float32)
    H, W = g.shape; m = np.zeros_like(line)
    for cx, cy, w, h in hb:
        m[max(0, int(cy - h / 2) - 1):min(H, int(np.ceil(cy + h / 2)) + 1), max(0, int(cx - w / 2) - 1):min(W, int(np.ceil(cx + w / 2)) + 1)] = True
    smooth = fill_across(markerless, m, stick_angle(markerless))
    # 메운 부분에 주변과 같은 크기의 잡음을 넣어 매끈한 얼룩이 남지 않게 한다
    hf = markerless - cv2.GaussianBlur(markerless, (0, 0), 1.5); ring = ndimage.binary_dilation(m, iterations=6) & ~m
    sd = float(hf[ring].std()) if ring.any() else 2.0
    nz = cv2.GaussianBlur(rng.normal(0, 1, g.shape).astype(np.float32), (0, 0), 0.7); nz *= sd / max(nz.std(), 1e-6)
    bag = smooth.copy(); bag[m] += nz[m]
    return np.clip(bag, 0, 255), smooth, markerless


def cut_patch(smooth, markerless, cx, cy, w, h):
    """이물의 어두운 점만 남긴 투과율 패치(점 밖은 1 = 변화 없음).
    투과율 = 사진 / 그 위치의 국소 밝기(7×7 가운데값: 3픽셀 남짓한 점은 지워지고 막대·봉지 밝기만 남음).
    정답 상자 안에서 가장 어두운 픽셀을 찾고, 그 어두움의 절반 이상 어두운 이웃 픽셀까지를 이물로 본다.
    이 영역을 1픽셀 넓히고 가장자리는 흐림(σ 0.7)으로 부드럽게 잇는다."""
    r = int(np.ceil(max(w, h) / 2)) + 2; icx, icy = int(round(cx)), int(round(cy)); H, W = markerless.shape
    if icy - r - 4 < 0 or icx - r - 4 < 0 or icy + r + 5 > H or icx + r + 5 > W: return None
    big = markerless[icy - r - 4:icy + r + 5, icx - r - 4:icx + r + 5]
    ratio = (big / np.maximum(ndimage.median_filter(big, size=7), 1.0))[4:-4, 4:-4]; n = 2 * r + 1
    hw, hh = int(np.ceil(w / 2)), int(np.ceil(h / 2)); box = np.full((n, n), np.inf); box[r - hh:r + hh + 1, r - hw:r + hw + 1] = ratio[r - hh:r + hh + 1, r - hw:r + hw + 1]
    y0, x0 = np.unravel_index(np.argmin(box), box.shape); dmin = 1 - ratio[y0, x0]
    if dmin <= 0.02: return None
    lab, _ = ndimage.label(ratio < 1 - dmin / 2); core = lab == lab[y0, x0]
    core = ndimage.binary_dilation(core, iterations=1); soft = np.maximum(cv2.GaussianBlur(core.astype(np.float32), (0, 0), 0.7), core)
    p = np.clip(1 + (np.minimum(ratio, 1.0) - 1) * soft, 0.1, 1.0).astype(np.float32)
    p = np.roll(np.roll(p, r - y0, 0), r - x0, 1)   # 가장 어두운 픽셀을 패치 가운데로
    return p, float(dmin * np.median(big))


def stick_pos(stick, cx, cy):
    """이물이 막대 안 어디에 있는지: 막대 길이 방향 위치 u(-1~1, 0=가운데)"""
    lab, n = ndimage.label(stick); icx, icy = int(round(cx)), int(round(cy))
    k = lab[icy, icx] if 0 <= icy < lab.shape[0] and 0 <= icx < lab.shape[1] else 0
    if k == 0:
        d = ndimage.distance_transform_edt(lab == 0, return_indices=True)[1]; k = lab[d[0][icy, icx], d[1][icy, icx]]
        if k == 0: return None, False
    ys, xs = np.where(lab == k); (scx, scy), (w_, h_), ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
    L = max(w_, h_); a = np.deg2rad(ang if w_ >= h_ else ang + 90)
    return float(((cx - scx) * np.cos(a) + (cy - scy) * np.sin(a)) / (L / 2)), bool(stick[icy, icx])


def main():
    (OUT / "bags").mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(0)
    P = all_paths(); stems = [s for s in test_stems() if s in P]
    patches, pmeta, prof_acc, cache = [], [], {m: {} for m in MACHINES}, dict(img=[], layout=[], machine=[], stem=[])
    def add(m, k, v): prof_acc[m].setdefault(k, []).append(float(v))
    for i, s in enumerate(stems):
        g, mk = load_raw(P[s]); H, W = g.shape; m = machine_of(P[s]); hb = human_boxes(s, W, H)
        bag, smooth, markerless = separate(g, mk, hb, rng)
        cv2.imwrite(str(OUT / "bags" / f"{s}.png"), bag.astype(np.uint8))
        pm = product_masks(bag)
        for cx, cy, w, h in hb:
            r = cut_patch(smooth, markerless, cx, cy, w, h)
            if r is None: continue
            u, inside = stick_pos(pm[1], cx, cy) if pm is not None else (None, False)
            patches.append(r[0]); pmeta.append(dict(stem=s, machine=m, w=w, h=h, depth=r[1], u=u, in_stick=inside))
        if pm is None: continue
        prod, stick = pm; pack = prod & ~stick; bgm = ~ndimage.binary_dilation(prod, iterations=6); bgm[:, :4] = False
        hp = bag - cv2.GaussianBlur(bag, (0, 0), 1.5)
        add(m, "W", W); add(m, "H", H)
        for k, msk in [("bg", bgm), ("pack", pack), ("stick", stick)]:
            if msk.sum() > 50: add(m, k, np.median(bag[msk])); add(m, "noise_" + k, np.std(hp[msk]))
        ys, xs = np.where(prod); (bcx, bcy), (w, h), _ = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
        add(m, "bag_l", max(w, h)); add(m, "bag_s", min(w, h)); add(m, "cx", bcx / W); add(m, "cy", bcy / H)
        lab, n = ndimage.label(stick)
        for k in range(1, n + 1):
            yy, xx = np.where(lab == k)
            if len(yy) < 300: continue
            (_, _), (w2, h2), _ = cv2.minAreaRect(np.stack([xx, yy], 1).astype(np.float32)); add(m, "stick_w", min(w2, h2)); add(m, "stick_len", max(w2, h2) / max(max(w, h), 1))
        add(m, "n_sticks", sum(1 for k in range(1, n + 1) if (lab == k).sum() >= 300))
        sq, _ = square_pad(bag, 210.0); lm, _ = square_pad(layout_map(prod, stick), 0)
        cache["img"].append(cv2.resize(sq, (256, 256), interpolation=cv2.INTER_AREA).astype(np.uint8))
        cache["layout"].append(cv2.resize(lm, (256, 256), interpolation=cv2.INTER_NEAREST)); cache["machine"].append(MACHINES.index(m)); cache["stem"].append(s)
        if i % 100 == 0: print(f"{i}/{len(stems)}", flush=True)
    prof = {}
    for m, acc in prof_acc.items():
        med = {k: float(np.median(v)) for k, v in acc.items()}
        prof[m] = dict(W=int(med["W"]), H=int(med["H"]), bg=med["bg"], pack=med["pack"], stick=med["stick"], noise_bg=med["noise_bg"], noise_pack=med["noise_pack"],
                       noise_stick=med["noise_stick"], bag=(med["bag_l"], med["bag_s"]), stick_w=med["stick_w"], stick_len=med["stick_len"],
                       cx=(float(np.percentile(acc["cx"], 5)), float(np.percentile(acc["cx"], 95))), cy=(float(np.percentile(acc["cy"], 5)), float(np.percentile(acc["cy"], 95))),
                       n_images=len(acc["W"]))
    json.dump(prof, open(OUT / "profiles_500.json", "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(OUT / "patches.npz", patch=np.array(patches, dtype=object), meta=json.dumps(pmeta, ensure_ascii=False))
    np.savez_compressed(OUT / "diffusion_cache_500.npz", img=np.stack(cache["img"]), layout=np.stack(cache["layout"]), machine=np.array(cache["machine"]), stem=np.array(cache["stem"]))
    measure_levels(OUT)
    from synth.compose import prompt_stats   # 합성 지시어의 장비·막대 수 비율(release/generate.py 가 대회 데이터 없이 쓰도록 저장)
    json.dump(prompt_stats(), open(OUT / "prompt_stats.json", "w"), ensure_ascii=False, indent=1)
    u = np.array([abs(p["u"]) for p in pmeta if p["u"] is not None]); ins = np.mean([p["in_stick"] for p in pmeta])
    print(json.dumps(prof, ensure_ascii=False, indent=1))
    print(f"사진 {len(stems)}장, 봉지·막대 캐시 {len(cache['img'])}장, 이물 패치 {len(patches)}개, 막대 안 {ins:.3f}, |u| 분위 {np.percentile(u, [10, 25, 50, 75, 90]).round(2)}")


def measure_levels(out: Path):
    """500장 봉지·막대 이미지 각각의 바탕·봉지·막대 밝기와 봉지 기울기 → levels_500.json (합성 이미지 밝기 맞추기·기울기에 씀)"""
    P = all_paths(); res = {}
    for s in test_stems():
        if s not in P: continue
        g = cv2.imread(str(out / "bags" / f"{s}.png"), 0).astype(np.float32); pm = product_masks(g)
        if pm is None: continue
        prod, st = pm; pack = prod & ~ndimage.binary_dilation(st, iterations=3); bgm = ~ndimage.binary_dilation(prod, iterations=6); bgm[:, :4] = False
        ys, xs = np.where(prod); (_, _), (w, h), ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
        a_ = ang if w >= h else ang - 90; a_ = (a_ + 90) % 180 - 90
        if a_ > 45: a_ -= 90
        if a_ < -45: a_ += 90
        res.setdefault(machine_of(P[s]), []).append(dict(bg=float(np.median(g[bgm])), pack=float(np.median(g[pack])),
                                                          stick=float(np.median(g[ndimage.binary_erosion(st, iterations=2)])) if st.sum() > 200 else None, angle=float(a_)))
    json.dump(res, open(out / "levels_500.json", "w"), ensure_ascii=False)


def patches_only():
    """봉지·막대 사진과 Diffusion 학습 캐시는 두고 이물 패치만 다시 만든다"""
    rng = np.random.default_rng(0); P = all_paths(); stems = [s for s in test_stems() if s in P]; patches, pmeta = [], []
    for s in stems:
        g, mk = load_raw(P[s]); H, W = g.shape; m = machine_of(P[s]); hb = human_boxes(s, W, H)
        bag, smooth, markerless = separate(g, mk, hb, rng); pm = product_masks(bag)
        for cx, cy, w, h in hb:
            r = cut_patch(smooth, markerless, cx, cy, w, h)
            if r is None: continue
            u, inside = stick_pos(pm[1], cx, cy) if pm is not None else (None, False)
            patches.append(r[0]); pmeta.append(dict(stem=s, machine=m, w=w, h=h, depth=r[1], u=u, in_stick=inside))
    np.savez_compressed(OUT / "patches.npz", patch=np.array(patches, dtype=object), meta=json.dumps(pmeta, ensure_ascii=False))
    print("이물 패치", len(patches), "개, 패치 안 어두운 점 픽셀 수 중간값", float(np.median([int((p < 0.97).sum()) for p in patches])))


if __name__ == "__main__":
    patches_only() if "--patches-only" in sys.argv else main()
