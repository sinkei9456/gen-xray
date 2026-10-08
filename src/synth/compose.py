"""합성 이미지 한 장 만들기(develop/make_synth.py 와 release/generate.py 가 함께 씀).
1) 지시어("2호기, 스틱 3개, 이물 3개")를 규칙으로 읽어 배치도(봉지·막대 위치)를 만든다(synth.layout).
2) 빈 바탕 화면에서 출발해 Diffusion 모델이 배치도대로 봉지와 막대를 그린다(sd.sample). 모델은 256×256으로 그리고, 결과를 장비 크기로 되돌린다.
3) 레이블 500장 중 같은 장비 사진 한 장의 바탕·봉지·막대 밝기에 맞춘다(match_tone).
4) 이물(500장에서 떼어 낸 점 패치, 또는 shapes 비율만큼 새 형상)을 막대 끝 쪽 안에 투과율로 곱해 넣는다. 넣은 위치가 곧 레이블 상자다.
재료(material): data/material_500 (profiles_500.json, levels_500.json, patches.npz, shapes_500.json, prompt_stats.json)
"""
from __future__ import annotations
import collections, json
from pathlib import Path
import numpy as np, cv2
from scipy import ndimage
from .layout import parse_prompt
from .render import PROFILES, MACHINES, procedural_background
from .shapes import KINDS, draw_shape
import sd.sample as sds


def load_profiles(material):
    p = json.load(open(Path(material) / "profiles_500.json"))
    for m, v in p.items():
        v["bag"] = tuple(v["bag"]); v["cx"] = tuple(v["cx"]); v["cy"] = tuple(v["cy"]); v["taper"] = False   # 실제 막대 끝은 네모(뾰족하지 않음)
        PROFILES[m] = v
    return p


def load_levels(material):
    """500장 각각의 바탕·봉지·막대 밝기와 봉지 기울기(장비별 목록)"""
    return json.load(open(Path(material) / "levels_500.json"))


def match_tone(img, bag, stick, lv):
    """Diffusion 모델이 그린 사진의 바탕·봉지·막대 밝기를 실제 사진 한 장(lv)의 밝기에 맞춘다(세 밝기를 그 사진의 값으로 옮기고, 그 사이 밝기는 직선으로 이어 바꿈).
    Diffusion 모델이 그린 그대로는 실제보다 밝게 나와(500장과 비교한 장비별 중간값 차이: 바탕 +5~8, 봉지 +21~38, 막대 +23~41) 이를 바로잡는다."""
    pack = bag & ~ndimage.binary_dilation(stick, iterations=3); bgm = ~ndimage.binary_dilation(bag, iterations=6); st = ndimage.binary_erosion(stick, iterations=2)
    if lv["stick"] is None or st.sum() < 50 or pack.sum() < 50: return img
    src = [0.0, float(np.median(img[st])), float(np.median(img[pack])), float(np.median(img[bgm])), 255.0]
    dst = [0.0, lv["stick"], lv["pack"], lv["bg"], 255.0]
    if not (np.all(np.diff(src) > 0) and np.all(np.diff(dst) > 0)): return img
    return np.interp(img, src, dst).astype(np.float32)


def load_patches(material):
    z = np.load(Path(material) / "patches.npz", allow_pickle=True); meta = json.loads(str(z["meta"]))
    by_m = collections.defaultdict(list)
    for p, mt in zip(z["patch"], meta): by_m[mt["machine"]].append((p, mt))
    u = collections.defaultdict(list)
    for mt in meta:
        if mt["u"] is not None and mt["in_stick"]: u[mt["machine"]].append(abs(mt["u"]))
    return by_m, u


def prompt_stats():
    """레이블 500장의 (장비, 막대 수) 장 수 — 대회 원본(data/dataset/)의 레이블을 읽는다. 순서는 이전 instructions() 와 같다(난수 순서를 같게)"""
    from splits import all_paths, test_stems, machine_of
    P = all_paths(); cnt = collections.Counter()
    for s in test_stems():
        if s in P: cnt[(machine_of(P[s]), 3 if s[4:12] <= "20200715" else 1)] += 1
    return [[m, k, n] for (m, k), n in cnt.items()]


def instructions(n, rng, stats):
    """레이블 500장의 장비·막대 수 비율대로 지시어 n개를 만든다(막대마다 이물 하나). stats = prompt_stats() 또는 material_500/prompt_stats.json"""
    keys = [(m, k) for m, k, _ in stats]; pr = np.array([c for _, _, c in stats], float); pr /= pr.sum()
    out = []
    for i in rng.choice(len(keys), n, p=pr):
        m, k = keys[i]; out.append(f"{m}, 스틱 {k}개, 이물 {'하나' if k == 1 else f'{k}개'}")
    return out


def insert_patch(img, stick, patch, u_abs, rng):
    """막대 길이 방향 위치 |u|(0=가운데, 1=끝)에 패치를 곱해 넣는다 → 중심 좌표 또는 None"""
    ys, xs = np.where(stick)
    if len(ys) < 50: return None
    (scx, scy), (w, h), ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
    L, Wd = max(w, h), min(w, h); a = np.deg2rad(ang if w >= h else ang + 90)
    side = rng.choice([-1, 1]); v = rng.uniform(-0.15, 0.15) * Wd
    cx = scx + side * u_abs * L / 2 * np.cos(a) - v * np.sin(a); cy = scy + side * u_abs * L / 2 * np.sin(a) + v * np.cos(a)
    if rng.random() < 0.5: patch = patch[:, ::-1]
    if rng.random() < 0.5: patch = patch[::-1, :]
    r = patch.shape[0] // 2; icx, icy = int(round(cx)), int(round(cy)); H, W = img.shape
    if icy - r < 0 or icx - r < 0 or icy + r + 1 > H or icx + r + 1 > W or not stick[icy, icx]: return None
    img[icy - r:icy + r + 1, icx - r:icx + r + 1] *= patch
    return icx, icy


def make_one(text, by_m, u_by_m, rng, levels=None, bg="diffusion", shapes=None):
    """bg: diffusion(제안 방법) | procedural(비교용: 같은 배치도에 밝기 규칙으로 봉지·막대를 그림)
    shapes: None 또는 dict(frac=새 형상 비율, sigma=흐림 σ). 이물마다 frac 확률로 점 패치 대신 새 형상을 넣는다"""
    spec = parse_prompt(text); spec.background = "diffusion"
    prof = PROFILES[spec.machine]; lv = None
    if levels:   # 500장 중 같은 장비 사진 한 장을 골라 그 사진의 밝기와 봉지 기울기 크기를 따른다(기울어지는 방향은 무작위)
        lv = levels[spec.machine][rng.integers(len(levels[spec.machine]))]
        if spec.rotation_deg is None: spec.rotation_deg = float(abs(lv["angle"]) * rng.choice([-1, 1]))
    if bg == "procedural":   # Diffusion 방식과 난수 순서를 맞춘다(Diffusion 잡음 시드로 쓰는 한 번을 똑같이 뽑음) → 배치도·이물 패치·위치·레이블이 같다
        img, bag, stick_mask, sticks = procedural_background(spec, prof, rng); rng.integers(1 << 31); rng.normal(0, 1, bag.shape)
    else:
        img, bag, stick_mask, sticks = sds.diffusion_background(spec, prof, rng)
    img = img.astype(np.float32)
    if lv: img = match_tone(img, bag, stick_mask, lv)
    base = img.copy(); labels = []; info = []
    for k, d in enumerate(spec.defects):
        if not sticks: break
        sm = sticks[k % len(sticks)]
        for _ in range(20):
            p, mt = by_m[spec.machine][rng.integers(len(by_m[spec.machine]))]; w, h, src = mt["w"], mt["h"], mt["stem"]
            if shapes and rng.random() < shapes["frac"]:   # 새 형상: 어두운 정도는 방금 고른 점 패치의 가장 어두운 투과율
                kind = KINDS[rng.integers(len(KINDS))]; p, w, h = draw_shape(kind, rng, shapes["sigma"], float(p.min())); src = f"shape:{kind}"
            c = insert_patch(img, sm, p, float(rng.choice(u_by_m[spec.machine])), rng)
            if c: labels.append((c[0], c[1], w, h)); info.append(dict(src=src, depth=mt["depth"])); break
    return np.clip(img, 0, 255).astype(np.uint8), labels, info, spec, np.clip(base, 0, 255).astype(np.uint8), (bag, stick_mask)


def level_error(by_m, u_by_m, prof500, t, rng, n=12):
    """빈 화면 출발점(t)별로 그린 봉지·막대 밝기가 500장의 값과 얼마나 다른지(장비 평균, 회색값 차이)"""
    sds._STATE["t_start"] = t; err = []
    for m in MACHINES:
        for _ in range(n):
            spec = parse_prompt(f"{m}, 스틱 3개"); spec.background = "diffusion"
            img, bag, stick_mask, _ = sds.diffusion_background(spec, PROFILES[m], rng)
            pack = bag & ~ndimage.binary_dilation(stick_mask, iterations=3); st = ndimage.binary_erosion(stick_mask, iterations=2)
            err.append(abs(np.median(img[pack]) - prof500[m]["pack"]) + abs(np.median(img[st]) - prof500[m]["stick"]))
    return float(np.mean(err))
