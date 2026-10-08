"""대회 이미지 도구와 배치도 그리기.

지금 결과에 쓰는 것: load_raw(대회 BMP 읽기와 검사 장비 색 표시), marker_boxes, product_masks(봉지·막대 영역),
procedural_background(배치도의 봉지·막대 위치 만들기. 봉지·막대 그림은 Diffusion 모델이 그린다, sd/sample.py),
PROFILES(장비별 설정값. 실행할 때 synth/compose.load_profiles 가 레이블 500장에서 잰 값으로 바꾼다).
Renderer·RealBank·절차적 이물(thickness_map, defect_delta, MU)은 이전 방법으로 지금 결과에는 쓰지 않는다.
"""
from __future__ import annotations
import glob, os, json
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np, cv2
from scipy import ndimage
from PIL import Image
from .layout import Spec, Defect, MACHINES

# ------------------------------------------------------------------ 장비별 설정값
PROFILES = {  # 기본값(이전 방법에서 잰 값). 지금 결과는 실행할 때 compose.load_profiles()가 data/material_500/profiles_500.json(레이블 500장)으로 바꾼다
    "1호기": dict(W=352, H=332, bg=210.0, pack=144.0, stick=113.0, noise_bg=0.3, noise_pack=2.94, noise_stick=5.18, bag=(198, 150), stick_w=17.1, stick_len=0.62),
    "2호기": dict(W=316, H=332, bg=207.0, pack=107.0, stick=77.0, noise_bg=0.91, noise_pack=2.47, noise_stick=2.47, bag=(196, 151), stick_w=15.96, stick_len=0.6),
    "3호기": dict(W=576, H=444, bg=210.0, pack=125.98, stick=92.0, noise_bg=0.84, noise_pack=3.77, noise_stick=4.34, bag=(270, 206), stick_w=20.04, stick_len=0.6),
}
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
RAW_ROOT = DATA_ROOT / "dataset" / "test1" / "yolov3" / "X선이물검출기(06.23_09.22)"
LABEL_DIR = DATA_ROOT / "dataset" / "라벨링 6종 세트" / "labels"   # 사람 레이블 500개(학습 재료로 쓰고, 평가에는 쓰지 않는다)
CLS_DEFECT, CLS_STICK, CLS_BAG = 0, 1, 2
BACKENDS = {}   # 이름 -> fn(spec, prof, rng) -> (img, bag_mask, stick_mask, sticks). sd.sample 이 "diffusion" 을 등록


# ------------------------------------------------------------------ 공통 유틸
def load_raw(path: str) -> Tuple[np.ndarray, np.ndarray]:
    im = Image.open(path); pal = np.array(im.getpalette()).reshape(-1, 3); idx = np.array(im)
    gray = np.clip(pal[idx].mean(-1), 0, 255).astype(np.float32)
    colored = np.where((np.abs(pal[:, 0] - pal[:, 1]) + np.abs(pal[:, 1] - pal[:, 2])) > 0)[0]
    return gray, np.isin(idx, colored)


def marker_boxes(mask: np.ndarray, min_size: int = 5):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3))); out = []
    for sl in ndimage.find_objects(lab):
        if sl is not None and (sl[0].stop - sl[0].start) >= min_size and (sl[1].stop - sl[1].start) >= min_size:
            out.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
    return out


def product_masks(img: np.ndarray):
    """회색 이미지 → (봉지 마스크, 막대 마스크) 또는 None. 실제 이미지에서 봉지·막대 영역을 찾을 때 쓴다."""
    H, W = img.shape; bg = np.median(img[:, W // 2:])
    prod = img < bg - 15; prod[:, :4] = False; prod = ndimage.binary_opening(prod, iterations=2)
    if prod.sum() < 500: return None
    pv = img[prod]; pack = np.median(pv); low = np.percentile(pv, 2)
    thr = pack - 0.55 * (pack - low) if pack - low > 15 else pack - 25     # 이미지별 적응 임계값 (포장재↔스틱 중간)
    stick = ndimage.binary_opening((img < thr) & prod, iterations=2)
    stick = ndimage.binary_closing(stick, iterations=2) & prod
    lab, n = ndimage.label(stick); sizes = ndimage.sum(stick, lab, range(1, n + 1))
    stick = np.isin(lab, [i + 1 for i, a in enumerate(sizes) if a >= 300])
    return prod, stick


def zone_masks(prod: np.ndarray, stick: np.ndarray) -> Dict[str, np.ndarray]:
    return dict(
        stick_in=ndimage.binary_erosion(stick, iterations=4),
        stick_edge=ndimage.binary_dilation(stick, iterations=2) & ~ndimage.binary_erosion(stick, iterations=3),
        packaging=ndimage.binary_erosion(prod, iterations=8) & ~ndimage.binary_dilation(stick, iterations=6),
        prod_edge=prod & ~ndimage.binary_erosion(prod, iterations=7) & ndimage.binary_erosion(prod, iterations=2),
    )


def bbox_of(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(ys) == 0: return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


# ------------------------------------------------------------------ 절차적 이물(이전 방법, 지금 결과에는 쓰지 않음)
def thickness_map(shape: str, size: float, rng: np.random.Generator, ss: int = 4) -> np.ndarray:
    """두께맵 t(x,y) [단위: 기준 구 지름], ss배 슈퍼샘플링 캔버스."""
    R = 1.8 * size                     # 기준 이물 반지름 ≈ 1.8px (지름 3~4px)
    n = int(np.ceil(R * 2 + 6)) | 1; c = n // 2
    yy, xx = np.mgrid[:n * ss, :n * ss] / ss - c
    if shape == "ball":
        r2 = np.clip(R ** 2 - (xx ** 2 + yy ** 2), 0, None); t = 2 * np.sqrt(r2) / (2 * 1.8)
    elif shape == "wire":
        L = R * rng.uniform(3, 6); w = max(0.5, R * 0.45); a = rng.uniform(0, np.pi)
        u = xx * np.cos(a) + yy * np.sin(a); v = -xx * np.sin(a) + yy * np.cos(a)
        n2 = int(np.ceil(L + 6)) | 1
        yy, xx = np.mgrid[:n2 * ss, :n2 * ss] / ss - n2 // 2
        u = xx * np.cos(a) + yy * np.sin(a); v = -xx * np.sin(a) + yy * np.cos(a)
        t = (np.abs(u) < L / 2) * np.clip(w ** 2 - v ** 2, 0, None) ** 0.5 * 2 / (2 * 1.8)
    else:  # flake: 불규칙 다각형, 얇고 넓음
        n2 = int(np.ceil(R * 3 + 6)) | 1
        yy, xx = np.mgrid[:n2 * ss, :n2 * ss] / ss - n2 // 2
        ang = np.arctan2(yy, xx); rad = np.hypot(xx, yy)
        k = rng.integers(3, 7); ph = rng.uniform(0, np.pi)
        rlim = R * 1.5 * (1 + 0.35 * np.cos(k * ang + ph) + 0.2 * np.cos((k + 2) * ang))
        t = (rad < rlim) * rng.uniform(0.25, 0.6)
    t = cv2.resize(t.astype(np.float32), (t.shape[1] // ss, t.shape[0] // ss), interpolation=cv2.INTER_AREA)
    return t


MU = 0.95  # 기준 구(두께 1.0) 감쇠 exp(-MU). 이전 방법용(지금 합성은 레이블 500장에서 떼어 낸 이물 패치의 투과율과 새 형상을 쓴다)


def defect_delta(local_level: float, shape: str, size: float, contrast: float, rng: np.random.Generator) -> np.ndarray:
    t = thickness_map(shape, size, rng)
    delta = local_level * (np.exp(-MU * contrast * t) - 1.0)          # Beer-Lambert: I = I0·exp(-μt)
    return cv2.GaussianBlur(delta, (0, 0), 0.7).astype(np.float32)     # 시스템 PSF


# ------------------------------------------------------------------ 절차적 봉지·막대(procedural_background: 배치도의 봉지·막대 위치를 만드는 데 지금도 쓴다)
def _rounded_rect(shape, cx, cy, w, h, r, ang_deg):
    H, W = shape; yy, xx = np.mgrid[:H, :W].astype(np.float32)
    a = np.deg2rad(ang_deg); u = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a); v = -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
    qx = np.abs(u) - (w / 2 - r); qy = np.abs(v) - (h / 2 - r)
    d = np.hypot(np.clip(qx, 0, None), np.clip(qy, 0, None)) + np.minimum(np.maximum(qx, qy), 0) - r
    return d  # signed distance (<0 내부)


def _pillow_sdf(shape, cx, cy, w, h, ang_deg, p=3.2, wave=0.0, rng=None):
    """베개(pillow)형 봉지 실루엣: 초타원(superellipse) + 가장자리 물결. 반환: 정규화 거리(<1 내부), 회전 좌표 (u,v)"""
    H, W = shape; yy, xx = np.mgrid[:H, :W].astype(np.float32)
    a = np.deg2rad(ang_deg); u = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a); v = -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
    hw, hh = w / 2, h / 2
    if wave and rng is not None:
        ph = rng.uniform(0, 2 * np.pi, 3)
        hh = hh * (1 + wave * (0.6 * np.sin(2 * np.pi * u / w * 1.0 + ph[0]) + 0.4 * np.sin(2 * np.pi * u / w * 2.3 + ph[1])))
        hw = hw * (1 + 0.5 * wave * np.sin(2 * np.pi * v / h * 1.7 + ph[2]))
    r = (np.abs(u) / hw) ** p + (np.abs(v) / hh) ** p
    return r, u, v, hw, hh


def procedural_background(spec: Spec, prof: dict, rng: np.random.Generator):
    W, H = prof["W"], prof["H"]; img = np.full((H, W), float(prof["bg"]), np.float32)
    ang = spec.rotation_deg if spec.rotation_deg is not None else rng.uniform(-25, 25)
    bw, bh = prof["bag"]; bw *= rng.uniform(0.96, 1.04); bh *= rng.uniform(0.96, 1.04)
    cx, cy = W * rng.uniform(*prof.get("cx", (0.60, 0.70))), H * rng.uniform(*prof.get("cy", (0.42, 0.58)))   # 봉지 중심 범위(설정값에 있으면 그 값)
    half = 0.5 * np.hypot(bw, bh) * 0.92 + 10
    cx = float(np.clip(cx, half, W - half)); cy = float(np.clip(cy, half * 0.85, H - half * 0.85))
    if spec.placement == "left_cut": cx = rng.uniform(-0.1, 0.25) * bw
    elif spec.placement == "right_cut": cx = W - rng.uniform(-0.1, 0.25) * bw
    elif spec.placement == "top_cut": cy = rng.uniform(-0.1, 0.25) * bh
    elif spec.placement == "off_frame" or spec.product == "empty": cx = -2 * bw
    r, u, v, hw, hh = _pillow_sdf((H, W), cx, cy, bw, bh, ang, p=rng.uniform(3.8, 5.0), wave=0.025, rng=rng)
    bag = r < 1.0
    bg = prof["bg"]
    if spec.product == "dense_bag":
        edge_lvl, center_lvl = prof["stick"] + 12, prof["stick"] - 4
    else:
        edge_lvl, center_lvl = prof["pack"] + 10, prof["pack"] - 14
    # 베개 두께 셰이딩: 가장자리(얇음)는 밝고 중앙(두꺼움)은 어두움. 실측 1호기: 림 ~145 → 중앙 ~120
    thick = np.clip(1 - r, 0, 1) ** 0.45
    level = edge_lvl - (edge_lvl - center_lvl) * thick
    # 주름/두께 요동: 저주파 노이즈
    wr = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 7); wr *= 4.0 / max(wr.std(), 1e-6)
    img[bag] = (level + wr)[bag]
    # 양끝 거싯 플랩: 봉지 좌우 끝에서 바깥으로 6~12px, 밝은 사다리꼴
    fl = rng.uniform(6, 12)
    flap = (~bag) & (np.abs(u) > hw * 0.9) & (np.abs(u) < hw + fl) & (np.abs(v) < hh * (0.75 - 0.35 * np.clip((np.abs(u) - hw) / fl, 0, 1)))
    img[flap] = bg - rng.uniform(18, 30) + wr[flap] * 0.5
    sticks = []
    n = spec.n_sticks
    if n:
        sl = bw * prof["stick_len"] * rng.uniform(0.95, 1.05); sw = prof["stick_w"] * rng.uniform(0.9, 1.1)
        gap = sw * 0.45
        offs = [0.0] if n == 1 else [-(sw + gap), 0.0, (sw + gap)]
        a = np.deg2rad(ang)
        for k, o in enumerate(offs):
            scx, scy = cx - o * np.sin(a) + rng.uniform(-3, 3), cy + o * np.cos(a) + rng.uniform(-2, 2)
            ds = _rounded_rect((H, W), scx, scy, sl, sw, 3, ang)
            m = ds < 0
            # 한쪽 끝 테이퍼(뾰족한 스틱 끝): 끝 8px 범위에서 폭이 선형으로 줄어듦
            us = (np.mgrid[:H, :W][1] - scx) * np.cos(a) + (np.mgrid[:H, :W][0] - scy) * np.sin(a)
            vs = -(np.mgrid[:H, :W][1] - scx) * np.sin(a) + (np.mgrid[:H, :W][0] - scy) * np.cos(a)
            tip = rng.choice([-1, 1]); tl = 8.0
            taper = (tip * us > sl / 2 - tl) & (np.abs(vs) > (sw / 2) * (1 - (tip * us - (sl / 2 - tl)) / tl))
            if prof.get("taper", True): m &= ~taper
            img[m] = prof["stick"] + wr[m] * 0.6; sticks.append(m)
    img = cv2.GaussianBlur(img, (0, 0), 1.1)
    stick_mask = np.any(sticks, axis=0) if sticks else np.zeros((H, W), bool)
    nz = rng.normal(0, 1, (H, W)).astype(np.float32)
    lvl = np.where(stick_mask, prof["noise_stick"], np.where(bag, prof["noise_pack"], prof["noise_bg"]))
    img += cv2.GaussianBlur(nz, (0, 0), 0.6) * lvl / 0.62
    streak = cv2.GaussianBlur(rng.normal(0, 1, (H, 1)).astype(np.float32), (0, 0), 2)[:, 0] * rng.uniform(0.4, 1.2)
    img += streak[:, None]
    return np.clip(img, 0, 255), bag, stick_mask, sticks


# ------------------------------------------------------------------ 실제 봉지 사진 모음(RealBank, 이전 방법, 지금 결과에는 쓰지 않음)
class RealBank:
    def __init__(self, raw_root: Path = RAW_ROOT, rng: Optional[np.random.Generator] = None):
        self.paths: Dict[str, str] = {}
        for p in sorted(glob.glob(str(raw_root / "*" / "*" / "*.bmp"))):
            self.paths.setdefault(os.path.splitext(os.path.basename(p))[0], p)
        self.rng = rng or np.random.default_rng(0)
        # 환경변수 XRAY_BANK_EXCLUDE_LABELED=1 이면 사람 레이블이 있는 500장을 실제 봉지 사진 모음에서 뺀다(이전 방법용)
        self.exclude = set()
        if os.environ.get("XRAY_BANK_EXCLUDE_LABELED", "0") == "1" or os.environ.get("XRAY_BANK_STRICT", "0") == "1":
            self.exclude = {os.path.splitext(os.path.basename(t))[0] for t in glob.glob(str(LABEL_DIR / "*.txt"))}
        if os.environ.get("XRAY_BANK_STRICT", "0") == "1":
            # 이전 방법의 나누기(splits.bank_stems)를 쓰는 경우
            import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from splits import bank_stems
            keep = set(bank_stems(self.paths)); self.exclude |= {s for s in self.paths if s not in keep}
        self.index = {}   # (machine, product) -> stems
        for s, p in self.paths.items():
            if s in self.exclude: continue
            m = p.split("/")[-3][:3]; era = "stick3" if s[4:12] <= "20200715" else "stick1"
            if s[4:12] == "20200727" and m == "1호기": continue   # 7/27 이벤트 제외
            self.index.setdefault((m, era), []).append(s)
        self._cache: Dict[str, tuple] = {}

    def clean(self, stem: str, pad: int = 4):
        if stem in self._cache: return self._cache[stem]
        gray, mask = load_raw(self.paths[stem]); boxes = marker_boxes(mask)
        m = np.zeros_like(mask)
        for x1, y1, x2, y2 in boxes: m[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = True
        m |= ndimage.binary_dilation(mask, iterations=1)
        ip = cv2.inpaint(gray.astype(np.uint8), m.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA).astype(np.float32)
        hf = ip - cv2.GaussianBlur(ip, (0, 0), 1.5)
        for x1, y1, x2, y2 in boxes:
            ring = np.zeros_like(m); ring[max(0, y1 - 14):y2 + 14, max(0, x1 - 14):x2 + 14] = True; ring &= ~m
            sd = hf[ring].std() if ring.any() else 2.0
            reg = np.zeros_like(m); reg[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = True
            nz = cv2.GaussianBlur(self.rng.normal(0, 1, ip.shape).astype(np.float32), (0, 0), 0.7); nz *= sd / max(nz.std(), 1e-6)
            ip[reg] += nz[reg]
        excl = ndimage.binary_dilation(m, iterations=14)
        self._cache[stem] = (np.clip(ip, 0, 255), excl); return self._cache[stem]

    def sample(self, machine: str, product: str):
        key = (machine, product if product in ("stick1", "stick3") else "stick1")
        stem = self.rng.choice(self.index[key]); img, excl = self.clean(stem); return stem, img.copy(), excl


def shift_for_placement(img: np.ndarray, prod: np.ndarray, placement: str, bg: float, rng):
    """실제 봉지 사진을 밀어서 봉지가 화면 끝에 잘리거나 화면 밖에 있는 이미지를 만든다(이전 방법용)"""
    if placement == "center": return img, 0, 0
    H, W = img.shape; b = bbox_of(prod)
    if b is None: return img, 0, 0
    x0, y0, x1, y1 = b; bw, bh = x1 - x0, y1 - y0
    dx = dy = 0
    if placement == "left_cut": dx = -x0 - int(rng.uniform(0.3, 0.8) * bw)
    elif placement == "right_cut": dx = (W - x0) - int(rng.uniform(0.2, 0.7) * bw)
    elif placement == "top_cut": dy = -y0 - int(rng.uniform(0.3, 0.8) * bh)
    elif placement == "off_frame": dx = -x1 - 5
    out = np.full_like(img, bg); ys, xs = np.where(np.ones_like(img, bool))
    src = img
    ys2, xs2 = ys + dy, xs + dx; ok = (ys2 >= 0) & (ys2 < H) & (xs2 >= 0) & (xs2 < W)
    out[ys2[ok], xs2[ok]] = src[ys[ok], xs[ok]]
    return out, dx, dy


# ------------------------------------------------------------------ 렌더러(이전 방법, 지금 결과에는 쓰지 않음)
class Renderer:
    def __init__(self, seed: int = 0, real_bank: Optional[RealBank] = None, dot_bank_path: Optional[Path] = None):
        self.rng = np.random.default_rng(seed)
        self.bank = real_bank
        self.dots = None
        p = dot_bank_path or (DATA_ROOT / "dot_bank.npz")
        if p.exists():
            z = np.load(p, allow_pickle=True); self.dots = list(z["delta"])

    def _bank(self):
        if self.bank is None: self.bank = RealBank(rng=self.rng)
        return self.bank

    def render(self, spec: Spec, use_real_dots: bool = False, label_sticks: bool = True):
        rng = self.rng
        machine = spec.machine if spec.machine in MACHINES else rng.choice(MACHINES)
        prof = PROFILES[machine]
        meta = dict(machine=machine, product=spec.product, placement=spec.placement, background=spec.background, note=spec.note)
        if spec.background == "real" and spec.product in ("stick1", "stick3"):
            stem, img, excl = self._bank().sample(machine, spec.product)
            pm = product_masks(img)
            if pm is None: return self.render(spec, use_real_dots, label_sticks)
            prod, stick = pm
            img, dx, dy = shift_for_placement(img, prod, spec.placement, prof["bg"], rng)
            if dx or dy:
                prod = np.roll(np.roll(prod, dy, 0), dx, 1); stick = np.roll(np.roll(stick, dy, 0), dx, 1); excl = np.roll(np.roll(excl, dy, 0), dx, 1)
                # roll wrap 방지
                if dx < 0: prod[:, dx:] = stick[:, dx:] = False
                if dx > 0: prod[:, :dx] = stick[:, :dx] = False
                if dy < 0: prod[dy:, :] = stick[dy:, :] = False
                if dy > 0: prod[:dy, :] = stick[:dy, :] = False
            lab, n = ndimage.label(stick); sticks = [lab == i for i in range(1, n + 1) if (lab == i).sum() > 150]
            sticks.sort(key=lambda m: np.where(m)[0].mean())
            meta["source_stem"] = stem
        elif spec.background in BACKENDS and spec.product != "empty":
            img, prod, stick, sticks = BACKENDS[spec.background](spec, prof, rng); excl = np.zeros_like(prod)
        else:
            img, prod, stick, sticks = procedural_background(spec, prof, rng); excl = np.zeros_like(prod)
        H, W = img.shape
        excl = excl.copy(); excl[:10, :] = excl[-10:, :] = excl[:, :10] = excl[:, -10:] = True
        zones = zone_masks(prod, stick)
        labels: List[dict] = []
        for d in spec.defects:
            zn = d.zone if d.zone in zones else "stick_in"
            cand_mask = zones[zn] & ~excl
            if zn in ("stick_in", "stick_edge") and sticks:
                idx = d.stick_index if d.stick_index is not None and d.stick_index < len(sticks) else rng.integers(len(sticks))
                sm = sticks[idx]
                cand_mask &= (ndimage.binary_erosion(sm, iterations=4) if zn == "stick_in" else ndimage.binary_dilation(sm, iterations=2))
                if getattr(d, "along", "random") in ("end", "center"):
                    ys_, xs_ = np.where(sm); pts = np.stack([xs_, ys_], 1).astype(np.float32)
                    (scx, scy), (w_, h_), ang_ = cv2.minAreaRect(pts); L = max(w_, h_)
                    a_ = np.deg2rad(ang_ if w_ >= h_ else ang_ + 90)
                    yy_, xx_ = np.mgrid[:H, :W]; u = ((xx_ - scx) * np.cos(a_) + (yy_ - scy) * np.sin(a_)) / (L / 2)
                    cand_mask &= (np.abs(u) > 0.82) if d.along == "end" else (np.abs(u) < 0.3)
            cand = np.argwhere(cand_mask)
            if len(cand) == 0:
                cand = np.argwhere(zones["packaging"] & ~excl)
                if len(cand) == 0: continue
            cy, cx = map(int, cand[rng.integers(len(cand))])
            local = float(np.median(img[max(0, cy - 4):cy + 5, max(0, cx - 4):cx + 5]))
            if use_real_dots and self.dots and d.shape == "ball":
                delta = self.dots[rng.integers(len(self.dots))] * d.contrast
                if d.size != 1.0:
                    n = max(3, int(round(delta.shape[0] * d.size)) | 1); delta = cv2.resize(delta, (n, n), interpolation=cv2.INTER_AREA if d.size < 1 else cv2.INTER_LINEAR)
            else:
                delta = defect_delta(local, d.shape, d.size, d.contrast, rng)
            r = delta.shape[0] // 2; y0, y1, x0, x1 = cy - r, cy + r + 1, cx - r, cx + r + 1
            if y0 < 0 or x0 < 0 or y1 > H or x1 > W: continue
            img[y0:y1, x0:x1] = np.clip(img[y0:y1, x0:x1] + delta, 0, 255)
            ys, xs = np.where(delta < -6)
            if len(ys) == 0: ys, xs = np.array([r]), np.array([r])
            bx0, by0, bx1, by1 = x0 + xs.min() - 2, y0 + ys.min() - 2, x0 + xs.max() + 3, y0 + ys.max() + 3
            labels.append(dict(cls=CLS_DEFECT, box=(int(bx0), int(by0), int(bx1), int(by1)), zone=zn, size=d.size, contrast=d.contrast, shape=d.shape,
                               depth=float(-delta.min()), local_level=local))
            excl[max(0, by0 - 10):by1 + 10, max(0, bx0 - 10):bx1 + 10] = True
        if label_sticks:
            for sm in sticks:
                b = bbox_of(sm)
                if b: labels.append(dict(cls=CLS_STICK, box=b))
            b = bbox_of(prod)
            if b: labels.append(dict(cls=CLS_BAG, box=b))
        meta["n_sticks_visible"] = len(sticks); meta["bag_visible"] = bool(prod.any())
        meta["bag_touches_border"] = bool(prod[:, :6].any() or prod[:, -6:].any() or prod[:6, :].any() or prod[-6:, :].any())
        meta["inspectable"] = bool(meta["bag_visible"] and not meta["bag_touches_border"] and len(sticks) in (1, 3))
        return np.clip(img, 0, 255).astype(np.uint8), labels, meta


def yolo_lines(labels: List[dict], W: int, H: int, classes=(CLS_DEFECT,)) -> List[str]:
    out = []
    for L in labels:
        if L["cls"] not in classes: continue
        x0, y0, x1, y1 = L["box"]
        out.append(f"{L['cls']} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return out
