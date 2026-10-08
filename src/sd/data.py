"""Diffusion 모델 입력용 함수: square_pad(긴 변 길이의 정사각형으로 채우기), layout_map(배치도: 봉지 128, 막대 255).
build_cache()·CACHE 는 이전 방법(대회 이미지 전체로 캐시)으로 지금 결과에는 쓰지 않는다. 지금 Diffusion 학습 입력은
develop/prepare_material.py 가 레이블 500장으로 만든다(data/material_500/diffusion_cache_500.npz)."""
from __future__ import annotations
import os, sys, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synth.render import RealBank, product_masks, DATA_ROOT, MACHINES

CACHE = Path(os.environ.get("XRAY_SD_CACHE", str(DATA_ROOT / "diffusion_cache_256.npz")))   # 이전 방법용
BG = 210.0


def square_pad(img: np.ndarray, fill: float) -> tuple[np.ndarray, tuple]:
    H, W = img.shape[:2]; S = max(H, W); out = np.full((S, S) + img.shape[2:], fill, img.dtype)
    y0, x0 = (S - H) // 2, (S - W) // 2; out[y0:y0 + H, x0:x0 + W] = img
    return out, (y0, x0, H, W, S)


def layout_map(prod: np.ndarray, stick: np.ndarray) -> np.ndarray:
    lm = np.zeros(prod.shape, np.uint8); lm[prod] = 128; lm[stick] = 255; return lm


def build_cache(size: int = 256, path: Path = CACHE, limit: int | None = None) -> Path:
    bank = RealBank(); stems = [s for lst in bank.index.values() for s in lst]
    if limit: stems = stems[:limit]
    imgs, lays, machines, names = [], [], [], []
    for i, stem in enumerate(stems):
        img, _ = bank.clean(stem); pm = product_masks(img)
        if pm is None: continue
        prod, stick = pm
        sq, _ = square_pad(img, BG); lm, _ = square_pad(layout_map(prod, stick), 0)
        imgs.append(cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA).astype(np.uint8))
        lays.append(cv2.resize(lm, (size, size), interpolation=cv2.INTER_NEAREST))
        machines.append(MACHINES.index(bank.paths[stem].split("/")[-3][:3])); names.append(stem)
        if i % 200 == 0: print(f"  cache {i}/{len(stems)}")
    np.savez_compressed(path, img=np.stack(imgs), layout=np.stack(lays), machine=np.array(machines), stem=np.array(names))
    print("cache saved:", path, len(imgs)); return path


if __name__ == "__main__":
    build_cache()
