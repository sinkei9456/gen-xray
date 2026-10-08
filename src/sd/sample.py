"""학습한 Diffusion 모델로 봉지·막대를 그리는 함수(load, diffusion_background). develop/make_synth.py, release/generate.py, synth/compose.py 가 쓴다.
지금 결과는 init="blank": 장비 바탕 밝기만 채운 빈 화면에서 출발해, 배치도(봉지·막대 위치)를 보며 Diffusion 모델이 봉지와 막대를 그린다
(t_start = 999, models/diffusion/sample.json). 아래 main()은 이전 방법(synth.generate)과 함께 쓰던 실험용으로 지금 결과에는 쓰지 않는다.
"""
from __future__ import annotations
import argparse, sys, numpy as np, cv2, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synth.render import BACKENDS, procedural_background, MACHINES
from synth.layout import Spec
from sd.data import square_pad, layout_map
from sd.model import make_cond, sample
from diffusers import UNet2DModel

_STATE = {}


def load(ckpt: str, device="cuda", steps: int = 50, t_start: int = 700, init: str = "procedural"):
    """init: blank(아무것도 없는 빈 바탕 화면에서 출발, 지금 결과) | procedural(밝기 규칙 그림에서 출발, 이전 실험용)"""
    _STATE["unet"] = UNet2DModel.from_pretrained(ckpt).to(device).eval(); _STATE["steps"] = steps; _STATE["dev"] = device; _STATE["t_start"] = t_start; _STATE["init"] = init
    BACKENDS["diffusion"] = diffusion_background
    return _STATE["unet"]


def diffusion_background(spec: Spec, prof: dict, rng: np.random.Generator):
    """procedural_background 로 배치도(봉지·막대 위치)만 만들고, 봉지와 막대 그림은 Diffusion 모델이 그린다."""
    proc, bag, stick_mask, sticks = procedural_background(spec, prof, rng)
    H, W = bag.shape; lm, (y0, x0, _, _, S) = square_pad(layout_map(bag, stick_mask), 0)
    size = _STATE["unet"].config.sample_size
    lm_s = cv2.resize(lm, (size, size), interpolation=cv2.INTER_NEAREST)
    # 출발 그림: blank(지금 결과)면 장비 바탕 밝기만 채운 빈 화면, procedural(이전 실험용)이면 밝기 규칙으로 그린 그림
    if _STATE.get("init") == "blank": proc = np.full_like(proc, float(prof["bg"]))   # 빈 바탕 화면: 봉지·막대는 Diffusion 모델이 배치도만 보고 그림
    pi, _ = square_pad(proc.astype(np.float32), float(prof["bg"]))
    init = torch.from_numpy(cv2.resize(pi, (size, size), interpolation=cv2.INTER_AREA)).float()[None, None] / 127.5 - 1
    m_idx = MACHINES.index(spec.machine) if spec.machine in MACHINES else int(rng.integers(3))
    g = torch.Generator(device=_STATE["dev"]).manual_seed(int(rng.integers(1 << 31)))
    im = sample(_STATE["unet"], make_cond(lm_s[None], [m_idx]), steps=_STATE["steps"], generator=g, device=_STATE["dev"], init=init, t_start=_STATE["t_start"])[0]
    im = cv2.resize(im, (S, S), interpolation=cv2.INTER_CUBIC).astype(np.float32)[y0:y0 + H, x0:x0 + W]
    # 256×256 그림을 늘리면서 흐려진 만큼 장비별 크기의 잡음을 더한다
    nz = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.6)
    lvl = np.where(stick_mask, prof["noise_stick"], np.where(bag, prof["noise_pack"], prof["noise_bg"])) * 0.6
    im = np.clip(im + nz * lvl / 0.62, 0, 255)
    return im, bag, stick_mask, sticks


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("prompt", nargs="+"); ap.add_argument("--ckpt", required=True); ap.add_argument("-n", type=int, default=8)
    ap.add_argument("-o", "--out", default="../data/synthetic_sd"); ap.add_argument("--steps", type=int, default=50); ap.add_argument("--t-start", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--all-classes", action="store_true")
    a = ap.parse_args(); load(a.ckpt, steps=a.steps, t_start=a.t_start)
    from synth import generate, parse_prompt
    specs = []
    for p in a.prompt:
        s = parse_prompt(p); s.background = "diffusion"; specs.append(s)
    df = generate(specs, n=a.n, out_dir=a.out, seed=a.seed, label_classes=(0, 1, 2) if a.all_classes else (0,), prefix="sd")
    print(df.groupby(["product", "placement"]).size().to_string()); print("→", a.out)


if __name__ == "__main__":
    main()
