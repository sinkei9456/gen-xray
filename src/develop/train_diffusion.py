"""Diffusion 모델 학습(배치도·장비 번호를 조건으로 받는 U-Net, 256×256). 학습 재료: data/material_500/diffusion_cache_500.npz(prepare_material.py 결과,
레이블 500장의 봉지·막대 이미지와 배치도). 시드 0, GPU 결정적 계산 모드. 1000 스텝마다 EMA 가중치 저장 + 샘플 그림.
사용: python develop/train_diffusion.py → runs/diffusion/ (ema/, last/), 끝나면 ema 를 models/diffusion 으로 복사
     짧은 실행 확인: python develop/train_diffusion.py --out <임시 폴더> --steps 20 --save-every 20 --no-export
"""
from __future__ import annotations
import argparse, os, sys, time, copy, json, random
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (공통 함수)
from paths import RUNS, MATERIAL, DIFFUSION
from sd.model import build_unet, make_cond, train_scheduler, sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUNS / "diffusion")); ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cache", default=str(MATERIAL / "diffusion_cache_500.npz")); ap.add_argument("--no-export", action="store_true", help="models/diffusion 으로 복사하지 않음")
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--size", type=int, default=256); ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--resume", default=None); ap.add_argument("--ema", type=float, default=0.999); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # 재현: 시드 고정(초기 가중치·잡음·단계 선택·배치 순서) + GPU 계산 순서 고정
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True)
    z = np.load(a.cache); imgs = torch.from_numpy(z["img"]).float() / 127.5 - 1.0; lays = z["layout"]; mach = z["machine"]
    N = len(imgs); print("dataset", N, "images"); dev = "cuda"
    unet = build_unet(a.size).to(dev); ema = copy.deepcopy(unet).eval()
    if a.resume:
        from diffusers import UNet2DModel
        unet = UNet2DModel.from_pretrained(a.resume).to(dev); ema = copy.deepcopy(unet).eval()
    opt = torch.optim.AdamW(unet.parameters(), lr=a.lr, weight_decay=0.01); sch = train_scheduler()
    print("params M:", sum(p.numel() for p in unet.parameters()) / 1e6)
    rng = np.random.default_rng(0); t0 = time.time()   # 로그는 stdout 하나로 (tmux 는 tee, nohup 은 > 파일 리다이렉트)
    fixed_idx = rng.choice(N, 8, replace=False); fixed_cond = make_cond(lays[fixed_idx], mach[fixed_idx])
    for step in range(1, a.steps + 1):
        idx = rng.choice(N, a.batch); x0 = imgs[idx][:, None].clone(); cond = make_cond(lays[idx], mach[idx])
        if rng.random() < 0.5: x0 = x0.flip(-1); cond = cond.flip(-1)          # 좌우 반전 증강
        x0, cond = x0.to(dev), cond.to(dev)
        t = torch.randint(0, sch.config.num_train_timesteps, (a.batch,), device=dev)
        noise = torch.randn_like(x0); xt = sch.add_noise(x0, noise, t)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = unet(torch.cat([xt, cond], 1), t).sample
        loss = torch.nn.functional.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), unet.parameters()): pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f} {(time.time()-t0)/step:.3f}s/step", flush=True)
        if step % a.save_every == 0 or step == a.steps:
            ema.save_pretrained(out / "ema"); unet.save_pretrained(out / "last")
            g = torch.Generator(device=dev).manual_seed(0); ims = sample(ema, fixed_cond, steps=50, generator=g, device=dev)
            grid = np.hstack([np.vstack([lays[i], im]) for i, im in zip(fixed_idx, ims)])
            cv2.imwrite(str(out / f"sample_{step:06d}.png"), grid); json.dump(dict(step=step), open(out / "state.json", "w"))
            print("saved", step)
    if not a.no_export:
        import shutil
        if DIFFUSION.exists(): shutil.rmtree(DIFFUSION)
        shutil.copytree(out / "ema", DIFFUSION); print("모델 복사:", DIFFUSION)


if __name__ == "__main__":
    main()
