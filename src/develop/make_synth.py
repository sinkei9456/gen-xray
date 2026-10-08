"""합성 학습 데이터 만들기: 지시어 + Diffusion 모델 → 봉지·막대 이미지, 그 위에 이물(점 패치 70% + 새 형상 30%)을 막대 안에 넣고 레이블을 함께 저장.
한 장 만드는 방법은 synth/compose.py(make_one). 지시어는 레이블 500장의 장비·막대 수 비율대로 뽑는다(data/material_500/prompt_stats.json).
빈 화면에서 다시 그리는 깊이(t)는 500장의 봉지·막대 밝기와 가장 가까운 값으로 정한다(검증셋·테스트셋 안 봄).
앞 n - n_val 장은 학습셋, 뒤 n_val 장은 검증셋. 이미지 이름 synth_00000.png …(번호 = 생성 순서).
사용: python develop/make_synth.py → data/v5_synth5000/ (train/, val/, meta.csv, generation.json), models/diffusion/sample.json(정한 t)   [v5: 5,000장, 검증 500, 새 형상 30%, 시드 0]
     재현 점검: python develop/make_synth.py --limit 40 --out <임시 폴더>  (앞 40장만 그려 제출본과 비교)
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, cv2, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (공통 함수)
sys.path.insert(0, str(Path(__file__).resolve().parent))      # src/develop/
from paths import MATERIAL, DIFFUSION, data_dir
from synth.render import MACHINES
from synth.shapes import load_blur
from synth.compose import load_profiles, load_levels, load_patches, instructions, make_one, level_error
import sd.sample as sds


def save_sample_json(ckpt, t):
    """정한 빈 화면 출발 깊이 t 를 Diffusion 모델 폴더의 sample.json 에 적는다(release/generate.py 가 읽음)"""
    json.dump(dict(steps=50, init="blank", t_start=int(t), note="합성 5,000장을 만들 때 정한 값(develop/make_synth.py, 500장 밝기와 가장 가까운 t)"),
              open(Path(ckpt) / "sample.json", "w"), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=str(DIFFUSION)); ap.add_argument("--n", type=int, default=5000); ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--out", default=str(data_dir(5))); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--material", default=str(MATERIAL))
    ap.add_argument("--t-start", type=int, default=None, help="주면 밝기 비교 없이 이 값 사용(점검용)")
    ap.add_argument("--limit", type=int, default=None, help="재현 점검용: 앞 limit장만 그림")
    ap.add_argument("--bg", default="diffusion", choices=["diffusion", "procedural"], help="봉지·막대 그리는 방법(procedural 은 비교 실험용)")
    ap.add_argument("--shape-frac", type=float, default=0.3, help="이물 중 새 형상(선·조각)으로 넣는 비율(0이면 점 패치만)")
    a = ap.parse_args(); out = Path(a.out); rng = np.random.default_rng(a.seed)
    import random, torch; random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)   # 사진마다 쓰는 torch 난수는 rng 에서 뽑아 고정
    M = Path(a.material); prof500 = load_profiles(M); by_m, u_by_m = load_patches(M); levels = load_levels(M)
    shapes = dict(frac=a.shape_frac, sigma=load_blur(M)) if a.shape_frac > 0 else None
    if a.bg == "diffusion": sds.load(a.ckpt, steps=50, init="blank")
    # 빈 화면에서 얼마나 깊게 다시 그릴지(t)는 500장의 봉지·막대 밝기와 가장 가까운 값으로 정한다(검증셋·테스트셋 안 봄)
    errs = {0: 0.0} if a.bg == "procedural" else {a.t_start: 0.0} if a.t_start else {t: level_error(by_m, u_by_m, prof500, t, np.random.default_rng(1)) for t in (800, 900, 950, 999)}
    t_best = min(errs, key=errs.get)
    if a.bg == "diffusion": sds._STATE["t_start"] = t_best
    print("빈 화면 출발점별 밝기 차이:", {k: round(v, 2) for k, v in errs.items()}, "→ t =", t_best, flush=True)
    if a.bg == "diffusion" and a.limit is None and a.t_start is None and Path(a.ckpt).resolve() == DIFFUSION.resolve():
        save_sample_json(a.ckpt, t_best)   # 제출 모델 폴더의 models/diffusion/sample.json(점검 실행·다른 모델 폴더에서는 쓰지 않음)
    texts = instructions(a.n, rng, json.load(open(M / "prompt_stats.json"))); rows = []
    for split in ("train", "val"): (out / split / "images").mkdir(parents=True, exist_ok=True); (out / split / "labels").mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(texts):
        if a.limit is not None and i >= a.limit: break
        split = "val" if i >= a.n - a.n_val else "train"
        img, labels, info, spec, _, _ = make_one(t, by_m, u_by_m, rng, levels, a.bg, shapes); H, W = img.shape; name = f"synth_{i:05d}"
        cv2.imwrite(str(out / split / "images" / f"{name}.png"), img)
        with open(out / split / "labels" / f"{name}.txt", "w") as f:
            for cx, cy, w, h in labels: f.write(f"0 {cx / W:.6f} {cy / H:.6f} {w / W:.6f} {h / H:.6f}\n")
        rows.append(dict(image=name, split=split, instruction=t, machine=spec.machine, n_sticks=spec.n_sticks, n_defects=len(labels),
                         n_shapes=sum(x["src"].startswith("shape:") for x in info), patch_src=";".join(x["src"] for x in info), t_start=t_best))
        if i % 100 == 0: print(f"{i}/{a.n}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / "meta.csv", index=False)
    json.dump(dict(bg=a.bg, t_start=t_best, level_error=errs, seed=a.seed, n=a.n, n_val=a.n_val, shapes=shapes), open(out / "generation.json", "w"), indent=1)
    print(df.groupby(["split", "machine", "n_sticks"]).size().to_string()); print("이물 수:", df.groupby("split").n_defects.sum().to_dict(), "새 형상 수:", df.groupby("split").n_shapes.sum().to_dict())



if __name__ == "__main__":
    main()
