"""Diffusion 모델로 합성 X-ray 이미지와 레이블 만들기. 학습 없음.

지시어("2호기, 스틱 3개, 이물 3개")를 규칙으로 읽어 배치도(봉지·막대 위치)를 만들고, Diffusion 모델이 빈 화면에서 배치도대로 봉지·막대를 그린다
(모델은 256×256으로 그리고 장비 크기로 되돌림). 밝기를 레이블 500장 중 한 장에 맞춘 뒤, 500장에서 떼어 낸 이물 점 패치
(또는 --shape-frac 비율만큼 새 형상)를 막대 안에 넣는다. 넣은 위치가 곧 레이블이다. 한 장 만드는 방법은 develop/make_synth.py 와 같다(synth/compose.py).
사용: python release/generate.py --prompt "2호기, 스틱 3개, 이물 3개" --n 5 --out ../runs/generate
     python release/generate.py --prompt "1호기, 스틱 1개, 이물 하나" --prompt "3호기, 스틱 3개, 이물 3개" --n 2 --out ../runs/generate
     python release/generate.py --random 20 --out ../runs/generate     (지시어를 레이블 500장의 장비·막대 수 비율대로 무작위로)
결과(--out 폴더): images/gen_00000.png …(장비 크기 회색조), labels/gen_00000.txt(YOLO 형식: 0 cx cy w h, 0~1), meta.csv(지시어·장비·막대 수·이물 수·이물 종류),
                  preview.png(앞 8장과 레이블 상자), generation.json(시드·새 형상 비율·t 등 실행 설정)
재료: data/material_500(이물 패치·밝기·장비별 설정값), Diffusion 모델: models/diffusion. 시드가 같으면 같은 이미지가 나온다.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, cv2, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (공통 함수)
from paths import MATERIAL, DIFFUSION, data_dir
from synth.shapes import load_blur
from synth.compose import load_profiles, load_levels, load_patches, instructions, make_one, level_error
from synth.render import MACHINES
import sd.sample as sds


def t_start_default(ckpt: Path, M: Path, by_m, u_by_m, prof500):
    """빈 화면에서 다시 그리는 깊이 t: models/diffusion/sample.json → 합성 5,000장을 만들 때 정한 값 → 없으면 500장 밝기와 비교해 정함"""
    for f in (Path(ckpt) / "sample.json", data_dir(5) / "generation.json"):
        if f.exists(): return int(json.load(open(f))["t_start"])
    errs = {t: level_error(by_m, u_by_m, prof500, t, np.random.default_rng(1)) for t in (800, 900, 950, 999)}
    return min(errs, key=errs.get)


def preview(out: Path, names, k=8):
    tiles = []
    for n in names[:k]:
        g = cv2.cvtColor(cv2.imread(str(out / "images" / f"{n}.png"), 0), cv2.COLOR_GRAY2BGR); H, W = g.shape[:2]
        for l in (out / "labels" / f"{n}.txt").read_text().split("\n"):
            if not l.strip(): continue
            _, cx, cy, w, h = map(float, l.split()); cv2.rectangle(g, (int((cx - w / 2) * W), int((cy - h / 2) * H)), (int((cx + w / 2) * W), int((cy + h / 2) * H)), (0, 0, 255), 1)
        s = 320 / max(H, W); tiles.append(cv2.copyMakeBorder(cv2.resize(g, (int(W * s), int(H * s))), 0, 330 - int(H * s), 0, 330 - int(W * s), cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    if tiles:
        while len(tiles) % 4: tiles.append(np.full_like(tiles[0], 255))
        cv2.imwrite(str(out / "preview.png"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prompt", action="append", default=[], help='지시어(여러 번 줄 수 있음), 예: "2호기, 스틱 3개, 이물 3개"')
    ap.add_argument("--n", type=int, default=1, help="지시어마다 만들 장 수"); ap.add_argument("--random", type=int, default=0, help="지시어를 무작위로 뽑아 만들 장 수")
    ap.add_argument("--shape-frac", type=float, default=0.3, help="이물 중 새 형상(선·조각) 비율(v5 학습 데이터와 같은 0.3, 0이면 점 패치만)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--material", default=str(MATERIAL)); ap.add_argument("--ckpt", default=str(DIFFUSION)); ap.add_argument("--t-start", type=int, default=None)
    a = ap.parse_args(); out = Path(a.out); M = Path(a.material); rng = np.random.default_rng(a.seed)
    if not a.prompt and not a.random: sys.exit("--prompt 또는 --random 을 주세요")
    import random, torch; random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    prof500 = load_profiles(M); by_m, u_by_m = load_patches(M); levels = load_levels(M)
    shapes = dict(frac=a.shape_frac, sigma=load_blur(M)) if a.shape_frac > 0 else None
    sds.load(a.ckpt, steps=50, init="blank"); sds._STATE["t_start"] = a.t_start or t_start_default(Path(a.ckpt), M, by_m, u_by_m, prof500)
    texts = [p for p in a.prompt for _ in range(a.n)] + (instructions(a.random, rng, json.load(open(M / "prompt_stats.json"))) if a.random else [])
    for d in ("images", "labels"): (out / d).mkdir(parents=True, exist_ok=True)
    rows = []
    for i, t in enumerate(texts):
        img, labels, info, spec, _, _ = make_one(t, by_m, u_by_m, rng, levels, "diffusion", shapes); H, W = img.shape; name = f"gen_{i:05d}"
        cv2.imwrite(str(out / "images" / f"{name}.png"), img)
        with open(out / "labels" / f"{name}.txt", "w") as f:
            for cx, cy, w, h in labels: f.write(f"0 {cx / W:.6f} {cy / H:.6f} {w / W:.6f} {h / H:.6f}\n")
        rows.append(dict(image=name, instruction=t, machine=spec.machine, width=W, height=H, n_sticks=spec.n_sticks, n_defects=len(labels),
                         n_shapes=sum(x["src"].startswith("shape:") for x in info), defect_src=";".join(x["src"] for x in info)))
        print(f"{i + 1}/{len(texts)} {name}: {t} → 이물 {len(labels)}개", flush=True)
    pd.DataFrame(rows).to_csv(out / "meta.csv", index=False); preview(out, [r["image"] for r in rows])
    json.dump(dict(seed=a.seed, shape_frac=a.shape_frac, t_start=sds._STATE["t_start"], prompts=a.prompt, n=a.n, random=a.random), open(out / "generation.json", "w"), ensure_ascii=False, indent=1)
    print("결과:", out)


if __name__ == "__main__":
    main()
