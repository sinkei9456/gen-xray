"""v1 학습 없는 필터: 크기·기준값 정하기(학습 대신). 어두운 점을 강조하는 black-hat 필터(detect.filter_candidates)의
필터 크기 4가지(5·7·9·11픽셀) × 기준값 29가지(4~60) 중 레이블 500장(검사 장비 표시 지움)에서 F1이 가장 높은 조합을 고른다.
사용: python develop/v1_train.py → models/v1_filter/params.json(고른 값), runs/v1_train/val_grid.csv(모든 조합의 F1)
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, cv2, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (공통 함수)
from paths import data_dir, model_dir, RUNS
from detect import filter_candidates, FILTER_KS, FILTER_TS, merge, score_image, prf, read_gt, _rel


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--val", default=str(data_dir(2) / "all" / "images"), help="크기·기준값을 고를 데이터(레이블 500장, 표시 지움)")
    ap.add_argument("--out", default=str(RUNS / "v1_train")); ap.add_argument("--model-dir", default=str(model_dir(1))); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(Path(a.val) / "*.png"))); lbl = Path(a.val).parent / "labels"; val = []
    for f in files:
        g = cv2.imread(f, 0); H, W = g.shape; val.append((read_gt(lbl / (Path(f).stem + ".txt"), W, H), {k: filter_candidates(g, k) for k in FILTER_KS}))
    rows = []
    for k in FILTER_KS:
        for t in FILTER_TS:
            tp = fp = fn = 0
            for gt, cand in val:
                bx = merge([b for b, _ in cand[k]], [c for _, c in cand[k]], t); x, y, n, _ = score_image(bx, gt, [], []); tp += x; fp += y; fn += n - x
            rows.append(dict(kernel=k, threshold=t, **prf(tp, fp, fn)))
    grid = pd.DataFrame(rows); grid.to_csv(out / "val_grid.csv", index=False)
    best = grid.sort_values(["f1", "kernel", "threshold"], ascending=[False, True, False]).iloc[0]; k, t = int(best.kernel), float(best.threshold)
    md = Path(a.model_dir); md.mkdir(parents=True, exist_ok=True)
    json.dump(dict(kernel=k, threshold=t, val=best.to_dict(), rule="--val 데이터에서 F1 최대, 같으면 작은 필터·큰 임계값", tuned_on=_rel(a.val)), open(md / "params.json", "w"), ensure_ascii=False, indent=1)
    print(f"튜닝 데이터에서 고른 값: 필터 {k}픽셀, 임계값 {t} (F1 {best.f1:.4f})", flush=True)


if __name__ == "__main__":
    main()
