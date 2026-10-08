"""v1 학습 없는 필터 평가: models/v1_filter/params.json 의 필터 크기·기준값으로 테스트셋 1(1,870장)을 한 번 잰다.
정답 규칙·상자 합치기·짝짓기는 YOLO 모델과 같다(detect.py). 판정은 상자가 있으면 불량, 없으면 정상(재검사 범위 없음).
테스트셋 1 이미지(표시 지움)와 정답은 v5 평가 폴더(runs/v5_eval/set, test1_answers.json)를 같이 쓴다 → v5_eval.py 다음에 실행.
사용: python develop/v1_eval.py → runs/v1_eval/ (test1_per_image.csv, test1_result.csv)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, cv2, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (공통 함수)
from paths import model_dir, eval_dir
from splits import all_paths, machine_of
from test_answers import box_size_500
from detect import filter_candidates, merge, point_boxes, score_image, prf, center_in


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model-dir", default=str(model_dir(1))); ap.add_argument("--eval", default=str(eval_dir(5)), help="테스트셋 1 이미지·정답이 있는 평가 폴더")
    ap.add_argument("--out", default=str(eval_dir(1))); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); E = Path(a.eval)
    ch = json.load(open(Path(a.model_dir) / "params.json")); k, t = ch["kernel"], float(ch["threshold"])
    P = all_paths(); bs = box_size_500(); ans = json.load(open(E / "test1_answers.json")); per = []
    for s, A_ in ans.items():
        g = cv2.imread(str(E / "set" / "test1" / f"{s}.png"), 0); m = machine_of(P[s]); c = filter_candidates(g, k)
        bx = merge([b for b, _ in c], [v for _, v in c], t); regs = [(tuple(b), n) for b, n in A_["regions"]]; ign = [tuple(b) for b in A_["ignore"]]; gt = point_boxes(A_["points"], bs[m])
        tp, fp, n, _ = score_image(bx, gt, regs, ign); tps, fps, _, _ = score_image(bx, gt, regs, ign, strict=True)
        per.append(dict(image=s, machine=m, answers=n, tp=tp, fp=fp, fn=n - tp, tp_std=tps, fp_std=fps, decision="defect" if any(not any(center_in(p, b) for b in ign) for p, _ in bx) else "normal"))
    per = pd.DataFrame(per); per.to_csv(out / "test1_per_image.csv", index=False); res = []
    for name, sub in [("전체", per)] + [(m, per[per.machine == m]) for m in ("1호기", "2호기", "3호기")]:
        s1 = prf(int(sub.tp.sum()), int(sub.fp.sum()), int(sub.fn.sum())); s2 = prf(int(sub.tp_std.sum()), int(sub.fp_std.sum()), int(sub.answers.sum() - sub.tp_std.sum()))
        res.append(dict(set=name, images=len(sub), defects=int(sub.answers.sum()), **s1, f1_standard=s2["f1"], ok_images=int(((sub["tp"] == sub["answers"]) & (sub["fp"] == 0)).sum()),
                        불량=int((sub.decision == "defect").sum()), 정상=int((sub.decision == "normal").sum())))
    df = pd.DataFrame(res); df.to_csv(out / "test1_result.csv", index=False); print(df.to_string(index=False))


if __name__ == "__main__":
    main()
