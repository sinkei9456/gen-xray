"""테스트셋 2(검사 장비 bounding box 없는 107장, 정성 평가)에서 v1~v5의 판정 비교. 학습·임계값 선택 없음(각 모델이 이미 정한 값 그대로).
v1 학습 없는 필터: models/v1_filter/params.json 의 필터 크기·기준값, 상자가 있으면 불량(재검사 범위 없음)
v2~v5 YOLO26s: 각자 검증셋에서 정한 α·β(runs/v번호_eval/thresholds.json), 이미지의 가장 높은 confidence로 불량 / 재검사 / 정상(이물 없다고 판정)
이미지 종류는 analyze.py 결과(runs/v5_eval/test2_decisions_kind.csv)를 쓴다. v5 판정이 v5_eval.py 결과와 같은지 확인한다.
사용: python develop/test2_compare.py → runs/test2_compare/ (per_image.csv, summary.csv, boxes.json, images/)
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, cv2, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (공통 함수)
from detect import merge, shown_boxes, decide, predict_all, filter_candidates as candidates
from paths import RUNS, NAMES, eval_dir, model_dir

SET = eval_dir(5) / "set" / "test2"   # 테스트셋 2 이미지(v5 평가 폴더)
# (키, 그림 표시 이름, 모델 파일(None = v1 필터), 임계값이 든 평가 폴더)
MODELS = [(f"v{n}", NAMES[n][2], None if n == 1 else model_dir(n) / "best.pt", eval_dir(n)) for n in (1, 2, 3, 4, 5)]
CHECK = "v5"   # 이 모델의 판정이 v5_eval.py 결과와 같은지 확인
COL = {"defect": (0, 0, 255), "recheck": (0, 140, 255)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", action="append", default=[], help="키=모델 파일 (예: v3=models/v3_synth500/best.pt)")
    ap.add_argument("--kinds", default=str(eval_dir(5) / "test2_decisions_kind.csv")); ap.add_argument("--out", default=str(RUNS / "test2_compare")); a = ap.parse_args()
    models, check_key = MODELS, CHECK; OUT = Path(a.out)
    over = dict(kv.split("=", 1) for kv in a.weights)
    models = [(k_, lab, over.get(k_, w), ed) for k_, lab, w, ed in models]; keys = [m[0] for m in models]
    YOLO = [(k_, w, ed) for k_, _, w, ed in models if w is not None]   # 임계값은 각 평가 폴더의 thresholds.json
    import random, torch; random.seed(0); np.random.seed(0); torch.manual_seed(0)
    from ultralytics import YOLO as Y
    OUT.mkdir(parents=True, exist_ok=True); kind = pd.read_csv(a.kinds).set_index("image")["kind"]
    files = sorted(SET.glob("*.png")); res = {}
    ch = json.load(open(model_dir(1) / "params.json")); k, t = ch["kernel"], ch["threshold"]
    for f in files:
        c = candidates(cv2.imread(str(f), 0), k); bx = merge([b for b, _ in c], [v for _, v in c], t)
        res.setdefault(f.stem, {})["v1"] = dict(boxes=[(p, v, "defect") for p, v in bx], decision="defect" if bx else "normal", score=max([v for _, v in bx], default=0.0))
    for name, w, ed in YOLO:
        th = json.load(open(ed / "thresholds.json")); A, B = th["alpha"], th["beta"]; model = Y(str(w))
        for r in predict_all(model, SET):
            bx = shown_boxes(r["pred"], r["conf"], A, B)
            res[r["image"]][name] = dict(boxes=[(p, c, "defect" if c >= A else "recheck") for p, c in bx], decision=decide(bx, A, B), score=max([c for _, c in bx], default=0.0))
        del model; torch.cuda.empty_cache()
    rows = []
    for s in sorted(res):
        row = dict(image=s, kind=kind[s])
        for m in keys:
            d = res[s][m]; row[f"{m}_decision"] = d["decision"]; row[f"{m}_n_defect_boxes"] = sum(x[2] == "defect" for x in d["boxes"]); row[f"{m}_max"] = round(d["score"], 4)
        rows.append(row)
    per = pd.DataFrame(rows); per.to_csv(OUT / "per_image.csv", index=False)
    json.dump({s: {m: [dict(box=[round(v, 1) for v in p], score=round(c, 4), level=lv) for p, c, lv in d["boxes"]] for m, d in res[s].items()} for s in res},
              open(OUT / "boxes.json", "w"), ensure_ascii=False)   # 모델별 검출 상자(v1: 필터 반응값, v2~v5: confidence)
    old = pd.read_csv(dict((m[0], m[3]) for m in models)[check_key] / "test2_decisions.csv").set_index("image").decision   # v5 평가(v5_eval.py) 결과와 같은지 확인
    assert all(old[s] == per.set_index("image").loc[s, f"{check_key}_decision"] for s in per.image), "최종 모델 판정이 평가 결과와 다름"
    summ = []
    for m in keys:
        for kd in ["전체"] + list(dict.fromkeys(per.kind)):
            sub = per if kd == "전체" else per[per.kind == kd]; vc = sub[f"{m}_decision"].value_counts()
            summ.append(dict(model=m, kind=kd, images=len(sub), defect=int(vc.get("defect", 0)), recheck=int(vc.get("recheck", 0)), normal=int(vc.get("normal", 0))))
    pd.DataFrame(summ).to_csv(OUT / "summary.csv", index=False); print(pd.DataFrame(summ).to_string(index=False))
    # 어느 한 모델이라도 정상이 아닌 이미지: 다섯 모델 결과를 옆으로 붙인 그림(빨강 = 불량 상자, 주황 = 재검사 상자)
    (OUT / "images").mkdir(exist_ok=True)
    for s in per.image:
        if all(res[s][m]["decision"] == "normal" for m in res[s]): continue
        g = cv2.imread(str(SET / f"{s}.png"), 0); tiles = []
        for m, lab, _, _ in models:
            im = cv2.cvtColor(cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            for (x0, y0, x1, y1), v, lv in res[s][m]["boxes"]:
                cv2.rectangle(im, (int(2 * x0), int(2 * y0)), (int(2 * x1), int(2 * y1)), COL[lv], 2)
                cv2.putText(im, f"{v:.2f}" if m != "v1" else f"{v:.0f}", (int(2 * x0), max(12, int(2 * y0) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL[lv], 1)
            cv2.putText(im, f"{lab}: {res[s][m]['decision']}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2); tiles.append(im)
        cv2.imwrite(str(OUT / "images" / f"{s}.png"), cv2.hconcat(tiles))
    print("그림:", len(list((OUT / "images").glob("*.png"))), "장 →", OUT / "images")


if __name__ == "__main__":
    main()
