"""v1~v5 비교표(테스트셋 1 정량 평가): v1 학습 없는 필터, v2 실제 500장, 새 형상을 섞은 Diffusion 합성 v3 500장·v4 2,000장·v5 5,000장(최종).
결과를 모으기만 하고 무엇을 고르지 않는다. 철사 모양 이물(테스트셋 1에서 알려진 점이 아닌 이물 1개) 위치의 최고 confidence 도 함께 적는다.
v5의 오검출마다 같은 위치(같은 이미지, 상자 중심 6픽셀 안)에서 v2·v3·v4도 오검출이 나왔는지와 그 위치의 가장 높은 confidence 를 적는다(fp_spots.csv, fp_by_case.csv).
사용: python develop/compare.py → runs/compare/ (compare.csv, fp_spots.csv, fp_by_case.csv)   (v5 오검출 경우는 analyze.py 결과 runs/v5_eval/test1_fp.csv)
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, cv2, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (공통 함수)
from splits import all_paths, machine_of
from test_answers import box_size_500
from detect import merge, point_boxes, score_image
from paths import RUNS, NAMES, eval_dir, model_dir

FINAL = eval_dir(5)   # 테스트셋 1 정답·이미지(모든 모델 공통)
WIRE = "002_20200915_215738(0)"   # 테스트셋 1의 철사 모양 이물 이미지(오류 분석에서 찾음)
# (번호, 학습 데이터, 학습 / 검증). 학습 기록(results.csv)은 models/v번호_*/, 평가 결과는 runs/v번호_eval/
MODELS = [(1, "레이블 500장(표시 지움)", "튜닝 500"), (2, "레이블 500장(표시 지움)", "400 / 100"), (3, "합성 5,000장의 앞 500장", "400 / 100"),
          (4, "합성 5,000장의 앞 2,000장", "1,800 / 200"), (5, "합성 5,000장", "4,500 / 500")]


def wire_score(ed, filt):
    """철사 모양 이물 정답 위치(중심에서 15픽셀 안) 검출의 최고 confidence(v1은 필터 반응값)"""
    ans = json.load(open(FINAL / "test1_answers.json"))[WIRE]; pts = ans["points"] + [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b, _ in ans["regions"]]
    near = lambda b: any(abs((b[0] + b[2]) / 2 - x) <= 15 and abs((b[1] + b[3]) / 2 - y) <= 15 for x, y in pts)
    if filt:
        from detect import filter_candidates as candidates
        ch = json.load(open(model_dir(1) / "params.json")); g = cv2.imread(str(FINAL / "set" / "test1" / f"{WIRE}.png"), 0)
        return max([v for b, v in candidates(g, ch["kernel"]) if near(b)], default=0.0)
    r = next(r for r in json.load(open(ed / "test1_predictions.json")) if r["image"] == WIRE)
    return max([c for b, c in zip(r["pred"], r["conf"]) if near(b)], default=0.0)


CASES = {"boxed_in_other_shot": "같은 봉지 다른 사진에서는 장비가 표시", "found_in_all_shots": "같은 봉지 모든 사진에서 모델만 찾음",
         "beside_missed": "미검출 정답 바로 옆(같은 이물)", "this_shot_only": "이 사진에서만 검출"}   # analyze.fp_cases


def model_fps(ed, P, bs, ans):
    """평가 폴더의 테스트셋 1 예측에서 α 이상 오검출 [(이미지, x, y, confidence)](detect.py 와 같은 짝짓기)와 이미지별 원래 상자"""
    A = json.load(open(ed / "thresholds.json"))["alpha"]; out, raw = [], {}
    for r in json.load(open(ed / "test1_predictions.json")):
        s = r["image"]; a = ans[s]; bx = merge(r["pred"], r["conf"], A); raw[s] = list(zip(r["pred"], r["conf"]))
        _, _, _, tags = score_image(bx, point_boxes(a["points"], bs[machine_of(P[s])]), [(tuple(b), n) for b, n in a["regions"]], [tuple(b) for b in a["ignore"]])
        out += [(s, (p[0] + p[2]) / 2, (p[1] + p[3]) / 2, c) for (p, c), t in zip(bx, tags) if t == "fp"]
    return out, raw


def fp_spots(fp_csv, out):
    """v5의 오검출(analyze.py 의 test1_fp.csv, 경우 나누기 포함)과 같은 위치에서 v2·v3·v4도 오검출이 나왔는지. 같은 위치 = 같은 이미지에서 상자 중심 6픽셀 안"""
    P = all_paths(); bs = box_size_500(); ans = json.load(open(FINAL / "test1_answers.json"))
    fp = pd.read_csv(fp_csv); fp["x"], fp["y"] = (fp.x0 + fp.x1) / 2, (fp.y0 + fp.y1) / 2
    near = lambda s, x, y, L: [c for i, u, v, c in L if i == s and np.hypot(u - x, v - y) <= 6]
    spots = fp[["image", "machine", "x", "y", "conf", "depth", "case"]].rename(columns={"conf": "v5_conf"}).round(3); extra = {}
    for k in (2, 3, 4, 5):
        F, raw = model_fps(eval_dir(k), P, bs, ans)
        spots[f"v{k}_오검출"] = [bool(near(r.image, r.x, r.y, F)) for r in fp.itertuples()]
        spots[f"v{k}_그 위치 최고 confidence"] = [round(max([c for p, c in raw[r.image] if np.hypot((p[0] + p[2]) / 2 - r.x, (p[1] + p[3]) / 2 - r.y) <= 6], default=0.0), 3) for r in fp.itertuples()]
        extra[k] = (len(F), sum(not near(s, x, y, [(r.image, r.x, r.y, 0) for r in fp.itertuples()]) for s, x, y, _ in F))   # (오검출 수, 그중 v5와 다른 위치인 것)
    assert spots["v5_오검출"].all() and extra[5] == (len(fp), 0), "v5 오검출이 test1_fp.csv 와 다름"
    spots.to_csv(out / "fp_spots.csv", index=False)
    rows = [{"경우": CASES[c], **{f"v{k}": int(spots[spots.case == c][f"v{k}_오검출"].sum()) for k in (2, 3, 4, 5)}} for c in CASES]   # 숫자 = 각 모델의 오검출 수
    rows.append({"경우": "합계(각 모델의 오검출 수)", **{f"v{k}": int(spots[f"v{k}_오검출"].sum()) for k in (2, 3, 4, 5)}})
    rows.append({"경우": "v5와 다른 위치의 오검출", **{f"v{k}": extra[k][1] for k in (2, 3, 4, 5)}})
    tab = pd.DataFrame(rows); tab.to_csv(out / "fp_by_case.csv", index=False); print(tab.to_string(index=False))
    diff = spots[~spots["v2_오검출"]]
    print("v5는 오검출, v2는 아님:"); print(diff[["image", "machine", "case", "v5_conf", "v2_그 위치 최고 confidence"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(RUNS / "compare"))
    ap.add_argument("--fp", default=str(FINAL / "test1_fp.csv"), help="v5 오검출 목록(analyze.py 결과)"); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows = []
    for n, data, split in MODELS:
        ed = eval_dir(n); tr = None if n == 1 else model_dir(n) / "results.csv"
        if not (ed / "test1_result.csv").exists(): print("결과 없음:", ed); continue
        t = pd.read_csv(ed / "test1_result.csv").set_index("set"); s = t.loc["전체"]
        if tr is None:
            ch = json.load(open(model_dir(1) / "params.json")); thr = f"필터 {ch['kernel']}픽셀, 기준 {ch['threshold']:g}"; vf1 = ch["val"]["f1"]; ep = "-"
        else:
            th = json.load(open(ed / "thresholds.json")); thr = f"α {th['alpha']:g}, β {th['beta']:g}"; vf1 = th["val_at_alpha"]["f1"]
            r = pd.read_csv(tr); r.columns = [c.strip() for c in r.columns]; f = 0.1 * r["metrics/mAP50(B)"] + 0.9 * r["metrics/mAP50-95(B)"]
            ep = f"{len(r)} (가장 좋은 {int(r.loc[f.idxmax(), 'epoch'])})"
        rows.append({"모델": NAMES[n][1], "학습 데이터": data, "학습 / 검증": split, "임계값": thr, "튜닝·검증 F1": round(vf1, 4), "epoch": ep,
                     "이물": int(s.defects), "검출": int(s.tp), "미검출": int(s.fn), "오검출": int(s.fp),
                     "정밀도": round(s.precision, 4), "재현율": round(s.recall, 4), "F1": round(s.f1, 4), "IoU 0.5 기준 F1": round(s.f1_standard, 4),
                     **{f"{m} F1": round(t.loc[m].f1, 4) for m in ("1호기", "2호기", "3호기")},
                     "불량": int(s["불량"]), "재검사": int(s["재검사"]) if "재검사" in s and not pd.isna(s["재검사"]) else "-", "정상": int(s["정상"]),
                     "철사 모양 이물 최고값": round(wire_score(ed, tr is None), 3)})
    df = pd.DataFrame(rows); df.to_csv(out / "compare.csv", index=False); print(df.T.to_string())
    if Path(a.fp).exists(): fp_spots(Path(a.fp), out)
    else: print("오검출 목록 없음(analyze.py 를 먼저 실행하거나 --fp 로 지정):", a.fp)

if __name__ == "__main__":
    main()
