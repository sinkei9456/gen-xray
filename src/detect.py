"""검출 모델 공통 함수(develop·release가 함께 씀).
- YOLO 학습 설정(train_yolo), 묶음 추론(predict_all, 입력 640×640), 정답 읽기(read_gt)
- 상자 합치기(merge), 결과 파일·그림에 보이는 상자(shown_boxes: 불량 = α 이상끼리 합침, 재검사 = β 이상을 모두 합쳤을 때 α 이상이 없는 묶음), 정답과 짝짓기(score_image), 정밀도·재현율·F1(prf), 검증셋에서 α·β 정하기(choose_on_val)
- 검사 장비 표시 지우기(erase_marks), 테스트셋 만들기(build_sets), 평가 절차(evaluate), 판정 불량·재검사·정상(decide), 결과 그림(draw)
- v1 학습 없는 필터의 후보 찾기(filter_candidates)
입력 크기: YOLO 학습·추론 모두 640×640. 사진의 긴 변을 640으로 맞추고(비율 유지) 남는 곳은 회색(114)으로 채운다. 상자는 원래 사진 좌표로 되돌린다.
"""
from __future__ import annotations
import collections, glob, json, os, sys
from pathlib import Path
import numpy as np, cv2, pandas as pd
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, PRETRAINED
from splits import all_paths, test_stems, machine_of, test_session_siblings
from synth.render import load_raw, product_masks
from test_answers import answers, near_copies, box_size_500

IMGSZ = 640                                                   # YOLO 입력 640×640
CONFS = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]    # α·β 후보(0.05 간격)
FILTER_KS = (5, 7, 9, 11); FILTER_TS = list(range(4, 62, 2))  # v1 필터 크기·기준값 후보

def read_gt(lbl, W, H):
    out = []
    if lbl.exists():
        for line in lbl.read_text().strip().splitlines():
            c, cx, cy, w, h = map(float, line.split())
            if int(c) == 0: out.append((cx * W - w * W / 2, cy * H - h * H / 2, cx * W + w * W / 2, cy * H + h * H / 2))
    return out


def predict_all(model, img_dir, imgsz=IMGSZ):
    """폴더의 png를 이름 순서로 32장씩 추론 → 사진마다 (정답 상자(옆 labels/ 폴더가 있으면), 예측 상자, confidence). confidence 0.01 이상 전부"""
    files = sorted(glob.glob(str(Path(img_dir) / "*.png"))); lbl_dir = Path(img_dir).parent / "labels"; rows = []
    for i in range(0, len(files), 32):
        b = files[i:i + 32]; res = model.predict(b, conf=0.01, imgsz=imgsz, verbose=False, classes=[0], rect=False)   # 항상 640×640 이미지(긴 변 640, 남는 곳 회색 여백)
        for f, r in zip(b, res):
            H, W = r.orig_shape
            rows.append(dict(image=Path(f).stem, W=W, H=H, gt=read_gt(lbl_dir / (Path(f).stem + ".txt"), W, H),
                             pred=r.boxes.xyxy.cpu().numpy().tolist(), conf=r.boxes.conf.cpu().numpy().tolist()))
    return rows


def _score(p, g, strict):
    if strict:   # 표준 매칭: 두 상자를 같은 10픽셀로 맞춘 뒤 IoU 0.5 이상
        pc, gc = ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2), ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)
        p = (pc[0] - 5, pc[1] - 5, pc[0] + 5, pc[1] + 5); g = (gc[0] - 5, gc[1] - 5, gc[0] + 5, gc[1] + 5)
    ix = max(0, min(p[2], g[2]) - max(p[0], g[0])); iy = max(0, min(p[3], g[3]) - max(p[1], g[1])); inter = ix * iy
    iou = inter / ((p[2] - p[0]) * (p[3] - p[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter + 1e-9)
    if strict: return iou if iou >= 0.5 else 0
    cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2; inside = g[0] <= cx <= g[2] and g[1] <= cy <= g[3]
    s = max(iou, 0.5 if inside else 0); return s if s >= 0.3 else 0


def merge(pred, conf, th):
    """confidence th 이상 상자 중 겹치거나 맞닿는 것들을 바깥 테두리 하나로 합친다 → [(box, conf)]"""
    bs = [(list(p), c) for p, c in zip(pred, conf) if c >= th]; changed = True
    while changed:
        changed = False
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                a, b = bs[i][0], bs[j][0]
                if not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]):
                    bs[i] = ([min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])], max(bs[i][1], bs[j][1])); bs.pop(j); changed = True; break
            if changed: break
    return sorted(bs, key=lambda x: -x[1])


def shown_boxes(pred, conf, alpha, beta):
    """결과 파일·그림에 보이는 상자 → [(box, conf)].
    불량 상자 = confidence α 이상 후보끼리 합친 상자(점수를 셀 때와 같은 상자).
    재검사 상자 = β 이상 후보끼리 합친 상자 중 가장 높은 confidence가 α 미만인 것(불량 상자와 겹치는 약한 후보는 불량 상자에 들어가지 않고 버려진다).
    이미지 판정(가장 높은 confidence)은 β 이상 후보를 모두 합쳤을 때와 같다."""
    return sorted(merge(pred, conf, alpha) + [(p, c) for p, c in merge(pred, conf, beta) if c < alpha], key=lambda x: -x[1])


def center_in(p, b): cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2; return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def point_boxes(pts, wh): w, h = wh; return [(x - w / 2, y - h / 2, x + w / 2, y + h / 2) for x, y in pts]


def score_image(boxes, gt, regs, ign, strict=False):
    """boxes: 합친 예측 [(box, conf)], gt: 점 정답 상자 목록 → (검출, 오검출, 정답 수, 상자별 판정 목록)"""
    cap = [n for _, n in regs]; used = set(); tp = fp = 0; tags = []
    for p, c in boxes:
        if any(center_in(p, b) for b in ign): tags.append("ignored"); continue
        best, bj = 0, -1
        for j, g in enumerate(gt):
            if j in used: continue
            s = _score(p, g, strict)
            if s > best: best, bj = s, j
        if bj >= 0: used.add(bj); tp += 1; tags.append("hit"); continue
        k = next((k for k, (b, _) in enumerate(regs) if center_in(p, b)), None)
        if k is not None:
            if cap[k] > 0: cap[k] -= 1; tp += 1; tags.append("hit")
            else: tags.append("ignored")
            continue
        fp += 1; tags.append("fp")
    return tp, fp, len(gt) + sum(n for _, n in regs), tags


def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0; R = tp / (tp + fn) if tp + fn else 0
    return dict(tp=tp, fp=fp, fn=fn, precision=P, recall=R, f1=2 * P * R / (P + R) if P + R else 0)


def choose_on_val(model, val_dir, imgsz):
    """검증셋에서 α·β 결정: 상자 합치기 뒤 α = F1 최대(동점이면 가운데), β = α에서 내려가며 정밀도 0.9 이상을 유지하는 가장 작은 값"""
    rows = predict_all(model, val_dir, imgsz); tab = []
    for c in CONFS:
        tp = fp = fn = 0
        for r in rows:
            t, f, n, _ = score_image(merge(r["pred"], r["conf"], c), r["gt"], [], []); tp += t; fp += f; fn += n - t
        tab.append(dict(conf=c, **prf(tp, fp, fn)))
    tab = pd.DataFrame(tab); best = tab.f1.max(); tied = tab[tab.f1 >= best - 1e-9].conf.tolist(); alpha = float(tied[len(tied) // 2]); beta = alpha
    for c in sorted([c for c in CONFS if c <= alpha], reverse=True):
        if float(tab[tab.conf == c].precision.iloc[0]) >= 0.9: beta = c
        else: break
    return alpha, float(beta), tab


def erase_marks(g, mk):
    return cv2.inpaint(g.astype(np.uint8), ndimage.binary_dilation(mk, iterations=1).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)


def build_sets(root: Path):
    P = all_paths(); T = set(test_stems()); SIB = test_session_siblings(P); cnt = collections.Counter(); ans = {}; t1, t2 = [], []
    for d in ("test1", "test2"): (root / d).mkdir(parents=True, exist_ok=True)
    cand = [s for s in P if s not in T and s not in SIB]; NEAR = near_copies(P, [s for s in T if s in P], cand)
    for s, p in P.items():
        g, mk = load_raw(p); pts, regs, ign = answers(mk); boxed = bool(pts or regs)
        if s in T: cnt["정답 있는 500장(학습 재료)"] += 1; continue
        if not boxed: cnt["테스트셋 2: bounding box 없음"] += 1; cv2.imwrite(str(root / "test2" / f"{s}.png"), erase_marks(g, mk)); t2.append(s); continue
        if s in SIB: cnt["500장과 60초 안 연속 촬영"] += 1; continue
        if s in NEAR: cnt["500장과 거의 같은 사진(다른 날)"] += 1; continue
        cnt["테스트셋 1"] += 1; cv2.imwrite(str(root / "test1" / f"{s}.png"), erase_marks(g, mk)); t1.append(s)
        ans[s] = dict(points=pts, regions=[[list(b), n] for b, n in regs], ignore=[list(b) for b in ign])
    return P, t1, t2, ans, dict(cnt)


def draw(g, boxes, alpha, beta, pts=(), regs=(), wh=(10, 10), scale=2):
    im = cv2.cvtColor(cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR); S = lambda v: int(round(v * scale))
    w, h = wh
    for x, y in pts: cv2.rectangle(im, (S(x - w / 2), S(y - h / 2)), (S(x + w / 2), S(y + h / 2)), (0, 200, 0), 1)
    for (x0, y0, x1, y1), n in regs:
        for k in range(S(x0), S(x1), 6): cv2.line(im, (k, S(y0)), (min(k + 3, S(x1)), S(y0)), (0, 200, 0), 1); cv2.line(im, (k, S(y1)), (min(k + 3, S(x1)), S(y1)), (0, 200, 0), 1)
        for k in range(S(y0), S(y1), 6): cv2.line(im, (S(x0), k), (S(x0), min(k + 3, S(y1))), (0, 200, 0), 1); cv2.line(im, (S(x1), k), (S(x1), min(k + 3, S(y1))), (0, 200, 0), 1)
    for (x0, y0, x1, y1), c in boxes:
        if c < beta: continue
        col = (0, 0, 255) if c >= alpha else (0, 165, 255)
        cv2.rectangle(im, (S(x0) - 3, S(y0) - 3), (S(x1) + 3, S(y1) + 3), col, 1); cv2.putText(im, f"{c:.2f}", (S(x1) + 5, S(y0) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return im


def decide(boxes, alpha, beta):
    m = max([c for _, c in boxes], default=0.0); return "defect" if m >= alpha else ("recheck" if m >= beta else "normal")


def filter_candidates(g, k):
    """v1 필터: → [(box, 반응)] 봉지 안의 black-hat 반응 극댓값(반응 4 이상)"""
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    res = cv2.GaussianBlur(cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, K).astype(np.float32), (0, 0), 0.8)
    pm = product_masks(g.astype(np.float32)); inside = ndimage.binary_dilation(pm[0], iterations=2) if pm is not None else np.ones_like(res, bool)
    peak = (res == ndimage.maximum_filter(res, size=5)) & (res >= FILTER_TS[0]) & inside
    ys, xs = np.where(peak)
    return [((float(x) - 5, float(y) - 5, float(x) + 5, float(y) + 5), float(res[y, x])) for y, x in zip(ys, xs)]


def train_yolo(data_dir, out, model=None, epochs=100, patience=20, imgsz=IMGSZ, batch=32, seed=0, fraction=1.0, export_dir=None):
    """YOLO26s 학습(v2~v5 공통 설정). 최대 epochs, 검증셋 성능이 patience epoch 동안 나아지지 않으면 멈추고 가장 좋은 epoch의 가중치(best.pt)를 남긴다.
    data_dir: train/·val/ 이 있는 폴더. fraction: 학습셋 중 쓸 비율(1보다 작으면 짧은 실행 확인용). export_dir 를 주면 best.pt·args.yaml·results.csv 를 그 폴더로 복사"""
    import shutil
    out = Path(out); out.mkdir(parents=True, exist_ok=True); data_dir = Path(data_dir).resolve(); model = str(model or PRETRAINED)
    data = out / "data.yaml"; data.write_text(f"path: {data_dir}\ntrain: train/images\nval: val/images\nnames:\n  0: defect\n")   # 데이터 위치는 실행할 때 절대 경로로 적음
    import torch, gc
    from ultralytics import YOLO
    last = out / "train" / "weights" / "last.pt"
    if not (out / "train_done").exists():          # 재부팅으로 끊겼으면 last.pt 에서 이어서
        if last.exists() and fraction == 1.0:
            try: YOLO(str(last)).train(resume=True)
            except AssertionError as e: print("resume skipped:", e)
        else:
            YOLO(model).train(data=str(data), epochs=epochs, patience=patience, imgsz=imgsz, batch=batch, project=str(out.resolve()), name="train", exist_ok=True,
                                seed=seed, fraction=fraction, mosaic=0.5, scale=0.2, fliplr=0.5, flipud=0.5, degrees=10, hsv_h=0, hsv_s=0, hsv_v=0.2, close_mosaic=5, workers=4, verbose=False)
        (out / "train_done").write_text("ok"); gc.collect(); torch.cuda.empty_cache()
    res = pd.read_csv(out / "train" / "results.csv"); res.columns = [c.strip() for c in res.columns]
    print(f"epoch {len(res)} (검증셋 mAP50 최고 {res['metrics/mAP50(B)'].max():.3f})")
    if export_dir:
        e = Path(export_dir); e.mkdir(parents=True, exist_ok=True)
        for f in ("weights/best.pt", "args.yaml", "results.csv"): shutil.copy(out / "train" / f, e / Path(f).name)
        print("모델 복사:", e)



def _rel(p):
    p = Path(p).resolve(); return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def evaluate(weights, val, out, png=None, imgsz=IMGSZ, no_test2=False):
    """YOLO 모델 평가(v2~v5 공통): 검증셋에서 α·β → 테스트셋 1 정량 평가(정답 규칙: test_answers.py) → 테스트셋 2 판정(no_test2=True 면 건너뜀: v2.
    v2의 테스트셋 2 판정은 develop/test2_compare.py). png 를 주면 결과 그림 저장(v5).
    out/ 에 thresholds.json, val_thresholds_merged.csv, set/(표시 지운 테스트셋 1·2), test1_answers.json, test1_predictions.json, test1_per_image.csv, test1_result.csv, test2_predictions.json, test2_decisions.csv"""
    out = Path(out); out.mkdir(parents=True, exist_ok=True); png = Path(png) if png else None
    import random, torch; random.seed(0); np.random.seed(0); torch.manual_seed(0)
    from ultralytics import YOLO
    model = YOLO(str(weights))
    # 1) 검증셋에서 α·β (테스트셋을 보기 전에 정해서 저장)
    alpha, beta, tab = choose_on_val(model, val, imgsz); tab.to_csv(out / "val_thresholds_merged.csv", index=False)
    json.dump(dict(weights=_rel(weights), alpha=alpha, beta=beta, val_at_alpha=tab[tab.conf == alpha].iloc[0].to_dict(),
                   rule="상자 합치기 후처리 뒤, alpha=검증셋 F1 최대(동점이면 가운데), beta=alpha 에서 내려가며 검증셋 정밀도 0.9 이상을 유지하는 가장 작은 값"),
              open(out / "thresholds.json", "w"), ensure_ascii=False, indent=1)
    print(f"검증셋: α={alpha} β={beta}"); print(tab.to_string(index=False), flush=True)
    # 2) 테스트셋 구성(정답 규칙 포함) → 추론
    P, t1, t2, ans, cnt = build_sets(out / "set"); json.dump(ans, open(out / "test1_answers.json", "w")); json.dump(cnt, open(out / "set_counts.json", "w"), ensure_ascii=False, indent=1)
    print(cnt, flush=True)
    r1 = predict_all(model, out / "set" / "test1", imgsz); json.dump(r1, open(out / "test1_predictions.json", "w"))
    # 3) 테스트셋 1 정량 평가 + ok/ng 그림
    bs = box_size_500(); per = []
    if png: (png / "test1" / "ok").mkdir(parents=True, exist_ok=True); (png / "test1" / "ng").mkdir(parents=True, exist_ok=True)
    for r in r1:
        s = r["image"]; m = machine_of(P[s]); A_ = ans[s]; regs = [(tuple(b), n) for b, n in A_["regions"]]; ign = [tuple(b) for b in A_["ignore"]]
        bx_a = merge(r["pred"], r["conf"], alpha); bx_b = shown_boxes(r["pred"], r["conf"], alpha, beta)   # bx_a: 점수, bx_b: 판정·그림
        gt = point_boxes(A_["points"], bs[m]); tp, fp, n, _ = score_image(bx_a, gt, regs, ign); tps, fps, _, _ = score_image(bx_a, gt, regs, ign, strict=True)
        boxes_nonign = [(p, c) for p, c in bx_b if not any(center_in(p, b) for b in ign)]
        per.append(dict(image=s, machine=m, answers=n, tp=tp, fp=fp, fn=n - tp, tp_std=tps, fp_std=fps, decision=decide(boxes_nonign, alpha, beta)))
        ok = (tp == n and fp == 0)
        if png:
            g = cv2.imread(str(out / "set" / "test1" / f"{s}.png"), 0); name = f"{s}.png" if ok else f"FN{n - tp}_FP{fp}_{s}.png"
            cv2.imwrite(str(png / "test1" / ("ok" if ok else "ng") / name), draw(g, bx_b, alpha, beta, A_["points"], regs, bs[m]))
    per = pd.DataFrame(per); per.to_csv(out / "test1_per_image.csv", index=False); res = []
    for name, sub in [("전체", per)] + [(m, per[per.machine == m]) for m in ("1호기", "2호기", "3호기")]:
        s1 = prf(int(sub.tp.sum()), int(sub.fp.sum()), int(sub.fn.sum())); s2 = prf(int(sub.tp_std.sum()), int(sub.fp_std.sum()), int(sub.answers.sum() - sub.tp_std.sum()))
        res.append(dict(set=name, images=len(sub), defects=int(sub.answers.sum()), **s1, f1_standard=s2["f1"], ok_images=int(((sub["tp"] == sub["answers"]) & (sub["fp"] == 0)).sum()),
                        불량=int((sub.decision == "defect").sum()), 재검사=int((sub.decision == "recheck").sum()), 정상=int((sub.decision == "normal").sum())))
    df = pd.DataFrame(res); df.to_csv(out / "test1_result.csv", index=False); print(df.to_string(index=False))
    # 4) 테스트셋 2 판정(no_test2 면 건너뜀). 결과 그림은 png 를 준 경우(v5)만
    if no_test2: return
    r2 = predict_all(model, out / "set" / "test2", imgsz); json.dump(r2, open(out / "test2_predictions.json", "w"))
    if png: (png / "test2").mkdir(parents=True, exist_ok=True)
    d2 = []
    for r in r2:
        s = r["image"]; bx_b = shown_boxes(r["pred"], r["conf"], alpha, beta); dec = decide(bx_b, alpha, beta)
        if png: cv2.imwrite(str(png / "test2" / f"{dec}_{s}.png"), draw(cv2.imread(str(out / "set" / "test2" / f"{s}.png"), 0), bx_b, alpha, beta))
        d2.append(dict(image=s, decision=dec, n_boxes_alpha=sum(c >= alpha for _, c in bx_b), max_conf=max([c for _, c in bx_b], default=0.0)))
    d2 = pd.DataFrame(d2); d2.to_csv(out / "test2_decisions.csv", index=False); print("테스트셋 2 판정:", d2.decision.value_counts().to_dict())

