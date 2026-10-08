"""오류 분석과 합성·실제 비교(보고서 2.4·3장 숫자). v5 평가(v5_eval.py)가 끝난 뒤 실행한다. 테스트셋 결과로 무엇을 고르거나 바꾸지 않고, 결과를 나눠 세기만 한다.
1) 테스트셋 1 미검출: 이미지·장비·정답 종류(점/영역)·그 정답에 걸친 가장 높은 confidence
2) 테스트셋 1 오검출: 장비·위치(막대/봉지/밖)·그 위치의 어두움(7×7 가운데값 − 최저 밝기)·가장 가까운 정답까지 거리,
   경우 나누기(같은 봉지를 연속 촬영한 다른 사진에서 검사 장비가 표시했는지 등, fp_cases)
3) 합성 이미지(최종 모델 학습·검증 5,000장) vs 테스트셋 1 실제: 장비별 바탕·봉지·막대 밝기와 잡음(1.5픽셀 흐림과의 차이의 표준편차) 분위
4) Diffusion 모델이 그린 그대로(밝기 맞추기 전)의 밝기와 500장 밝기 차이
5) 테스트셋 2 종류별 판정
사용: python develop/analyze.py  → runs/v5_eval/analysis.json, test1_fn.csv, test1_fp.csv, synthetic_vs_real_stats.csv, test2_decisions_kind.csv
"""
from __future__ import annotations
import json, os, sys, collections, itertools
from pathlib import Path
import numpy as np, cv2, pandas as pd
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (공통 함수)
sys.path.insert(0, str(Path(__file__).resolve().parent))      # src/develop/
from splits import all_paths, machine_of, _ts
from synth.render import load_raw, product_masks, PROFILES, MACHINES
from test_answers import box_size_500
from detect import merge, point_boxes, center_in, score_image, _score
from paths import MATERIAL, DIFFUSION, data_dir, eval_dir

CKPT = DIFFUSION                 # Diffusion 모델
E = eval_dir(5)                  # v5(최종 모델) 평가 폴더
SYN = data_dir(5)                # v5의 합성 학습·검증 이미지


def depth(g, med, b):
    x0, y0, x1, y1 = [int(round(v)) for v in b]; w = g[max(0, y0):y1, max(0, x0):x1]; m = med[max(0, y0):y1, max(0, x0):x1]
    return float((m - w).max()) if w.size else 0.0


def zone(pm, b):
    if pm is None: return "unknown"
    prod, st = pm; cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2); H, W = prod.shape; cx, cy = min(max(cx, 0), W - 1), min(max(cy, 0), H - 1)
    return "stick" if ndimage.binary_dilation(st, iterations=2)[cy, cx] else ("bag" if prod[cy, cx] else "outside")


def bursts(stems, P):
    """연속 촬영 묶음 번호: 같은 장비·같은 날, 앞 사진과 60초 안으로 이어진 사진(splits.test_session_siblings 와 같은 기준). 같은 봉지를 몇 초 간격으로 다시 찍은 것"""
    out, sid, prev = {}, -1, None
    for m, d, t, s in sorted((machine_of(P[s]), s[4:12], _ts(s), s) for s in stems):
        if prev is None or m != prev[0] or d != prev[1] or t - prev[2] > 60: sid += 1
        out[s] = sid; prev = (m, d, t)
    return out


def layout_diff(p, q):
    """같은 개수의 두 점 묶음을 회전·이동으로 겹친 뒤 대응점 거리의 RMS(픽셀). 점의 짝은 모든 순서 중 가장 작은 것"""
    p = np.asarray(p, float); a = p - p.mean(0); best = np.inf
    for perm in itertools.permutations(range(len(q))):
        b = np.asarray(q, float)[list(perm)]; b = b - b.mean(0); U, _, Vt = np.linalg.svd(a.T @ b)
        rot = Vt.T @ np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0]) @ U.T
        best = min(best, float(np.sqrt((((rot @ a.T).T - b) ** 2).sum(1).mean())))
    return best


def fp_cases(fp, info, missed, P):
    """오검출 경우 나누기(나눠 세기만 함). 같은 봉지 = 연속 촬영 묶음(bursts). 위에서부터 먼저 맞는 경우 하나.
    beside_missed       오검출 상자가 같은 이미지의 미검출 정답(점 상자·영역)과 겹침: 정답 위치가 이물에서 살짝 벗어나 같은 이물을 미검출 1 + 오검출 1로 셈
    boxed_in_other_shot 같은 봉지의 다른 사진에서 검사 장비가 이 사진의 모델 검출 수만큼 bounding box를 그림: 장비가 이 사진에서만 빠뜨린 이물
    found_in_all_shots  같은 봉지의 모든 사진에서 모델은 같은 수를 찾았고, 장비는 모든 사진에서 그보다 적게 표시
    this_shot_only      그 밖(모델이 같은 봉지의 다른 사진보다 이 사진에서 더 많이 찾음)
    boxed_shot, layout_diff_px: boxed_in_other_shot 일 때, 장비가 다 표시한 사진 중 이 사진의 모델 검출 위치와 배치가 가장 가까운 사진과 그 차이"""
    bid = bursts(list(info), P); mem = collections.defaultdict(list)
    for s in info: mem[bid[s]].append(s)
    rows = []
    for r in fp.itertuples():
        s = r.image; d = info[s]["det"]; others = [o for o in mem[bid[s]] if o != s]; full = [o for o in others if info[o]["n_ans"] == len(d)]
        diffs = sorted((layout_diff(d, info[o]["ans"]), o) for o in full if len(info[o]["ans"]) == len(d))
        if any(not (r.x1 < b[0] or b[2] < r.x0 or r.y1 < b[1] or b[3] < r.y0) for b in missed[s]): case = "beside_missed"
        elif full: case = "boxed_in_other_shot"
        elif others and all(len(info[o]["det"]) == len(d) for o in others): case = "found_in_all_shots"
        else: case = "this_shot_only"
        rows.append(dict(case=case, same_bag_shots=len(others) + 1, boxed_shot=diffs[0][1] if case == "boxed_in_other_shot" and diffs else "",
                         layout_diff_px=round(diffs[0][0], 1) if case == "boxed_in_other_shot" and diffs else np.nan))
    return pd.concat([fp.reset_index(drop=True), pd.DataFrame(rows, columns=["case", "same_bag_shots", "boxed_shot", "layout_diff_px"])], axis=1)


def test1_errors(th, P, bs):
    A, B = th["alpha"], th["beta"]; ans = json.load(open(E / "test1_answers.json")); fns, fps, tpd = [], [], []
    info, missed = {}, collections.defaultdict(list)   # 이미지별 모델 검출(α 이상, 짝지은 것 + 오검출) 중심·정답 중심·정답 수, 미검출 정답 상자
    for r in json.load(open(E / "test1_predictions.json")):
        s = r["image"]; m = machine_of(P[s]); a = ans[s]; gt = point_boxes(a["points"], bs[m]); regs = [(tuple(b), n) for b, n in a["regions"]]; ign = [tuple(b) for b in a["ignore"]]
        g = cv2.imread(str(E / "set" / "test1" / f"{s}.png"), 0).astype(np.float32); med = ndimage.median_filter(g, size=7); pm = product_masks(g)
        bx = merge(r["pred"], r["conf"], A); _, _, _, tags = score_image(bx, gt, regs, ign)
        info[s] = dict(det=[((p[0] + p[2]) / 2, (p[1] + p[3]) / 2) for (p, c), t in zip(bx, tags) if t in ("hit", "fp")],
                       ans=a["points"] + [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b, _ in regs], n_ans=len(a["points"]) + sum(n for _, n in regs))
        for (p, c), t in zip(bx, tags):
            if t == "hit": tpd.append(depth(g, med, p))
            if t == "fp":
                d = min([np.hypot((p[0] + p[2]) / 2 - (q[0] + q[2]) / 2, (p[1] + p[3]) / 2 - (q[1] + q[3]) / 2) for q in gt + [b for b, _ in regs]] or [999.0])
                fps.append(dict(image=s, machine=m, x0=p[0], y0=p[1], x1=p[2], y1=p[3], conf=c, zone=zone(pm, p), depth=depth(g, med, p), dist_to_answer=d))
        # 미검출: 같은 매칭을 다시 해 짝 없는 정답을 찾는다
        used = set(); cap = [n for _, n in regs]
        for (p, c), t in zip(bx, tags):
            if t != "hit": continue
            best, bj = 0, -1
            for j, q in enumerate(gt):
                if j in used: continue
                sc = _score(p, q, False)
                if sc > best: best, bj = sc, j
            if bj >= 0: used.add(bj); continue
            k = next((k for k, (b, _) in enumerate(regs) if center_in(p, b) and cap[k] > 0), None)
            if k is not None: cap[k] -= 1
        raw = list(zip(r["pred"], r["conf"]))
        for j, q in enumerate(gt):
            if j in used: continue
            bc = max([c for p, c in raw if _score(p, q, False) > 0] or [0.0])
            fns.append(dict(image=s, machine=m, kind="point", x=(q[0] + q[2]) / 2, y=(q[1] + q[3]) / 2, best_conf=bc)); missed[s].append(q)
        for k, (b, n) in enumerate(regs):
            for _ in range(cap[k]):
                bc = max([c for p, c in raw if center_in(p, b)] or [0.0])
                fns.append(dict(image=s, machine=m, kind="region", x=(b[0] + b[2]) / 2, y=(b[1] + b[3]) / 2, best_conf=bc)); missed[s].append(b)
    fn = pd.DataFrame(fns, columns=["image", "machine", "kind", "x", "y", "best_conf"]); fp = pd.DataFrame(fps, columns=["image", "machine", "x0", "y0", "x1", "y1", "conf", "zone", "depth", "dist_to_answer"])
    fn["band"] = np.where(fn.best_conf >= A, "alpha_or_more", np.where(fn.best_conf >= B, "beta_to_alpha", "below_beta"))
    fp = fp_cases(fp, info, missed, P)
    fn.to_csv(E / "test1_fn.csv", index=False); fp.to_csv(E / "test1_fp.csv", index=False)
    d5 = float(np.percentile(tpd, 5))
    return dict(n_fn=len(fn), fn_by_machine=fn.machine.value_counts().to_dict(), fn_by_band=fn.band.value_counts().to_dict(), fn_by_kind=fn.kind.value_counts().to_dict(),
                fn_conf_beta_to_alpha=[round(x, 3) for x in sorted(fn[fn.band == "beta_to_alpha"].best_conf)],
                n_fp=len(fp), fp_by_machine=fp.machine.value_counts().to_dict(), fp_by_zone=fp.zone.value_counts().to_dict(),
                fp_dist_median=round(float(fp.dist_to_answer.median()), 1) if len(fp) else None,
                hit_depth_p5=round(d5, 1), fp_depth_ge_hit_p5=int((fp.depth >= d5).sum()),
                fp_by_case=fp.case.value_counts().to_dict(), fp_layout_diff_max=float(fp.layout_diff_px.max()) if fp.layout_diff_px.notna().any() else None)


def img_stats(img):
    pm = product_masks(img.astype(np.float32))
    if pm is None: return None
    prod, st = pm; pack = prod & ~ndimage.binary_dilation(st, iterations=3); bgm = ~ndimage.binary_dilation(prod, iterations=6); bgm[:, :4] = False
    hp = img.astype(np.float32) - cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5)
    return dict(bg=float(np.median(img[bgm])), pack=float(np.median(img[pack])), stick=float(np.median(img[ndimage.binary_erosion(st, iterations=2)])) if st.sum() > 200 else np.nan,
                noise_pack=float(np.std(hp[pack])), noise_bg=float(np.std(hp[bgm])))


def synth_vs_real(P):
    meta = pd.read_csv(SYN / "meta.csv"); t1 = sorted(f.stem for f in (E / "set" / "test1").glob("*.png")); levels = json.load(open(MATERIAL / "levels_500.json")); rows = []; extra = {}
    for m in MACHINES:
        syn = [x for x in (img_stats(cv2.imread(str(SYN / sp / "images" / f"{n}.png"), 0)) for sp, n, mm in zip(meta.split, meta.image, meta.machine) if mm == m) if x]
        real = [x for x in (img_stats(cv2.imread(str(E / "set" / "test1" / f"{s}.png"), 0)) for s in t1 if machine_of(P[s]) == m) if x]
        for name, st in (("synthetic", syn), ("test1_real", real)):
            row = dict(machine=m, set=name, n=len(st))
            for k in ("bg", "pack", "stick"): row[k] = round(float(np.nanmedian([x[k] for x in st])), 1)
            for k in ("noise_pack", "noise_bg"): row[k] = "/".join(f"{v:.1f}" for v in np.percentile([x[k] for x in st], [10, 50, 90]))
            rows.append(row)
        r90 = np.percentile([x["noise_bg"] for x in real], 90)
        extra[m] = dict(synthetic_bg_noise_above_real_p90=round(float(np.mean([x["noise_bg"] > r90 for x in syn])), 3),
                        synthetic_minus_500=dict(**{k: round(float(np.nanmedian([x[k] for x in syn]) - np.median([l[k] for l in levels[m] if l[k] is not None])), 1) for k in ("bg", "pack", "stick")}))
    df = pd.DataFrame(rows); df.to_csv(E / "synthetic_vs_real_stats.csv", index=False)
    return df.to_dict(orient="records"), extra


def raw_diffusion_offsets(n_each=20):
    """Diffusion 모델이 그린 그대로(밝기 맞추기 전)의 밝기 − 500장 밝기(장비별 중간값)"""
    import sd.sample as sds
    from synth import parse_prompt
    from synth.compose import load_profiles
    load_profiles(MATERIAL); gj = json.load(open(SYN / "generation.json")); levels = json.load(open(MATERIAL / "levels_500.json"))
    sds.load(str(CKPT), steps=50, init="blank", t_start=gj["t_start"]); rng = np.random.default_rng(0); out = {}
    for m in MACHINES:
        acc = collections.defaultdict(list)
        for i in range(n_each):
            spec = parse_prompt(f"{m}, 스틱 {3 if i % 2 == 0 else 1}개"); spec.background = "diffusion"
            img, bag, stick, _ = sds.diffusion_background(spec, PROFILES[m], rng)
            pack = bag & ~ndimage.binary_dilation(stick, iterations=3); bgm = ~ndimage.binary_dilation(bag, iterations=6); st = ndimage.binary_erosion(stick, iterations=2)
            acc["bg"].append(np.median(img[bgm])); acc["pack"].append(np.median(img[pack])); acc["stick"].append(np.median(img[st]))
        out[m] = {k: round(float(np.median(acc[k]) - np.median([l[k] for l in levels[m] if l[k] is not None])), 1) for k in ("bg", "pack", "stick")}
    return out


def test2_kinds(P):
    d = pd.read_csv(E / "test2_decisions.csv"); kinds = []
    for s in d.image:
        g = load_raw(P[s])[0]; pm = product_masks(g)
        if s[4:12] != "20200727": k = "full bag"
        elif pm is None: k = "no bag"
        else: k = "cut bag, stick" if ndimage.label(pm[1])[1] else "cut bag, no stick"
        kinds.append(k)
    d["kind"] = kinds; d.to_csv(E / "test2_decisions_kind.csv", index=False)
    return {k: g.decision.value_counts().to_dict() for k, g in d.groupby("kind")}


def main():
    P = all_paths(); bs = box_size_500(); th = json.load(open(E / "thresholds.json"))
    res = dict(test1_errors=test1_errors(th, P, bs)); print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    res["synthetic_vs_real"], res["synthetic_extra"] = synth_vs_real(P); print(pd.DataFrame(res["synthetic_vs_real"]).to_string(index=False)); print(res["synthetic_extra"], flush=True)
    res["test2_by_kind"] = test2_kinds(P); print(res["test2_by_kind"])
    res["raw_diffusion_minus_500"] = raw_diffusion_offsets(); print("밝기 맞추기 전 Diffusion − 500장:", res["raw_diffusion_minus_500"])
    json.dump(res, open(E / "analysis.json", "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
