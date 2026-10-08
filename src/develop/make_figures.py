"""보고서 그림(그림 안 글자는 영문만: 서버에 한글 폰트 없음). 결과를 읽어 그리기만 한다(모델·임계값은 여기서 정하지 않음. 예시 이미지는 정한 규칙이나 고정한 난수로 고름).
사용: python develop/make_figures.py <이름>   (이름: data testprep gen compare val test errors shapes fp models blackhat test2models fpmodels missed)
     → doc/fig/fig*.png(제목 있음), doc/fig/report/·report/slides_assets/(제목 없는 복사본)
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np, cv2, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (공통 함수)
sys.path.insert(0, str(Path(__file__).resolve().parent))      # src/develop/
from splits import all_paths, test_stems, machine_of
from synth.render import load_raw
from prepare_material import human_boxes, separate, cut_patch
from paths import ROOT, MATERIAL, DIFFUSION, RUNS, NAMES, data_dir, eval_dir, model_dir
from detect import shown_boxes   # 결과 그림에 보이는 상자(불량 = α 이상끼리 합침, 재검사 = β 이상을 모두 합쳤을 때 α 이상이 없는 묶음)

FIG = ROOT / "doc" / "fig"; FIG.mkdir(parents=True, exist_ok=True)
EVAL_DIR = eval_dir(5)   # v5(최종 모델) 평가 폴더
SYN = data_dir(5)        # v5의 합성 학습·검증 이미지
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9})


def rgb(path): return np.array(Image.open(path).convert("RGB"))


def pick(n_each=1):
    P = all_paths(); S = [s for s in test_stems() if s in P]; out = []
    for m in ("1호기", "2호기", "3호기"):
        for era in ("stick3", "stick1"):
            c = [s for s in S if machine_of(P[s]) == m and (s[4:12] <= "20200715") == (era == "stick3")]
            if c: out.append((m, era, c[len(c) // 2]))
    return P, out


def fig_data():
    P, items = pick(); MN = {"1호기": "Machine 1", "2호기": "Machine 2", "3호기": "Machine 3"}
    # 보고서 그림 1: 원본 사진(검사 장비가 그린 색 사각형) + 확대
    sel = [items[0], items[3], items[4]]
    fig, ax = plt.subplots(2, 3, figsize=(12, 7.2))
    for k, (m, era, s) in enumerate(sel):
        im = rgb(P[s]); g, mk = load_raw(P[s]); H, W = g.shape; hb = human_boxes(s, W, H)
        ax[0, k].imshow(im); ax[0, k].set_title(f"{MN[m]} original ({W}x{H} px)"); ax[0, k].axis("off")
        cx, cy, _, _ = hb[0]; y0, x0 = max(0, int(cy) - 25), max(0, int(cx) - 25)
        ax[1, k].imshow(im[y0:y0 + 50, x0:x0 + 50], interpolation="nearest"); ax[1, k].set_title("zoom: box drawn by the X-ray machine around the defect"); ax[1, k].axis("off")
        ax[0, k].add_patch(plt.Rectangle((x0, y0), 50, 50, fill=False, ec="yellow", lw=1.2))
    fig.suptitle("Fig. 1  Every image already has colored boxes drawn by the X-ray machine around each defect", y=0.99)
    plt.tight_layout(); save_both(plt.gcf(), "fig1_data_problem.png", dpi=110)
    # 보고서 그림 2: 500장에서 봉지·막대 사진과 이물 패치 분리
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(3, 4, figsize=(13, 9.5))
    for k, (m, era, s) in enumerate(sel):
        g, mk = load_raw(P[s]); H, W = g.shape; hb = human_boxes(s, W, H); bag, smooth, ml = separate(g, mk, hb, rng)
        cx, cy, w, h = hb[0]; y0, x0 = max(0, int(cy) - 30), max(0, int(cx) - 30)
        ax[k, 0].imshow(rgb(P[s])); ax[k, 0].set_title(f"{MN[m]}: original"); ax[k, 0].axis("off")
        ax[k, 1].imshow(bag, cmap="gray", vmin=0, vmax=255); ax[k, 1].set_title("bag + sticks only (boxes and defects removed)"); ax[k, 1].axis("off")
        ax[k, 2].imshow(np.hstack([g[y0:y0 + 60, x0:x0 + 60], bag[y0:y0 + 60, x0:x0 + 60]]), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax[k, 2].set_title("zoom: before | after (stick end kept square)"); ax[k, 2].axis("off")
        ratio = cut_patch(smooth, ml, cx, cy, w, h)[0]
        ax[k, 3].imshow(ratio, cmap="gray", vmin=0.4, vmax=1.0, interpolation="nearest"); ax[k, 3].set_title(f"defect patch (transmission, {ratio.shape[0]}x{ratio.shape[0]} px)"); ax[k, 3].axis("off")
    fig.suptitle("Fig. 2  Separating the 500 labeled images into bag+stick images and defect patches", y=0.995)
    plt.tight_layout(); save_both(plt.gcf(), "fig2_separation.png", dpi=110)
    # 보고서 그림 3: 이물 패치 모음
    z = np.load(MATERIAL / "patches.npz", allow_pickle=True); meta = json.loads(str(z["meta"])); pt = z["patch"]
    fig, ax = plt.subplots(3, 12, figsize=(13, 3.6))
    for row, m in enumerate(("1호기", "2호기", "3호기")):
        idx = [i for i, mt in enumerate(meta) if mt["machine"] == m]; sel_i = np.random.default_rng(row).choice(idx, 12, replace=False)
        for c, i in enumerate(sel_i):
            q = pt[i]; r = q.shape[0] // 2; ax[row, c].imshow(q[r - 5:r + 6, r - 5:r + 6], cmap="gray", vmin=0.4, vmax=1.0, interpolation="nearest"); ax[row, c].axis("off")
        ax[row, 0].set_title(MN[m], loc="left")
    n = {m: sum(mt["machine"] == m for mt in meta) for m in ("1호기", "2호기", "3호기")}
    fig.suptitle(f"Fig. 3  Defect patches (center 11x11 px) cut from the 500 labeled images (Machine 1/2/3: {n['1호기']}/{n['2호기']}/{n['3호기']}, total {len(meta)})", y=0.99)
    plt.tight_layout(); save_both(plt.gcf(), "fig3_patches.png", dpi=110)


def fig_testprep():
    """보고서 그림 7(fig6_testprep.png): 테스트셋 1 이미지 → 정답(검사 장비 bounding box) → 표시를 지운 모델 입력"""
    P = all_paths(); E = EVAL_DIR; ans = json.load(open(E / "test1_answers.json"))
    names = sorted(ans); meta = pd.DataFrame(dict(image=names, machine=[machine_of(P[n]) for n in names]))
    from test_answers import box_size_500
    bs = box_size_500(); fig, ax = plt.subplots(3, 3, figsize=(12, 10.5))
    for k, m in enumerate(("1호기", "2호기", "3호기")):
        sub = meta[(meta.machine == m) & meta.image.map(lambda n: len(ans[n]["points"]) > 0 and not ans[n]["regions"] and not ans[n]["ignore"])]
        s = sub.image.iloc[len(sub) // 2]; im = rgb(P[s]); g = cv2.imread(str(E / "set" / "test1" / f"{s}.png"), 0); w, h = bs[m]
        cx, cy = ans[s]["points"][0]; y0, x0 = max(0, int(cy) - 25), max(0, int(cx) - 25)
        ax[k, 0].imshow(im); ax[k, 0].set_title(f"Machine {k + 1} test image: {s}", fontsize=8); ax[k, 0].add_patch(plt.Rectangle((x0, y0), 50, 50, fill=False, ec="yellow", lw=1.2))
        ax[k, 1].imshow(im[y0:y0 + 50, x0:x0 + 50], interpolation="nearest"); ax[k, 1].set_title("answer = center of the machine's box (green)")
        ax[k, 2].imshow(g[y0:y0 + 50, x0:x0 + 50], cmap="gray", vmin=0, vmax=255, interpolation="nearest"); ax[k, 2].set_title("model input: machine marks inpainted, defect kept")
        for x, y in ans[s]["points"]:
            for a_ in (ax[k, 1], ax[k, 2]): a_.add_patch(plt.Rectangle((x - w / 2 - x0, y - h / 2 - y0), w, h, fill=False, ec="lime", lw=1.2))
        for a_ in ax[k]: a_.axis("off")
    fig.suptitle("Fig. 6  Test set 1 preparation: read the machine's boxes as answers, then remove the marks", y=0.995)
    plt.tight_layout(); save_both(plt.gcf(), "fig6_testprep.png", dpi=110)


def fig_gen():
    import sd.sample as sds
    from synth import parse_prompt
    from synth.render import procedural_background, PROFILES
    from sd.data import layout_map
    from synth.compose import load_profiles, load_patches, load_levels, make_one
    load_profiles(MATERIAL); by_m, u_by_m = load_patches(MATERIAL); levels = load_levels(MATERIAL); gj = json.load(open(SYN / "generation.json"))
    sds.load(str(DIFFUSION), steps=50, init="blank", t_start=gj["t_start"]); shapes = gj.get("shapes")   # 새 형상 비율·흐림(최종 모델 합성과 같게)
    texts = ["1호기, 스틱 3개, 이물 3개", "2호기, 스틱 1개, 이물 하나", "3호기, 스틱 3개, 이물 3개"]
    eng = ["'Machine 1, 3 sticks, 3 defects'", "'Machine 2, 1 stick, 1 defect'", "'Machine 3, 3 sticks, 3 defects'"]
    fig, ax = plt.subplots(3, 4, figsize=(13, 9.5))
    for k, (t, e) in enumerate(zip(texts, eng)):
        rng = np.random.default_rng(10 + k); spec = parse_prompt(t); spec.background = "diffusion"; prof = PROFILES[spec.machine]
        img, labels, info, _, base, (bag, stick) = make_one(t, by_m, u_by_m, rng, levels, "diffusion", shapes)   # 실제로 쓴 배치도
        ax[k, 0].imshow(layout_map(bag, stick), cmap="gray"); ax[k, 0].set_title(f"instruction {e}\n-> layout (gray=bag, white=stick)"); ax[k, 0].axis("off")
        ax[k, 1].imshow(np.full_like(img, int(prof["bg"])), cmap="gray", vmin=0, vmax=255); ax[k, 1].set_title("start: empty X-ray screen"); ax[k, 1].axis("off")
        ax[k, 3].imshow(img, cmap="gray", vmin=0, vmax=255); ax[k, 3].set_title("+ defects and label boxes\n(green = dot patch, red = new shape)"); ax[k, 3].axis("off")
        for (cx, cy, w, h), d in zip(labels, info): ax[k, 3].add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec="red" if d["src"].startswith("shape:") else "lime", lw=1.2))
        ax[k, 2].imshow(base, cmap="gray", vmin=0, vmax=255); ax[k, 2].set_title("Diffusion model output: bag + sticks"); ax[k, 2].axis("off")
    fig.suptitle(f"Fig. 4  Instruction -> layout -> Diffusion model draws bag and sticks on an empty screen -> defects inserted (dot patches 70%, new shapes 30%; labels come free)", y=0.995)
    plt.tight_layout(); save_both(plt.gcf(), "fig4_generation.png", dpi=110)
    # 실제(500장 재료) vs 생성 확인용 그림. 같은 파일 이름을 뒤에 도는 fig_compare(보고서 그림 6)가 덮어쓴다
    P, items = pick(); meta = pd.read_csv(SYN / "meta.csv")
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    for k, m in enumerate(("1호기", "2호기", "3호기")):
        s = [x for x in items if x[0] == m][0][2]; ax[0, k].imshow(cv2.imread(str(MATERIAL / "bags" / f"{s}.png"), 0), cmap="gray", vmin=0, vmax=255)
        ax[0, k].set_title(f"real (Machine {k + 1}, boxes/defects removed)"); ax[0, k].axis("off")
        name = meta[(meta.machine == m) & (meta.split == "train")].image.iloc[0]
        ax[1, k].imshow(cv2.imread(str(SYN / "train" / "images" / f"{name}.png"), 0), cmap="gray", vmin=0, vmax=255)
        ax[1, k].set_title(f"generated (Machine {k + 1})"); ax[1, k].axis("off")
    fig.suptitle("Fig. 5  Real bag+stick images (top) vs images generated by the Diffusion model with defect patches (bottom)", y=0.99)
    plt.tight_layout(); save_both(plt.gcf(), "fig5_real_vs_generated.png", dpi=110)


def fig_val():
    E = EVAL_DIR; tab = pd.read_csv(E / "val_thresholds_merged.csv"); th = json.load(open(E / "thresholds.json"))
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, lab in [("precision", "precision"), ("recall", "recall"), ("f1", "F1")]: ax.plot(tab.conf, tab[c], marker="o", ms=3, label=lab)
    ax.axvline(th["alpha"], color="red", ls="--", label=f"alpha = {th['alpha']} (defect)"); ax.axvline(th["beta"], color="orange", ls="--", label=f"beta = {th['beta']} (re-inspect)")
    ax.set_xlabel("confidence threshold"); ax.set_ylim(0, 1.02); ax.legend(loc="lower left")
    nv = len(list((SYN / "val" / "images").glob("*.png"))); fig.suptitle(f"Fig. 7  Validation set ({nv:,} synthetic images, overlapping boxes merged): alpha and beta", fontsize=10)
    plt.tight_layout(); save_both(plt.gcf(), "fig7_thresholds.png", dpi=120)


def _eval_ctx():
    from detect import merge, point_boxes, center_in
    from test_answers import box_size_500
    E = EVAL_DIR; th = json.load(open(E / "thresholds.json")); P = all_paths()
    rows = {r["image"]: r for r in json.load(open(E / "test1_predictions.json"))}; ans = json.load(open(E / "test1_answers.json")); bs = box_size_500()
    return E, th, P, rows, ans, bs, merge, point_boxes, center_in


def _draw(a_, boxes, th, pts=(), regs=(), ign=(), wh=(10, 10), x0=0, y0=0):
    w, h = wh
    for x, y in pts: a_.add_patch(plt.Rectangle((x - w / 2 - x0, y - h / 2 - y0), w, h, fill=False, ec="lime", lw=1.2))
    for (bx0, by0, bx1, by1), n in regs: a_.add_patch(plt.Rectangle((bx0 - x0, by0 - y0), bx1 - bx0, by1 - by0, fill=False, ec="lime", lw=1.2, ls="--"))
    for bx0, by0, bx1, by1 in ign: a_.add_patch(plt.Rectangle((bx0 - x0, by0 - y0), bx1 - bx0, by1 - by0, fill=False, ec="gray", lw=1.2, ls=":"))
    for (px0, py0, px1, py1), c in boxes:
        col = "red" if c >= th["alpha"] else "orange"
        a_.add_patch(plt.Rectangle((px0 - 3 - x0, py0 - 3 - y0), px1 - px0 + 6, py1 - py0 + 6, fill=False, ec=col, lw=1)); a_.text(px1 + 4 - x0, py0 - y0, f"{c:.2f}", color=col, fontsize=7, clip_on=True)


def fig_test():
    """보고서 그림 13(fig8_test_examples.png): 테스트셋 1 결과 예시"""
    E, th, P, rows, ans, bs, merge, point_boxes, center_in = _eval_ctx(); per = pd.read_csv(E / "test1_per_image.csv")
    ok = per[(per["tp"] == per["answers"]) & (per["fp"] == 0)]
    pick_ = [ok[ok.machine == m].image.iloc[len(ok[ok.machine == m]) // 3] for m in ("1호기", "2호기", "3호기")] + [ok[ok.machine == "3호기"].image.iloc[len(ok[ok.machine == "3호기"]) * 2 // 3]]
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.4))
    for a_, s in zip(ax, pick_):
        g = cv2.imread(str(E / "set" / "test1" / f"{s}.png"), 0); m = machine_of(P[s]); A_ = ans[s]
        a_.imshow(g, cmap="gray", vmin=0, vmax=255); a_.axis("off"); a_.set_title(f"Machine {m[0]}: {s}", fontsize=8)
        _draw(a_, shown_boxes(rows[s]["pred"], rows[s]["conf"], th["alpha"], th["beta"]), th, A_["points"], [(tuple(b), n) for b, n in A_["regions"]], [tuple(b) for b in A_["ignore"]], bs[m])
    fig.suptitle("Fig. 8  Test set 1 (real images, machine marks removed). green = answer from the machine's box, red = detection >= alpha", y=0.99)
    plt.tight_layout(); save_both(plt.gcf(), "fig8_test_examples.png", dpi=110)


def fig_errors():
    """보고서 그림 14(fig9_errors.png): 오류 분석 예시(analyze.py 결과에서 자동으로 고름). 위 = 원본(검사 장비 표시 있음), 아래 = 모델 입력(표시 지움) + 정답·검출"""
    E, th, P, rows, ans, bs, merge, point_boxes, center_in = _eval_ctx(); cases = []
    for s in sorted(rows):
        A_ = ans[s]
        if A_["ignore"] and not any(c[0] == "icon" for c in cases): b = A_["ignore"][0]; cases.append(("icon", s, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, ""))
        if A_["regions"] and not any(c[0] == "merged" for c in cases):
            b = A_["regions"][0][0]
            if any(center_in(p, b) for p, c in merge(rows[s]["pred"], rows[s]["conf"], th["alpha"])): cases.append(("merged", s, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, ""))
    fp = pd.read_csv(E / "test1_fp.csv"); fn = pd.read_csv(E / "test1_fn.csv")
    if len(fp):
        f0 = fp.sort_values("depth", ascending=False).iloc[0]; cases.append(("fp", f0.image, (f0.x0 + f0.x1) / 2, (f0.y0 + f0.y1) / 2, ""))
    WIRE = "002_20200915_215738(0)"
    near = fn[(fn.image != WIRE) & (fn.kind == "point")].sort_values("best_conf", ascending=False)
    if len(near): n0 = near.iloc[0]; cases.append(("near", n0.image, n0.x, n0.y, f"{n0.best_conf:.2f}"))
    if WIRE in rows:
        q = ans[WIRE]["points"][0]; w_fn = WIRE in set(fn.image)
        cw = max([c for p, c in merge(rows[WIRE]["pred"], rows[WIRE]["conf"], th["beta"]) if abs((p[0] + p[2]) / 2 - q[0]) < 8 and abs((p[1] + p[3]) / 2 - q[1]) < 8] or [0])
        cases.append(("wire" if w_fn else "wire_hit", WIRE, q[0], q[1], f"{cw:.2f}"))
    order = ["icon", "merged", "fp", "near", "wire", "wire_hit"]; cases.sort(key=lambda c: order.index(c[0]))
    band = lambda v: "re-inspect band" if float(v) >= th["beta"] else "below beta"
    title = {"icon": lambda v: "(a) 'T' mark outside the bag:\nnot a defect, excluded from answers", "merged": lambda v: "(b) overlapping boxes drawn as one outline:\nanswer = region (dashed)",
             "fp": lambda v: "(c) counted as false detection: clear dot\nbut the machine drew no box", "near": lambda v: f"(d) missed at alpha: confidence {v}\n({band(v)})",
             "wire": lambda v: f"(e) missed at alpha: wire-shaped defect\nconfidence {v} ({band(v)})", "wire_hit": lambda v: f"(e) wire-shaped defect:\ndetected (confidence {v})"}
    from matplotlib.patches import Patch
    n = len(cases); fig = plt.figure(figsize=(3.2 * n, 7.9)); gs = fig.add_gridspec(3, n, height_ratios=[1, 1, 0.28])
    ax = np.array([[fig.add_subplot(gs[j, k]) for k in range(n)] for j in range(2)])
    for k, (name, s, cx, cy, v) in enumerate(cases):
        im = rgb(P[s]); g = cv2.imread(str(E / "set" / "test1" / f"{s}.png"), 0); H, W = g.shape; m = machine_of(P[s]); A_ = ans[s]
        x0, y0 = int(min(max(cx - 25, 0), W - 50)), int(min(max(cy - 25, 0), H - 50))
        ax[0, k].imshow(im[y0:y0 + 50, x0:x0 + 50], interpolation="nearest"); ax[0, k].set_title(title[name](v), fontsize=8)
        ax[1, k].imshow(g[y0:y0 + 50, x0:x0 + 50], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        _draw(ax[1, k], shown_boxes(rows[s]["pred"], rows[s]["conf"], th["alpha"], th["beta"]), th, A_["points"], [(tuple(b), n) for b, n in A_["regions"]], [tuple(b) for b in A_["ignore"]], bs[m], x0, y0)
        ax[1, k].set_xlim(-0.5, 49.5); ax[1, k].set_ylim(49.5, -0.5)
        for a_ in ax[:, k]: a_.axis("off")
    lg = fig.add_subplot(gs[2, :]); lg.axis("off")   # 색 설명: 위 줄은 검사 장비 표시, 아래 줄은 정답·검출
    lg.text(0.5, 1.0, "top row: original image. Every colored box and the 'T' letter are marks drawn by the inspection machine (box colors are the machine's own; all treated the same)",
            ha="center", va="top", fontsize=9, transform=lg.transAxes)
    lg.legend(handles=[Patch(fc="none", ec="lime", label="answer (where the machine drew a box)"), Patch(fc="none", ec="lime", ls="--", label="answer region (overlapping boxes)"),
                       Patch(fc="none", ec="gray", ls=":", label="'T' mark area (excluded)"), Patch(fc="none", ec="red", label="YOLO >= alpha"), Patch(fc="none", ec="orange", label="YOLO beta~alpha (re-inspect)")],
              loc="lower center", ncol=5, fontsize=9, frameon=False, title="bottom row: model input (machine marks erased), number = confidence", title_fontsize=9)
    fig.suptitle("Fig. 9  Error analysis on test set 1 (top: original with machine marks, bottom: model input with answers and detections)", y=0.995)
    plt.tight_layout(); save_both(plt.gcf(), "fig9_errors.png", dpi=110)


def fig_compare():
    """보고서 그림 6(fig5_real_vs_generated.png): 합성 5장 / 테스트셋 1 실제 5장(장비 구성 같게, 고정 시드 무작위) / 테스트셋 2 종류별 5장 + 밝기·잡음 표"""
    from synth.render import load_raw, product_masks
    from scipy import ndimage
    E = EVAL_DIR; P = all_paths(); rng = np.random.default_rng(0); comp = ["1호기", "1호기", "2호기", "2호기", "3호기"]
    meta = pd.read_csv(SYN / "meta.csv"); t1 = sorted(f.stem for f in (E / "set" / "test1").glob("*.png")); t2 = sorted(f.stem for f in (E / "set" / "test2").glob("*.png"))
    syn = [rng.choice(meta[meta.machine == m].image.tolist()) for m in comp]; real = [rng.choice([s for s in t1 if machine_of(P[s]) == m]) for m in comp]
    def kind(s):
        g = load_raw(P[s])[0]; pm = product_masks(g)
        if s[4:12] != "20200727": return "full bag (no box)"
        if pm is None: return "no bag"
        prod, st = pm; return "cut bag, stick" if ndimage.label(st)[1] else "cut bag, no stick"
    kinds = {}
    for s in t2: kinds.setdefault(kind(s), []).append(s)
    t2pick = [kinds["full bag (no box)"][0], kinds["full bag (no box)"][-1]] + [kinds[k][0] for k in ("cut bag, stick", "cut bag, no stick", "no bag") if k in kinds]
    fig, ax = plt.subplots(3, 5, figsize=(16, 10))
    for k in range(5):
        a = cv2.imread(str(SYN / ("train" if (SYN / "train" / "images" / f"{syn[k]}.png").exists() else "val") / "images" / f"{syn[k]}.png"), 0)
        ax[0, k].imshow(a, cmap="gray", vmin=0, vmax=255); ax[0, k].set_title(f"synthetic, Machine {comp[k][0]}", fontsize=9)
        ax[1, k].imshow(cv2.imread(str(E / "set" / "test1" / f"{real[k]}.png"), 0), cmap="gray", vmin=0, vmax=255); ax[1, k].set_title(f"test set 1 (real), Machine {comp[k][0]}", fontsize=9)
        if k < len(t2pick):
            s = t2pick[k]; ax[2, k].imshow(cv2.imread(str(E / "set" / "test2" / f"{s}.png"), 0), cmap="gray", vmin=0, vmax=255); ax[2, k].set_title(f"test set 2 (no box): {kind(s)}", fontsize=9)
        for a_ in ax[:, k]: a_.axis("off")
    fig.suptitle("Fig. 5  Synthetic images (top) vs real test set 1 (middle, machine marks removed) vs test set 2 (bottom, images without any machine box)", y=0.995)
    plt.tight_layout(); save_both(plt.gcf(), "fig5_real_vs_generated.png", dpi=100)
    # 밝기·잡음 확인용 출력(장비별 중간값). 보고서의 표(10/50/90% 값)는 analyze.py 의 synthetic_vs_real_stats.csv 이고, 이 함수는 그 파일을 덮어쓰지 않는다
    def stats(img):
        pm = product_masks(img.astype(np.float32))
        if pm is None: return None
        prod, st = pm; pack = prod & ~ndimage.binary_dilation(st, iterations=3); bgm = ~ndimage.binary_dilation(prod, iterations=6); bgm[:, :4] = False
        hp = img.astype(np.float32) - cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5)
        return dict(bg=np.median(img[bgm]), pack=np.median(img[pack]), stick=np.median(img[ndimage.binary_erosion(st, iterations=2)]) if st.sum() > 200 else np.nan, noise=np.std(hp[pack]))
    rows = []
    for m in ("1호기", "2호기", "3호기"):
        for name, files in [("합성", [SYN / sp / "images" / f"{n}.png" for sp, n in zip(meta.split, meta.image) if meta.set_index("image").machine[n] == m]),
                            ("테스트셋 1(실제)", [E / "set" / "test1" / f"{s}.png" for s in t1 if machine_of(P[s]) == m])]:
            st_ = [x for x in (stats(cv2.imread(str(f), 0)) for f in files) if x]
            rows.append(dict(machine=m, set=name, n=len(st_), **{k: round(float(np.nanmedian([x[k] for x in st_])), 1) for k in ("bg", "pack", "stick", "noise")}))
    print(pd.DataFrame(rows).to_string(index=False)); print({k: len(v) for k, v in kinds.items()})


def save_both(fig, name, dpi=110):
    """제목 있는 그림은 doc/fig, 제목 없는 복사본은 보고서(doc/fig/report)·발표자료(report/slides_assets)용"""
    fig.savefig(FIG / name, dpi=dpi)
    if fig._suptitle is not None: fig._suptitle.set_text(""); fig.tight_layout()
    for d in (FIG / "report", ROOT / "report" / "slides_assets"): d.mkdir(parents=True, exist_ok=True); fig.savefig(d / name, dpi=dpi)
    plt.close(fig)


def fig_models():
    """보고서 그림 9(fig10_models.png): 비교 모델이 쓰는 데이터. v1 학습 없는 필터가 실제 테스트 이미지에서 잡는 것, v2 실제 500장(표시 지움)+사람 레이블, v3~v5 점 + 새 형상 합성"""
    from detect import filter_candidates as candidates, merge
    from test_answers import box_size_500
    E = EVAL_DIR; ans = json.load(open(E / "test1_answers.json")); ch = json.load(open(model_dir(1) / "params.json")); P = all_paths(); bs = box_size_500()
    per = pd.read_csv(eval_dir(1) / "test1_per_image.csv"); s = per[(per.machine == "2호기") & (per["fn"] == 0) & (per["fp"] == 2)].image.iloc[0]; g = cv2.imread(str(E / "set" / "test1" / f"{s}.png"), 0); m = machine_of(P[s]); w, h = bs[m]
    c = candidates(g, ch["kernel"]); bx = merge([b for b, _ in c], [v for _, v in c], ch["threshold"])
    def labels(f, W, H): return [tuple(map(float, l.split()[1:])) for l in Path(f).read_text().strip().splitlines()]
    def first(img_dir, mach, sticks3):
        for f in sorted(Path(img_dir).glob("*.png")):
            st = f.stem
            if st in P: ok = machine_of(P[st]) == mach and (st[4:12] <= "20200715") == sticks3
            else: r = meta[meta.image == st].iloc[0]; ok = r.machine == mach and (r.n_sticks == 3) == sticks3
            if ok: return f
    R5 = data_dir(2) / "train" / "images"; SN = SYN / "train" / "images"
    meta = meta_new = pd.read_csv(SYN / "meta.csv")
    def first_shape(mach):   # 새 형상이 들어간 첫 학습 이미지
        r = meta_new[(meta_new.split == "train") & (meta_new.machine == mach) & (meta_new.n_sticks == 3) & (meta_new.n_shapes > 0)].iloc[0]; return SN / f"{r.image}.png"
    panels = [(first(R5, m, True), data_dir(2) / "train" / "labels", "(v2) training image: real labeled image\n(machine marks erased) + human labels"),
              (first_shape(m), SYN / "train" / "labels", "(v3)-(v5) training image: Diffusion synthetic,\ndots + new shapes (first 500 / 2,000 / 5,000)")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 5.2))
    ys, xs = np.where(g < 180); y0, y1, x0, x1 = max(0, ys.min() - 10), ys.max() + 10, max(0, xs.min() - 10), xs.max() + 10
    ax[0].imshow(g[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
    for x, y in ans[s]["points"]: ax[0].add_patch(plt.Rectangle((x - w / 2 - x0, y - h / 2 - y0), w, h, fill=False, ec="lime", lw=1.3))
    for (px0, py0, px1, py1), v in bx: ax[0].add_patch(plt.Rectangle((px0 - x0, py0 - y0), px1 - px0, py1 - py0, fill=False, ec="red", lw=1.3))
    ax[0].set_title(f"(v1) no training: black-hat filter ({ch['kernel']} px, >= {ch['threshold']:g})\non a real test image (green = answer, red = found)", fontsize=10)
    for k, (f, ld, t) in enumerate(panels, 1):
        im = cv2.imread(str(f), 0); H, W = im.shape; ax[k].imshow(im, cmap="gray", vmin=0, vmax=255); ax[k].set_title(t, fontsize=10)
        for x, y, ww, hh in labels(ld / f"{f.stem}.txt", W, H): ax[k].add_patch(plt.Rectangle(((x - ww / 2) * W, (y - hh / 2) * H), ww * W, hh * H, fill=False, ec="red", lw=1.2))
    for a_ in ax: a_.axis("off")
    fig.suptitle("Fig. 10  Comparison models: what each model learns from (all from Machine 2, 3 sticks)", y=0.995)
    plt.tight_layout(); save_both(fig, "fig10_models.png")
    print("모델 그림:", s, [p_[0].stem for p_ in panels])


def fig_blackhat():
    """보고서 그림 10(fig11_blackhat.png): v1 필터의 black-hat 연산 예시와 파라미터 튜닝(레이블 500장, 표시 지움)"""
    from detect import filter_candidates as candidates, merge, center_in
    E = EVAL_DIR; ans = json.load(open(E / "test1_answers.json")); ch = json.load(open(model_dir(1) / "params.json")); k, th = ch["kernel"], ch["threshold"]
    per = pd.read_csv(eval_dir(1) / "test1_per_image.csv"); s = per[(per.machine == "2호기") & (per["fn"] == 0) & (per["fp"] == 2)].image.iloc[0]
    g = cv2.imread(str(E / "set" / "test1" / f"{s}.png"), 0); K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, K); bh = closed.astype(int) - g.astype(int)
    rs = cv2.GaussianBlur(cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, K).astype(np.float32), (0, 0), 0.8)   # 필터 반응값: black-hat 을 σ 0.8로 흐린 값(detect.filter_candidates 와 같음), 기준값과 비교하는 값
    x, y = [int(round(v)) for v in ans[s]["points"][0]]; d = 14; sl = (slice(y - d, y + d + 1), slice(x - d, x + d + 1))
    rd = float(rs[y - 2:y + 3, x - 2:x + 3].max())                                                       # 이물 위치의 반응값(주변 5×5 극댓값)
    fig = plt.figure(figsize=(16, 9)); gs = fig.add_gridspec(2, 4)
    a1 = fig.add_subplot(gs[0, 0]); a1.imshow(g[sl], cmap="gray", vmin=60, vmax=160, interpolation="nearest"); a1.set_title("(1) input (real test image)\ndark dot = defect", fontsize=10)
    a2 = fig.add_subplot(gs[0, 1]); a2.imshow(closed[sl], cmap="gray", vmin=60, vmax=160, interpolation="nearest"); a2.set_title(f"(2) closing with a {k}-px disk\n(max then min): small dark dot is filled", fontsize=10)
    a3 = fig.add_subplot(gs[0, 2]); im = a3.imshow(bh[sl], cmap="magma", vmin=0, vmax=50, interpolation="nearest")
    a3.set_title(f"(3) black-hat = (2) - (1): {bh[y, x]} at the dot\n(4) response = (3) smoothed (sigma 0.8): {rd:.1f}", fontsize=10)
    a3.text(d + 2, d - 2, f"{bh[y, x]}", color="white", fontsize=11, fontweight="bold"); fig.colorbar(im, ax=a3, fraction=0.046)
    a4 = fig.add_subplot(gs[0, 3]); xs = np.arange(-d, d + 1)
    a4.plot(xs, g[y, x - d:x + d + 1], "o-", ms=3, label="(1) input"); a4.plot(xs, closed[y, x - d:x + d + 1], "s-", ms=3, label="(2) closing")
    a4.bar(xs, bh[y, x - d:x + d + 1], color="tab:red", alpha=0.35, label="(3) black-hat"); a4.plot(xs, rs[y, x - d:x + d + 1], "^-", ms=3, color="tab:green", label="(4) response (compared with threshold)")
    a4.axhline(th, color="red", ls="--", lw=1, label=f"threshold {th:g}")
    a4.set_title("one row through the dot (gray level)", fontsize=10); a4.set_xlabel("pixel from the dot"); a4.legend(fontsize=8)
    for a_ in (a1, a2, a3): a_.axis("off")
    c = candidates(g, k); bx = merge([b for b, _ in c], [v for _, v in c], th)
    fps = [b for b, v in bx if not any(abs((b[0] + b[2]) / 2 - px) < 6 and abs((b[1] + b[3]) / 2 - py) < 6 for px, py in ans[s]["points"])]
    H_, W_ = g.shape; inner = [b for b in fps if min(b[0], b[1], W_ - b[2], H_ - b[3]) > 15]   # 이미지 끝이 아닌 봉지 쪽 오검출
    fp = (inner or fps)[0]
    fx, fy = int((fp[0] + fp[2]) / 2), int((fp[1] + fp[3]) / 2); fl = (slice(max(0, fy - d), fy + d + 1), slice(max(0, fx - d), fx + d + 1))
    a5 = fig.add_subplot(gs[1, 0]); a5.imshow(g[fl], cmap="gray", vmin=60, vmax=220, interpolation="nearest"); a5.set_title("false detection: input\n(edge of the bag's plastic flap)", fontsize=10)
    a6 = fig.add_subplot(gs[1, 1]); a6.imshow(bh[fl], cmap="magma", vmin=0, vmax=50, interpolation="nearest"); a6.set_title(f"false detection: black-hat = {bh[fy, fx]}, response = {rs[fy, fx]:.1f}\n(response >= {th:g}, so it is called a defect)", fontsize=10)
    for a_ in (a5, a6): a_.axis("off")
    a7 = fig.add_subplot(gs[1, 2:]); grid = pd.read_csv(RUNS / "v1_train" / "val_grid.csv")
    for kk in sorted(grid.kernel.unique()):
        sub = grid[grid.kernel == kk].sort_values("threshold"); a7.plot(sub.threshold, sub.f1, "o-", ms=3, label=f"filter {kk} px")
    a7.plot([th], [ch["val"]["f1"]], "r*", ms=16, label=f"chosen: {k} px, {th:g} (F1 {ch['val']['f1']:.3f})")
    a7.set_xlabel("threshold (on the response)"); a7.set_ylabel("F1 on the 500 labeled real images"); a7.legend(fontsize=9)
    a7.set_title("parameter tuning on the 500 labeled real images (machine marks erased, 1,147 defects)\n4 filter sizes x 29 thresholds = 116 combinations; test set 1 used once afterwards", fontsize=10)
    fig.suptitle("Fig. 11  v1 filter without learning: how black-hat finds a defect, and how its parameters were chosen", y=0.995)
    plt.tight_layout(); save_both(fig, "fig11_blackhat.png")
    print("예시 이미지", s, "이물 위치 black-hat", int(bh[y, x]), "반응값", round(rd, 1), "오검출 위치 black-hat", int(bh[fy, fx]), "반응값", round(float(rs[fy, fx]), 1))


def fig_test2_models():
    """보고서 그림 12(fig12_test2_models.png): 테스트셋 2(정답 없음)에서 v1~v5 비교 예시. 위 = 이물이 보이는 이미지, 가운데·아래 = 봉지가 잘린 7월 27일 이미지"""
    T = RUNS / "test2_compare"; B = json.load(open(T / "boxes.json")); per = pd.read_csv(T / "per_image.csv").set_index("image")
    ims = ["002_20200904_230047(1)", "002_20200727_111038(1)", "002_20200727_121541(9)"]
    cols = [("v1", "v1 filter"), ("v2", "v2 real 500"), ("v3", "v3 synth 500 + shapes"), ("v4", "v4 synth 2,000 + shapes"), ("v5", "v5 synth 5,000 + shapes\n(final)")]
    col = {"defect": "red", "recheck": "darkorange"}
    fig, ax = plt.subplots(len(ims), len(cols), figsize=(4.2 * len(cols), 3.9 * len(ims)))
    for i, s in enumerate(ims):
        g = cv2.imread(str(EVAL_DIR / "set" / "test2" / f"{s}.png"), 0)
        for j, (m, lab) in enumerate(cols):
            a_ = ax[i, j]; a_.imshow(g, cmap="gray", vmin=0, vmax=255)
            for d in B[s][m]:
                H, W = g.shape; c = col[d["level"]]
                x0, y0, x1, y1 = max(1, d["box"][0] - 3), max(1, d["box"][1] - 3), min(W - 2, d["box"][2] + 3), min(H - 2, d["box"][3] + 3)   # 이미지 안으로
                a_.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=c, lw=1.8))
                tx, ty, va = (min(x0, W - 30), y0 - 3, "bottom") if y0 > 15 else (min(x0, W - 30), y1 + 3, "top")
                a_.text(tx, ty, f"{d['score']:.0f}" if m == "v1" else f"{d['score']:.2f}", color=c, fontsize=9, fontweight="bold", va=va)
            dec = per.loc[s, f"{m}_decision"]; a_.set_title((lab + "\n" if i == 0 else "") + f"decision: {dec}", fontsize=10, color={"defect": "red", "recheck": "darkorange", "normal": "green"}[dec])
            a_.axis("off")
        ax[i, 0].text(-0.04, 0.5, ["visible defect (Sep 4)", "cut bag (Jul 27)", "cut bag (Jul 27)"][i], transform=ax[i, 0].transAxes, rotation=90, va="center", ha="right", fontsize=10)
    fig.suptitle("Fig. 12  Test set 2 (no answers): detections of the models (red = defect, orange = re-inspect; number = confidence or filter response)", y=0.995)
    plt.tight_layout(); save_both(fig, "fig12_test2_models.png")


def fig_shapes():
    """보고서 그림 5(fig13_shapes.png): 새 형상(곧은 선·휜 선·꺾인 선·작은 조각) 예시와, 새 형상을 섞은 합성 이미지(빨강 = 새 형상 레이블, 초록 = 점 패치 레이블)"""
    import synth.shapes as shp
    sig = json.load(open(MATERIAL / "shapes_500.json"))["blur_sigma"]; rng = np.random.default_rng(0)
    D = data_dir(5); meta = pd.read_csv(D / "meta.csv"); pick = meta[(meta.split == "train") & (meta.n_shapes > 0)].head(3)
    fig = plt.figure(figsize=(16, 9)); gs = fig.add_gridspec(2, 12)
    names = {"line": "straight line", "curve": "curved line", "bent": "bent line", "flake": "small flake"}
    for i, kind in enumerate(shp.KINDS):
        for j in range(3):
            pch, w, h = shp.draw_shape(kind, rng, sig, 0.5); r = pch.shape[0] // 2; a_ = fig.add_subplot(gs[0, i * 3 + j])
            a_.imshow(pch * 120, cmap="gray", vmin=0, vmax=130, interpolation="nearest"); a_.add_patch(plt.Rectangle((r - w / 2, r - h / 2), w, h, fill=False, ec="red", lw=1))
            a_.set_xlim(r - 20, r + 20); a_.set_ylim(r + 20, r - 20); a_.axis("off")
            if j == 1: a_.set_title(names[kind] + " (label box in red)", fontsize=10)
    for k, (_, row) in enumerate(pick.iterrows()):
        g = cv2.imread(str(D / "train" / "images" / f"{row.image}.png"), 0); H, W = g.shape; a_ = fig.add_subplot(gs[1, k * 4:(k + 1) * 4]); a_.imshow(g, cmap="gray", vmin=0, vmax=255)
        for (_, x, y, w, h), src in zip(np.loadtxt(D / "train" / "labels" / f"{row.image}.txt", ndmin=2), row.patch_src.split(";")):
            a_.add_patch(plt.Rectangle(((x - w / 2) * W, (y - h / 2) * H), w * W, h * H, fill=False, ec="red" if src.startswith("shape:") else "lime", lw=1.3))
        a_.set_title(f"synthetic image (Machine {row.machine[0]}): red = new shape, green = dot patch", fontsize=10); a_.axis("off")
    fig.suptitle(f"Fig. 13  New defect shapes drawn by program (blur sigma {sig} px measured from the 500 real dots) and synthetic images that include them", y=0.995)
    plt.tight_layout(); save_both(fig, "fig13_shapes.png")


FP_EXAMPLES = ["001_20200701_203610(7)", "001_20200707_083257(4)", "002_20200629_202541(5)"]   # 보고서 그림 15 예시: 3호기 장비 표시 2개, 3호기 1개(오검출 2개), 1호기 2개


def fig_fp():
    """보고서 그림 15(fig14_false_detections.png): 오검출 예시(analyze.py 의 경우 나누기). (a)~(f) 같은 봉지를 몇 초 간격으로 다시 찍은 사진 한 쌍: 왼쪽 = 장비가 3개 모두 표시, 오른쪽 = 장비가 덜 표시해 오검출로 셈.
    (g) 미검출 정답 바로 옆, (h) 이 사진에서만 검출. 초록 = 검사 장비 bounding box 위치(정답), 빨강 = YOLO α 이상, 노란 화살표 = 오검출"""
    import datetime as dt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    E, th, P, rows, ans, bs, merge, point_boxes, center_in = _eval_ctx(); fp = pd.read_csv(E / "test1_fp.csv"); T = lambda s: dt.datetime.strptime(s[4:19], "%Y%m%d_%H%M%S")

    def panel(a_, s, title, center=None, W_=100, H_=104, arrows=True):
        g = cv2.imread(str(E / "set" / "test1" / f"{s}.png"), 0); H, W = g.shape; m = machine_of(P[s]); A_ = ans[s]; bx = shown_boxes(rows[s]["pred"], rows[s]["conf"], th["alpha"], th["beta"])
        if center is None: center = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b, c in bx if c >= th["alpha"]] + A_["points"] + [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b, _ in A_["regions"]]).mean(0)
        x0, y0 = int(min(max(center[0] - W_ / 2, 0), W - W_)), int(min(max(center[1] - H_ / 2, 0), H - H_))
        a_.imshow(g[y0:y0 + H_, x0:x0 + W_], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        _draw(a_, bx, th, A_["points"], [(tuple(b), n) for b, n in A_["regions"]], [tuple(b) for b in A_["ignore"]], bs[m], x0, y0)
        for r in (fp[fp.image == s].itertuples() if arrows else []):
            yy = (r.y0 + r.y1) / 2 - y0; a_.annotate("", xy=(r.x0 - 4 - x0, yy), xytext=(r.x0 - 24 - x0, yy), arrowprops=dict(arrowstyle="-|>", color="yellow", lw=1.8))
        a_.set_xlim(-0.5, W_ - 0.5); a_.set_ylim(H_ - 0.5, -0.5); a_.set_xticks([]); a_.set_yticks([]); a_.set_title(title, fontsize=9)

    fig = plt.figure(figsize=(12, 7.2)); gs = fig.add_gridspec(3, 5, height_ratios=[0.07, 1, 1], width_ratios=[1, 1, 0.12, 1, 1])
    slots = [(1, 0), (1, 1), (1, 3), (1, 4), (2, 0), (2, 1), (2, 3), (2, 4)]; L = "abcdefgh"
    for k, s in enumerate(FP_EXAMPLES):
        r0 = fp[fp.image == s].iloc[0]; nb = r0.boxed_shot; assert r0.case == "boxed_in_other_shot", (s, r0.case)
        m = machine_of(P[s])[0]; sec = int(abs((T(nb) - T(s)).total_seconds()))
        n_m = lambda x: len(ans[x]["points"]) + sum(n for _, n in ans[x]["regions"]); n_y = lambda x: len(merge(rows[x]["pred"], rows[x]["conf"], th["alpha"]))
        panel(fig.add_subplot(gs[slots[2 * k]]), nb, f"({L[2 * k]}) Machine {m}, {T(nb):%m-%d %H:%M:%S}\nmachine boxes {n_m(nb)}, YOLO boxes {n_y(nb)}", arrows=False)
        panel(fig.add_subplot(gs[slots[2 * k + 1]]), s, f"({L[2 * k + 1]}) same bag, {sec} s {'earlier' if T(s) < T(nb) else 'later'}\nmachine boxes {n_m(s)}, YOLO boxes {n_y(s)}")
    g_ = fp[fp.case == "beside_missed"]; h_ = fp[fp.case == "this_shot_only"]
    if len(g_): panel(fig.add_subplot(gs[slots[6]]), g_.iloc[0].image, "(g) machine box region (dashed)\njust misses the dot")
    if len(h_): r = h_.iloc[0]; panel(fig.add_subplot(gs[slots[7]]), r.image, f"(h) faint spot (confidence {r.conf:.2f})\nfound only in this shot of the bag", center=((r.x0 + r.x1) / 2 + 20, (r.y0 + r.y1) / 2))
    lg = fig.add_subplot(gs[0, :]); lg.axis("off")
    lg.legend(handles=[Patch(fc="none", ec="lime", label="answer: where the inspection machine drew a box"), Patch(fc="none", ec="red", label=f"YOLO detection >= alpha {th['alpha']:.2f} (number = confidence)"),
                       Line2D([], [], color="yellow", marker=">", lw=2, label="YOLO box with no machine box = counted as false detection")], loc="center", ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle("Fig. 14  False detections on test set 1 (yellow arrows). In (a)-(f) the machine boxed the same dot in another shot of the same bag", y=0.995)
    plt.tight_layout(); save_both(fig, "fig14_false_detections.png", dpi=120)


def fig_fp_models():
    """보고서 그림 16(fig15_fp_models.png): v5는 오검출로 셌고 v2는 세지 않은 3개(compare.py 의 fp_spots.csv). 위 = v5, 아래 = v2. 초록 = 장비가 표시한 위치, 빨강 = 그 모델의 α 이상, 주황 = β~α"""
    import datetime as dt
    E, th, P, rows, ans, bs, merge, point_boxes, center_in = _eval_ctx(); T = lambda s: dt.datetime.strptime(s[4:19], "%Y%m%d_%H%M%S")
    sp = pd.read_csv(RUNS / "compare" / "fp_spots.csv"); order = ["boxed_in_other_shot", "found_in_all_shots", "this_shot_only", "beside_missed"]
    d = sp[~sp["v2_오검출"]].sort_values("case", key=lambda c: c.map(order.index), kind="stable").reset_index(drop=True); fp = pd.read_csv(E / "test1_fp.csv")
    EB = eval_dir(2); thB = json.load(open(EB / "thresholds.json")); rowsB = {r["image"]: r for r in json.load(open(EB / "test1_predictions.json"))}
    W_, H_ = 100, 104; fig, ax = plt.subplots(2, len(d), figsize=(3.3 * len(d), 7.4), squeeze=False)
    for k, r in d.iterrows():
        f0 = fp[(fp.image == r.image) & (abs((fp.x0 + fp.x1) / 2 - r.x) < 1) & (abs((fp.y0 + fp.y1) / 2 - r.y) < 1)].iloc[0]
        why = {"boxed_in_other_shot": lambda: f"machine boxed this dot in the shot\n{int(abs((T(f0.boxed_shot) - T(r.image)).total_seconds()))} s {'later' if T(f0.boxed_shot) > T(r.image) else 'earlier'} (same bag)",
               "found_in_all_shots": lambda: "v5 found this dot in all shots of the bag,\nthe machine never boxed it", "this_shot_only": lambda: "faint spot found only in this shot",
               "beside_missed": lambda: "machine box region just misses the dot"}[r.case]()
        g = cv2.imread(str(E / "set" / "test1" / f"{r.image}.png"), 0); H, W = g.shape; m = machine_of(P[r.image]); A_ = ans[r.image]
        x0, y0 = int(min(max(r.x + 20 - W_ / 2, 0), W - W_)), int(min(max(r.y - H_ / 2, 0), H - H_))
        for j, (lab, rr, t_, key) in enumerate([(NAMES[5][2], rows[r.image], th, "v5"), (NAMES[2][2], rowsB[r.image], thB, "v2")]):
            a_ = ax[j, k]; a_.imshow(g[y0:y0 + H_, x0:x0 + W_], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            _draw(a_, shown_boxes(rr["pred"], rr["conf"], t_["alpha"], t_["beta"]), t_, A_["points"], [(tuple(b), n) for b, n in A_["regions"]], [tuple(b) for b in A_["ignore"]], bs[m], x0, y0)
            a_.annotate("", xy=(r.x - 10 - x0, r.y - y0), xytext=(r.x - 30 - x0, r.y - y0), arrowprops=dict(arrowstyle="-|>", color="yellow", lw=1.8))
            c = r[f"{key}_그 위치 최고 confidence"]; res = "false detection" if r[f"{key}_오검출"] else ("not found" if c < t_["beta"] else f"below alpha {t_['alpha']:.2f}")
            a_.set_title((f"({'abc'[k]}) Machine {m[0]}, {T(r.image):%m-%d %H:%M:%S}\n{why}\n\n" if j == 0 else "") + f"{lab}: {c:.3f} -> {res}", fontsize=9)
            a_.set_xlim(-0.5, W_ - 0.5); a_.set_ylim(H_ - 0.5, -0.5); a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("Fig. 15  Spots counted as false detections for v5 (top) but not for v2 (bottom). green = machine box, red = >= alpha, orange = beta~alpha", y=0.995)
    plt.tight_layout(); save_both(fig, "fig15_fp_models.png", dpi=120)


def fig_missed():
    """보고서 그림 11(fig16_missed.png): v2·v4·v5가 놓친 이물 7개(모두 서로 다른 이미지에 하나씩). 짝짓기는 detect.score_image 와 같다.
    보고서용 fig16_missed.png(행 = 이물, 열 = 전체 이미지·원본·v2·v4·v5), 발표용 fig16_missed_wide.png(열 = 이물, 행 = 원본·v2·v4·v5).
    칸 위 글자: found = 검출, missed = 미검출, image: defect / recheck / normal = 그 이미지의 판정(불량 / 재검사 / 정상)"""
    from detect import _score
    from compare import WIRE
    from matplotlib.patches import Patch, Circle
    from matplotlib.lines import Line2D
    E, th, P, rows, ans, bs, merge, point_boxes, center_in = _eval_ctx()
    beside = set(pd.read_csv(E / "test1_fp.csv").query("case == 'beside_missed'").image)   # 정답 영역 바로 밖의 이물(analyze.fp_cases)

    def misses(ed):
        """그 모델이 놓친 정답 {(이미지, x, y): 종류}, 정답 위치마다 가장 높은 confidence, 이미지 판정"""
        t = json.load(open(ed / "thresholds.json")); per = pd.read_csv(ed / "test1_per_image.csv").set_index("image")
        rr = {r["image"]: r for r in json.load(open(ed / "test1_predictions.json"))}; out, conf_at = {}, {}
        for s, r in rr.items():
            a = ans[s]; gt = point_boxes(a["points"], bs[machine_of(P[s])]); regs = [(tuple(b), n) for b, n in a["regions"]]; ign = [tuple(b) for b in a["ignore"]]
            used, cap = set(), [n for _, n in regs]
            for p, c in merge(r["pred"], r["conf"], t["alpha"]):
                if any(center_in(p, b) for b in ign): continue
                best = max([(_score(p, g, False), j) for j, g in enumerate(gt) if j not in used], default=(0, -1))
                if best[0] > 0: used.add(best[1]); continue
                q = next((q for q, (b, _) in enumerate(regs) if center_in(p, b)), None)
                if q is not None and cap[q] > 0: cap[q] -= 1
            raw = list(zip(r["pred"], r["conf"]))
            for j, g in enumerate(gt):
                k = (s, round((g[0] + g[2]) / 2), round((g[1] + g[3]) / 2)); conf_at[k] = max([c for p, c in raw if _score(p, g, False) > 0], default=0.0)
                if j not in used: out[k] = "point"
            for (b, n), left in zip(regs, cap):
                k = (s, round((b[0] + b[2]) / 2), round((b[1] + b[3]) / 2)); conf_at[k] = max([c for p, c in raw if center_in(p, b)], default=0.0)
                if left > 0: out[k] = "region"
        return t, per, rr, out, conf_at

    M = [(lab, *misses(ed)) for lab, ed in [(NAMES[2][2], eval_dir(2)), (NAMES[4][2], eval_dir(4)), (NAMES[5][2], E)]]
    n_ans = lambda s: len(ans[s]["points"]) + sum(n for _, n in ans[s]["regions"]); n_miss = lambda k: sum(k in m[4] for m in M)
    keys = sorted(set().union(*[set(m[4]) for m in M]), key=lambda k: (n_ans(k[0]) != 3, k[0] == WIRE, -n_miss(k), k[0]))
    kind = lambda k: next(m[4][k] for m in M if k in m[4])
    desc = lambda k: ("dot just outside answer region" if k[0] in beside else "wire-shaped defect" if k[0] == WIRE else "dot inside answer region" if kind(k) == "region" else "dot")
    W_, H_ = 100, 104; L = "abcdefghij"

    def status(m, k):
        lab, t, per, rr, out, conf_at = m; dec = per.loc[k[0], "decision"]; c = conf_at[k]
        if k not in out: return f"found ({c:.2f})", "green"
        if k[0] in beside: return f"missed: box just outside\nthe answer region\n-> image: {dec}", "red"
        if c < t["beta"]: return f"missed: {c:.2f} < beta {t['beta']:.2f}\n-> image: {dec}", "red"
        return f"missed: {c:.2f} < alpha {t['alpha']:.2f}\n-> image: {dec}", "red"

    def tiles(k):
        s, x, y = k; m = machine_of(P[s]); g = cv2.imread(str(E / "set" / "test1" / f"{s}.png"), 0); Hh, Ww = g.shape; A_ = ans[s]
        x0, y0 = int(min(max(x - W_ / 2, 0), Ww - W_)), int(min(max(y - H_ / 2, 0), Hh - H_))
        o = cv2.cvtColor(cv2.imread(str(P[s])), cv2.COLOR_BGR2RGB)
        return s, x, y, m, g, A_, [(tuple(b), n) for b, n in A_["regions"]], [tuple(b) for b in A_["ignore"]], x0, y0, o

    def zoom(a_, img, k, x0, y0, A_, regs, ign, m, mm=None):
        a_.imshow(img[y0:y0 + H_, x0:x0 + W_], **({} if img.ndim == 3 else dict(cmap="gray", vmin=0, vmax=255)), interpolation="nearest")
        if mm is not None:
            _, t, _, rr, _, _ = mm; _draw(a_, shown_boxes(rr[k[0]]["pred"], rr[k[0]]["conf"], t["alpha"], t["beta"]), t, A_["points"], regs, ign, bs[m], x0, y0)
        a_.set_xlim(-0.5, W_ - 0.5); a_.set_ylim(H_ - 0.5, -0.5); a_.set_xticks([]); a_.set_yticks([])

    legend = [Patch(fc="none", ec="lime", label="answer (machine box)"), Patch(fc="none", ec="lime", ls="--", label="answer region (large machine mark)"),
              Patch(fc="none", ec="red", label="YOLO >= alpha (number = confidence)"), Patch(fc="none", ec="orange", label="YOLO beta~alpha (re-inspect)")]
    # 보고서용: 행 = 이물
    nr = len(keys); fig = plt.figure(figsize=(12, 2.05 * nr + 1.2)); gs = fig.add_gridspec(nr + 2, 5, height_ratios=[0.22, 0.2] + [1] * nr, width_ratios=[1.25, 1, 1, 1, 1])
    for j, h in enumerate(["whole image (machine marks erased)", "original image\n(machine marks)"] + [f"{m[0]}\nalpha {m[1]['alpha']:.2f}" for m in M]):
        a_ = fig.add_subplot(gs[1, j]); a_.axis("off"); a_.text(0.5, 0.0, h, ha="center", va="bottom", fontsize=9, fontweight="bold", transform=a_.transAxes)
    for i, k in enumerate(keys):
        s, x, y, m, g, A_, regs, ign, x0, y0, o = tiles(k)
        xs = [p_[0] for p_ in A_["points"]] + [(b[0] + b[2]) / 2 for b, _ in regs]; ys = [p_[1] for p_ in A_["points"]] + [(b[1] + b[3]) / 2 for b, _ in regs]
        cx0, cx1 = int(max(min(xs) - 110, 0)), int(min(max(xs) + 110, g.shape[1])); cy0, cy1 = int(max(min(ys) - 95, 0)), int(min(max(ys) + 95, g.shape[0]))
        a0 = fig.add_subplot(gs[i + 2, 0]); a0.imshow(g[cy0:cy1, cx0:cx1], cmap="gray", vmin=0, vmax=255); _draw(a0, [], th, A_["points"], regs, ign, bs[m], cx0, cy0)
        a0.add_patch(Circle((x - cx0, y - cy0), 16, fill=False, ec="yellow", lw=2)); a0.set_xticks([]); a0.set_yticks([])
        a0.set_title(f"({L[i]}) Machine {m[0]}, {n_ans(s)} defect{'s' if n_ans(s) > 1 else ''} in image\nyellow circle = {desc(k)}", fontsize=8.5)
        zoom(fig.add_subplot(gs[i + 2, 1]), o, k, x0, y0, A_, regs, ign, m)
        for j, mm in enumerate(M):
            a_ = fig.add_subplot(gs[i + 2, 2 + j]); zoom(a_, g, k, x0, y0, A_, regs, ign, m, mm); txt, col = status(mm, k); a_.set_title(txt, fontsize=8.5, color=col)
    lg = fig.add_subplot(gs[0, :]); lg.axis("off")
    lg.legend(handles=legend + [Line2D([], [], color="yellow", marker="o", ls="", mfc="none", ms=9, mew=2, label="the missed defect")], loc="center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"Fig. 16  Defects missed by v2, v4 or v5 on test set 1: {nr} defects, each in a different image (image: defect / recheck / normal = image decision)", y=0.997, fontsize=10)
    plt.tight_layout(); save_both(fig, "fig16_missed.png", dpi=110)
    # 발표용: 열 = 이물, 행 = 원본·v2·v4·v5. 칸 위 = 검출/미검출(한 줄), 칸 안 아래 = 미검출 이미지의 판정
    fig = plt.figure(figsize=(13, 5.9)); gs = fig.add_gridspec(6, nr + 1, height_ratios=[0.12, 0.42] + [1] * 4, width_ratios=[0.7] + [1] * nr)
    for r_, h in enumerate(["original image\n(machine marks)"] + [f"{m[0].replace(' (final)', chr(10) + '(final)')}\nalpha {m[1]['alpha']:.2f}" for m in M]):
        a_ = fig.add_subplot(gs[2 + r_, 0]); a_.axis("off"); a_.text(1.0, 0.5, h, ha="right", va="center", fontsize=9, fontweight="bold", transform=a_.transAxes)
    for i, k in enumerate(keys):
        s, x, y, m, g, A_, regs, ign, x0, y0, o = tiles(k)
        hd = fig.add_subplot(gs[1, i + 1]); hd.axis("off")
        hd.text(0.5, 0.0, f"({L[i]}) Machine {m[0]}, {n_ans(s)} defect{'s' if n_ans(s) > 1 else ''}\n{desc(k).replace(' answer', chr(10) + 'answer')}", ha="center", va="bottom", fontsize=8.5, transform=hd.transAxes)
        zoom(fig.add_subplot(gs[2, i + 1]), o, k, x0, y0, A_, regs, ign, m)
        for j, mm in enumerate(M):
            a_ = fig.add_subplot(gs[3 + j, i + 1]); zoom(a_, g, k, x0, y0, A_, regs, ign, m, mm)
            lab, t_, per, rr, out, conf_at = mm; c = conf_at[k]
            if k not in out: a_.set_title(f"found {c:.2f}", fontsize=8.5, color="green", pad=2); continue
            a_.set_title("missed: box outside region" if s in beside else f"missed {c:.2f} < {'beta' if c < t_['beta'] else 'alpha'}", fontsize=8.5, color="red", pad=2)
            a_.text(0.5, 0.03, f"image: {per.loc[s, 'decision']}", ha="center", va="bottom", fontsize=8, color="white", transform=a_.transAxes, bbox=dict(fc="black", alpha=0.65, pad=1.5, lw=0))
    lg = fig.add_subplot(gs[0, :]); lg.axis("off"); lg.legend(handles=legend, loc="center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("Fig. 16 (slide)  Defects missed by v2, v4 or v5 on test set 1", y=0.998, fontsize=10)
    plt.tight_layout(h_pad=0.4); save_both(fig, "fig16_missed_wide.png", dpi=110)


if __name__ == "__main__":
    {"data": fig_data, "testprep": fig_testprep, "gen": fig_gen, "val": fig_val, "test": fig_test, "errors": fig_errors, "compare": fig_compare, "models": fig_models, "blackhat": fig_blackhat, "test2models": fig_test2_models, "shapes": fig_shapes,
     "fp": fig_fp, "fpmodels": fig_fp_models, "missed": fig_missed}[sys.argv[1]]()
