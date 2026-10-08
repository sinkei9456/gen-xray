"""제출 폴더 만들기: 소스코드(src/), 모델(models/), 학습용 데이터(data/), 대회 원본 중 코드가 읽는 것(data/dataset/), 테스트데이터 예측결과(predictions/), README·설치 파일.
블라인드 규칙: 모델 파일·학습 기록 안의 서버 경로(계정 이름 포함)를 지운다. 가중치는 그대로라 추론 결과는 같다.
사용: python develop/export_submission.py --out submission   (그 뒤 zip)
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (공통 함수)
from paths import ROOT, DATA, MATERIAL, DIFFUSION, PRETRAINED, RUNS, OUTPUT, NAMES, data_dir, eval_dir, model_dir
from synth.render import RAW_ROOT, LABEL_DIR
from splits import all_paths, machine_of
from test_answers import box_size_500
from detect import merge, shown_boxes, point_boxes, score_image, center_in

SRC_FILES = ["paths.py", "detect.py", "splits.py", "test_answers.py", "mem_reset.py"]   # 공통 함수
SRC_DIRS = ["sd", "synth", "develop", "release"]
NOCACHE = shutil.ignore_patterns("*.cache", "__pycache__")   # YOLO 레이블 캐시(서버 경로가 들어 있음, 학습 때 다시 만들어짐)는 빼기


def export_predictions(E: Path, dst: Path):
    th = json.load(open(E / "thresholds.json")); A, B = th["alpha"], th["beta"]; P = all_paths(); bs = box_size_500()
    ans = json.load(open(E / "test1_answers.json")); (dst / "test1").mkdir(parents=True, exist_ok=True); (dst / "test2").mkdir(parents=True, exist_ok=True)
    rows = []; arows = []
    for r in json.load(open(E / "test1_predictions.json")):
        s = r["image"]; m = machine_of(P[s]); a = ans[s]; regs = [(tuple(b), n) for b, n in a["regions"]]; ign = [tuple(b) for b in a["ignore"]]
        gt = point_boxes(a["points"], bs[m]); bx_a = merge(r["pred"], r["conf"], A); _, _, _, tags = score_image(bx_a, gt, regs, ign)
        # α 이상: 평가에 쓴 상자 그대로(검출 hit / 오검출 fp / "T" 표시 안 ignored). β~α: 재검사 범위 상자
        for (p, c), t in zip(bx_a, tags):
            rows.append(dict(image=s, machine=m, x0=round(p[0], 1), y0=round(p[1], 1), x1=round(p[2], 1), y1=round(p[3], 1), confidence=round(c, 4), level="defect(>=alpha)", result_at_alpha=t))
        for p, c in shown_boxes(r["pred"], r["conf"], A, B):
            if c < A: rows.append(dict(image=s, machine=m, x0=round(p[0], 1), y0=round(p[1], 1), x1=round(p[2], 1), y1=round(p[3], 1), confidence=round(c, 4), level="recheck(beta~alpha)", result_at_alpha="-"))
        for (x, y), (w, h) in zip(a["points"], [bs[m]] * len(a["points"])):
            arows.append(dict(image=s, machine=m, type="point", x0=round(x - w / 2, 1), y0=round(y - h / 2, 1), x1=round(x + w / 2, 1), y1=round(y + h / 2, 1), n_defects=1))
        for b, n in regs: arows.append(dict(image=s, machine=m, type="region", x0=b[0], y0=b[1], x1=b[2], y1=b[3], n_defects=n))
        for b in ign: arows.append(dict(image=s, machine=m, type="ignore(T mark)", x0=b[0], y0=b[1], x1=b[2], y1=b[3], n_defects=0))
    pd.DataFrame(rows).to_csv(dst / "test1" / "test1_boxes.csv", index=False); pd.DataFrame(arows).to_csv(dst / "test1" / "test1_answers.csv", index=False)
    per = pd.read_csv(E / "test1_per_image.csv").rename(columns={"answers": "n_answers", "tp": "hit", "fp": "false_detection", "fn": "missed", "tp_std": "hit_standard", "fp_std": "false_detection_standard"})
    per.to_csv(dst / "test1" / "test1_per_image.csv", index=False); shutil.copy(E / "test1_result.csv", dst / "test1" / "test1_metrics.csv")
    rows = []
    for r in json.load(open(E / "test2_predictions.json")):
        for p, c in shown_boxes(r["pred"], r["conf"], A, B):
            rows.append(dict(image=r["image"], machine=machine_of(P[r["image"]]), x0=round(p[0], 1), y0=round(p[1], 1), x1=round(p[2], 1), y1=round(p[3], 1), confidence=round(c, 4),
                             level="defect(>=alpha)" if c >= A else "recheck(beta~alpha)"))
    pd.DataFrame(rows, columns=["image", "machine", "x0", "y0", "x1", "y1", "confidence", "level"]).to_csv(dst / "test2" / "test2_boxes.csv", index=False)
    shutil.copy(E / "test2_decisions.csv", dst / "test2" / "test2_per_image.csv"); shutil.copy(E / "thresholds.json", dst / "thresholds.json")


def scrub_model(d: Path):
    """학습 기록(args.yaml)과 모델 파일 안의 서버 경로를 <프로젝트 폴더>로 바꾼다"""
    import torch
    y = d / "args.yaml"; y.write_text(y.read_text(encoding="utf-8").replace(str(ROOT), "<프로젝트 폴더>"), encoding="utf-8")
    ck = torch.load(d / "best.pt", weights_only=False, map_location="cpu")
    fix = lambda v: v.replace(str(ROOT), "<프로젝트 폴더>") if isinstance(v, str) else v
    for k in ck.get("train_args", {}): ck["train_args"][k] = fix(ck["train_args"][k])   # 학습 설정의 모든 칸(model·data·project 등)
    for obj in (ck.get("model"), ck.get("ema")):          # 모델 객체 안에도 학습 설정과 시작 가중치 경로(pt_path)가 따로 저장됨
        if obj is None: continue
        if hasattr(obj, "pt_path"): obj.pt_path = fix(obj.pt_path)
        a_ = getattr(obj, "args", None)
        if isinstance(a_, dict):
            for k in a_: a_[k] = fix(a_[k])
        elif a_ is not None:
            for k, v in vars(a_).items(): setattr(a_, k, fix(v))
    torch.save(ck, d / "best.pt")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "submission")); ap.add_argument("--no-images", action="store_true", help="결과 그림(output/) 빼기")
    a = ap.parse_args(); out = Path(a.out)
    if out.exists(): shutil.rmtree(out)
    # 소스코드: 공통 함수 + develop(재현용) + release(사용용)
    (out / "src").mkdir(parents=True)
    for f in SRC_FILES: shutil.copy(ROOT / "src" / f, out / "src" / f)
    for d in SRC_DIRS: shutil.copytree(ROOT / "src" / d, out / "src" / d, ignore=NOCACHE)
    for f in ("requirements.txt", "requirements-lock.txt", "environment.yml"): shutil.copy(ROOT / f, out / f)
    # 모델: Diffusion, v1 필터 값, v2~v5 YOLO26s(+ 검증셋에서 정한 α·β), 학습 시작 가중치
    M = out / "models"; shutil.copytree(DIFFUSION, M / "diffusion"); shutil.copy(PRETRAINED, M / PRETRAINED.name)
    shutil.copytree(model_dir(1), M / NAMES[1][0])
    for n in (2, 3, 4, 5):
        d = M / NAMES[n][0]; d.mkdir(parents=True)
        for f in ("best.pt", "args.yaml", "results.csv", "thresholds.json"): shutil.copy(model_dir(n) / f, d / f)
        scrub_model(d)
    # 학습용 데이터: 학습 재료, v2(표시 지운 실제 500장), v5(합성 5,000장). v3·v4는 v5의 앞부분이라 develop/make_subsets.py 로 만든다
    D = out / "data"; shutil.copytree(MATERIAL, D / MATERIAL.name, ignore=NOCACHE)
    for n in (2, 5): shutil.copytree(data_dir(n), D / NAMES[n][0], ignore=NOCACHE)
    # 대회 원본 중 코드가 읽는 두 폴더(X-ray BMP 2,809장, 사람 레이블 500개). 경로는 대회 zip 안의 dataset/ 폴더를 data/dataset/ 에 둔 것과 같다 → 받은 사람이 원본을 따로 붙이지 않아도 됨
    for d in (RAW_ROOT, LABEL_DIR): shutil.copytree(d, D / d.relative_to(DATA))
    # 테스트셋(검사 장비 표시를 지운 이미지, v5 평가 때 만든 것)
    E = eval_dir(5)
    shutil.copytree(E / "set" / "test1", D / "test1" / "images"); shutil.copytree(E / "set" / "test2", D / "test2" / "images"); shutil.copy(E / "set_counts.json", D / "test_set_counts.json")
    # 테스트데이터 예측결과(v5)와 비교·분석 결과
    R = out / "predictions"; export_predictions(E, R)
    C = R / "comparison"; C.mkdir(parents=True, exist_ok=True)   # v1~v5 비교
    for f in ("compare.csv", "fp_spots.csv", "fp_by_case.csv"): shutil.copy(RUNS / "compare" / f, C / f)
    shutil.copy(model_dir(1) / "params.json", C / "v1_params.json"); shutil.copy(RUNS / "v1_train" / "val_grid.csv", C / "v1_val_grid.csv")
    for f in ("test1_result.csv", "test1_per_image.csv"): shutil.copy(eval_dir(1) / f, C / f"v1_{f}")
    for n in (2, 3, 4):
        for f in ("thresholds.json", "val_thresholds_merged.csv", "test1_result.csv", "test1_per_image.csv"): shutil.copy(eval_dir(n) / f, C / f"v{n}_{f}")
    for f in ("summary.csv", "per_image.csv", "boxes.json"): shutil.copy(RUNS / "test2_compare" / f, C / f"test2_{f}")   # 테스트셋 2 판정 비교
    (R / "analysis").mkdir(parents=True, exist_ok=True)   # v5 오류 분석(analyze.py) 결과
    for f in ("analysis.json", "test1_fn.csv", "test1_fp.csv", "synthetic_vs_real_stats.csv", "test2_decisions_kind.csv", "val_thresholds_merged.csv"):
        if (E / f).exists(): shutil.copy(E / f, R / "analysis" / f)
    if not a.no_images:
        shutil.copytree(OUTPUT / "test1" / "ok", R / "test1" / "images_ok"); shutil.copytree(OUTPUT / "test1" / "ng", R / "test1" / "images_ng")
        shutil.copytree(OUTPUT / "test2", R / "test2" / "images")
    shutil.copy(ROOT / "doc" / "submission_README.md", out / "README.md")
    print("done:", out)


if __name__ == "__main__":
    main()
