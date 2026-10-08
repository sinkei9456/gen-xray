"""v5(최종 모델)로 X-ray 이미지 추론 → 이물 상자·confidence·이미지 판정(불량 / 재검사 / 정상)과 결과 그림. 학습 없음.

입력: 이미지 파일 또는 폴더(.png .bmp .jpg). 대회 원본 BMP처럼 검사 장비 표시(색 픽셀)가 있으면 평가 때와 같은 방법(detect.erase_marks)으로 지운 뒤 추론한다.
처리(develop 평가와 같음):
  1) YOLO 입력 640×640: 사진의 긴 변을 640으로 맞추고(비율 유지) 남는 곳은 회색으로 채움. 상자는 원래 사진 좌표로 되돌림
  2) confidence α 이상 상자끼리 겹치거나 맞닿으면 바깥 테두리 하나로 합쳐 불량 상자(점수를 셀 때의 상자와 같음), β 이상 상자를 모두 같은 방법으로 합쳤을 때 α 이상이 하나도 없는 묶음은 재검사 상자(detect.shown_boxes)
  3) 이미지에서 가장 높은 confidence가 α 이상이면 불량, β 이상이면 재검사(사람이 다시 확인), 아니면 정상(이물이 없다고 판정)
     α·β는 v5 검증셋에서 정한 값(models/v5_synth5000/thresholds.json: α 0.40, β 0.05)
결과(--out 폴더):
  boxes.csv   상자별: image, x0, y0, x1, y1(원래 사진 픽셀), confidence, level(defect = α 이상 / recheck = β~α)
  images.csv  이미지별: image, source, width, height, erased_mark_pixels(지운 검사 장비 표시 픽셀 수), n_defect_boxes, n_recheck_boxes, max_confidence, decision(defect / recheck / normal)
  vis/        결과 그림(2배 확대, 빨강 = α 이상, 주황 = β~α, 숫자 = confidence), 파일 이름 앞에 판정
  inputs/     모델에 넣은 이미지(회색조 PNG, 검사 장비 표시를 지운 것). 이름이 같고 내용이 다른 사진은 이름 뒤에 __2 를 붙인다
사용: python release/predict.py --src ../data/test1/images --out ../runs/predict_test1
     python release/predict.py --src "../data/dataset/test1/yolov3/X선이물검출기(06.23_09.22)/2호기(2020.09.22)" --recursive --out ../runs/predict_raw
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, cv2, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (공통 함수)
from paths import model_dir
from synth.render import load_raw
from detect import predict_all, shown_boxes, decide, draw, erase_marks, IMGSZ

EXTS = (".png", ".bmp", ".jpg", ".jpeg")


def model_input(path: Path):
    """→ (회색조 uint8 이미지, 지운 검사 장비 표시 픽셀 수). BMP 는 평가 때와 같은 읽기(load_raw)·지우기(erase_marks)"""
    if path.suffix.lower() == ".bmp":
        g, mk = load_raw(str(path))
    else:
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is None: raise ValueError(f"읽을 수 없음: {path}")
        if im.ndim == 2: return im.astype(np.uint8), 0
        im = im[..., :3].astype(np.float32); g = im.mean(-1); mk = (im.max(-1) - im.min(-1)) > 0   # 색 픽셀 = 검사 장비 표시
    return (erase_marks(g, mk) if mk.any() else g.astype(np.uint8)), int(mk.sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True, help="이미지 파일 또는 폴더"); ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=str(model_dir(5) / "best.pt")); ap.add_argument("--thresholds", default=str(model_dir(5) / "thresholds.json"))
    ap.add_argument("--recursive", action="store_true", help="폴더 아래 하위 폴더까지"); ap.add_argument("--no-vis", action="store_true", help="결과 그림 저장 안 함")
    a = ap.parse_args(); src, out = Path(a.src), Path(a.out)
    files = [src] if src.is_file() else sorted(p for p in (src.rglob("*") if a.recursive else src.glob("*")) if p.suffix.lower() in EXTS)
    if not files: sys.exit(f"이미지 없음: {src}")
    th = json.load(open(a.thresholds)); A, B = th["alpha"], th["beta"]
    inp = out / "inputs"; inp.mkdir(parents=True, exist_ok=True); source, n_marks, seen = {}, {}, {}
    for f in files:   # 이름과 내용이 모두 같은 사진(대회 데이터의 이웃 날짜 폴더 중복)은 한 번만, 이름만 같고 내용이 다르면 이름 뒤에 __2 … 를 붙여 따로
        g, nm = model_input(f); h = hashlib.md5(g.tobytes()).hexdigest()
        if h in seen.get(f.stem, set()): continue
        seen.setdefault(f.stem, set()).add(h); name = f.stem if len(seen[f.stem]) == 1 else f"{f.stem}__{len(seen[f.stem])}"
        cv2.imwrite(str(inp / f"{name}.png"), g); source[name] = str(f); n_marks[name] = nm
    import random, torch; random.seed(0); np.random.seed(0); torch.manual_seed(0)
    from ultralytics import YOLO
    rows = predict_all(YOLO(a.weights), inp, IMGSZ)   # 이름 순서로 32장씩, 640×640
    boxes, imgs = [], []
    if not a.no_vis: (out / "vis").mkdir(exist_ok=True)
    for r in rows:
        s = r["image"]; bx = shown_boxes(r["pred"], r["conf"], A, B); dec = decide(bx, A, B)
        for (x0, y0, x1, y1), c in bx:
            boxes.append(dict(image=s, x0=round(x0, 1), y0=round(y0, 1), x1=round(x1, 1), y1=round(y1, 1), confidence=round(c, 4), level="defect" if c >= A else "recheck"))
        imgs.append(dict(image=s, source=source[s], width=r["W"], height=r["H"], erased_mark_pixels=n_marks[s], n_defect_boxes=sum(c >= A for _, c in bx),
                         n_recheck_boxes=sum(c < A for _, c in bx), max_confidence=round(max([c for _, c in bx], default=0.0), 4), decision=dec))
        if not a.no_vis: cv2.imwrite(str(out / "vis" / f"{dec}_{s}.png"), draw(cv2.imread(str(inp / f"{s}.png"), 0), bx, A, B))
    pd.DataFrame(boxes, columns=["image", "x0", "y0", "x1", "y1", "confidence", "level"]).to_csv(out / "boxes.csv", index=False)
    df = pd.DataFrame(imgs); df.to_csv(out / "images.csv", index=False)
    print(f"{len(df)}장, α {A} β {B} → 판정", df.decision.value_counts().to_dict(), "| 결과:", out)


if __name__ == "__main__":
    main()
