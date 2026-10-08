"""대회 이미지 경로와 나누기에 쓰는 함수(develop·release 공통).
- all_paths(): 대회 X-ray 이미지 전체(이름 → 경로, 이웃 날짜 폴더의 같은 이름 중복은 하나로). 대회 원본은 data/dataset/ 에 있다(제출본에 들어 있음).
- test_stems(): 사람 레이블이 있는 500장의 이름(이름은 예전 그대로 test 지만, 지금은 학습 재료로 쓰고 평가에는 쓰지 않는다)
- machine_of(): 경로 → 장비(1호기·2호기·3호기), test_session_siblings(): 500장과 같은 연속 촬영 묶음(같은 장비·같은 날, 앞 사진과 60초 안)의 다른 사진
  (테스트셋 1에서 뺀다, detect.build_sets). val_stems·bank_stems·build_val_set 은 이전 방법에서 쓰던 함수로 지금 결과에는 쓰지 않는다.
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np, cv2
from scipy import ndimage
from synth.render import RAW_ROOT, DATA_ROOT, load_raw, marker_boxes
LABEL_DIR = DATA_ROOT / "dataset" / "라벨링 6종 세트" / "labels"
BAD = {"002_20200713_202507(0)"}   # 이물 점 3개가 보이는데 검사 장비가 bounding box를 그리지 않은 사진. 이전 방법의 나누기(_pool)에서만 뺐고, 지금은 테스트셋 2에 들어간다

def all_paths():
    paths = {}
    for p in sorted(glob.glob(str(RAW_ROOT / "*" / "*" / "*.bmp"))): paths.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    return paths

def machine_of(path: str) -> str: return path.split("/")[-3][:3]
def test_stems(): return sorted(os.path.splitext(os.path.basename(t))[0] for t in glob.glob(str(LABEL_DIR / "*.txt")))
def _ts(s): return int(s[13:15]) * 3600 + int(s[15:17]) * 60 + int(s[17:19])

def test_session_siblings(paths):
    """테스트 사진과 같은 연속 촬영 묶음(같은 장비·같은 날, 앞 사진과 60초 이내로 이어진 묶음)에 속한 다른 사진들.
    거의 같은 사진이라 테스트셋 1에서 뺀다(detect.build_sets, 33장)."""
    T = set(test_stems()); rows = sorted(((machine_of(p), s[4:12], _ts(s), s) for s, p in paths.items()))
    sess = {}; sid = -1; prev = None
    for m, d, t, s in rows:
        if prev is None or m != prev[0] or d != prev[1] or t - prev[2] > 60: sid += 1
        sess[s] = sid; prev = (m, d, t)
    tsess = {sess[s] for s in T if s in sess}
    return {s for s in paths if sess[s] in tsess and s not in T}

def _pool(paths):
    T = set(test_stems()); SIB = test_session_siblings(paths)
    return [s for s in paths if s not in T and s not in SIB and s not in BAD and not (s[4:12] == "20200727" and machine_of(paths[s]) == "1호기")]
def val_stems(paths=None):
    paths = paths or all_paths(); return [s for s in _pool(paths) if int(s[4:12]) % 2 == 1]
def bank_stems(paths=None):
    paths = paths or all_paths(); return [s for s in _pool(paths) if int(s[4:12]) % 2 == 0]

def build_val_set(out: Path, inpaint: bool = True, box_px: int = 8):
    """검증셋 사진(색 테두리 지움) + 사각형 정답 YOLO txt(class 0). 사각형이 하나도 없는 사진은 뺀다.
    정답 상자는 사각형 중심에 box_px 크기(기본 8px = 합성 학습 정답과 같은 크기). 사각형 자체(18px)를 쓰면 모델 상자(8px)와 IoU 가 0.2 대라
    ultralytics 의 mAP 기반 best.pt 선택이 망가진다(2026-10-01 확인: best.pt 재현율 0.35, last.pt 0.99)."""
    (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(parents=True, exist_ok=True)
    paths = all_paths(); n = 0; nb = 0
    for s in val_stems(paths):
        g, mk = load_raw(paths[s]); bx = marker_boxes(mk)
        if not bx: continue
        H, W = g.shape
        if inpaint: g = cv2.inpaint(g.astype(np.uint8), ndimage.binary_dilation(mk, iterations=1).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA).astype(np.float32)
        cv2.imwrite(str(out / "images" / f"{s}.png"), g.astype(np.uint8))
        with open(out / "labels" / f"{s}.txt", "w") as f:
            for x0, y0, x1, y1 in bx:
                f.write(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {box_px / W:.6f} {box_px / H:.6f}\n"); nb += 1
        n += 1
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images\nval: images\nnames:\n  0: defect\n  1: stick\n")
    return out / "data.yaml", n, nb

if __name__ == "__main__":
    P = all_paths(); d, b, t = val_stems(P), bank_stems(P), test_stems()
    print("all", len(P), "test", len(t), "val", len(d), "bank", len(b), "excluded", len(P) - len(t) - len(d) - len(b))
    import collections
    for name, st in [("val", d), ("bank", b), ("test", t)]: print(name, collections.Counter(machine_of(P[s]) for s in st if s in P))
