"""v2(실제 500장) 학습 데이터: 레이블 있는 500장에서 검사 장비 표시(bounding box 등 색 픽셀)를 지운 이미지 + 사람 레이블.
- 지우기: 테스트셋과 같은 방법(색 픽셀과 바로 옆 1픽셀을 OpenCV 인페인팅 Telea 반경 3으로 메움). 표시 안쪽의 이물 점은 그대로 둔다.
- all/: 500장 전부(v1 필터의 크기·기준값 정하기용)
- train/ 400장, val/ 100장(v2 YOLO 학습용): 같은 연속 촬영 묶음(같은 장비·같은 날, 60초 안)이 갈라지지 않게 묶음 단위로 무작위(시드 0)로 나눈다.
사용: python develop/prepare_real500.py → data/v2_real500/
"""
from __future__ import annotations
import os, shutil, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (공통 함수)
from splits import all_paths, test_stems, machine_of, _ts, LABEL_DIR
from synth.render import load_raw
from detect import erase_marks
from paths import data_dir

OUT = data_dir(2)


def main(n_val=100, seed=0):
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(data_dir(2))); OUT = Path(ap.parse_args().out)
    P = all_paths(); T = sorted(s for s in test_stems() if s in P)
    if OUT.exists(): shutil.rmtree(OUT)
    for d in ("all", "train", "val"): (OUT / d / "images").mkdir(parents=True); (OUT / d / "labels").mkdir(parents=True)
    rows = sorted((machine_of(P[s]), s[4:12], _ts(s), s) for s in T); sess, sid, prev = {}, -1, None
    for m, d, t, s in rows:
        if prev is None or m != prev[0] or d != prev[1] or t - prev[2] > 60: sid += 1
        sess.setdefault(sid, []).append(s); prev = (m, d, t)
    order = list(sess); np.random.default_rng(seed).shuffle(order); val, n = set(), 0
    for k in order:                       # 묶음 단위로 담되 정확히 100장이 되게(넘치는 묶음은 건너뜀)
        if n + len(sess[k]) <= n_val: val.update(sess[k]); n += len(sess[k])
        if n == n_val: break
    assert n == n_val, n
    for s in T:
        g, mk = load_raw(P[s]); img = erase_marks(g, mk); dst = "val" if s in val else "train"
        for d in ("all", dst):
            cv2.imwrite(str(OUT / d / "images" / f"{s}.png"), img); shutil.copy(LABEL_DIR / f"{s}.txt", OUT / d / "labels" / f"{s}.txt")
    import collections
    for d in ("train", "val"):
        st = [p.stem for p in (OUT / d / "images").glob("*.png")]; nd = sum(len((OUT / d / "labels" / f"{x}.txt").read_text().strip().splitlines()) for x in st)
        print(d, len(st), "장, 이물", nd, "개", dict(collections.Counter(machine_of(P[x]) for x in st)))


if __name__ == "__main__":
    main()
