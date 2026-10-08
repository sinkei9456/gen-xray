"""테스트셋 정답 만들기. 검사 장비가 그린 색 덩어리를 네 가지로 나눈다.
1) 폭·높이 24픽셀 이하: 사각형 하나 → 중심이 정답 위치(점 정답).
2) 24픽셀보다 크고 다른 덩어리를 감싸고 있음: 봉지 밖의 "T" 글자 표시. 이물이 아니므로 바깥 테두리와 안의 글자 모두 정답에서 뺀다.
3) 24픽셀보다 크고 안이 비어 있음: 사각형 여러 개가 겹쳐 바깥 테두리만 하나로 그려진 것. 테두리 안 전체를 정답 영역으로 보고,
   한 변이 36픽셀 이상이면 이물 2개, 아니면 1개로 센다(정답 있는 500장: 이런 덩어리 15개 모두 이 규칙과 개수가 맞음).
4) 폭이나 높이가 5픽셀 미만인 덩어리(다른 덩어리 안의 "T" 글자는 제외): 이물 표시가 아니므로 버린다(테스트셋 2의 왼쪽 끝 노란 세로선, 폭 2픽셀)."""
from __future__ import annotations
import collections, os, sys
import numpy as np, cv2
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splits import all_paths, test_stems, machine_of, LABEL_DIR
from synth.render import load_raw, marker_boxes


def answers(mask: np.ndarray, max_single=24):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3))); sls = ndimage.find_objects(lab)
    boxes = [(sl[1].start, sl[0].start, sl[1].stop, sl[0].stop) if sl is not None else None for sl in sls]
    big = [i for i, b in enumerate(boxes) if b and (b[2] - b[0] > max_single or b[3] - b[1] > max_single)]
    icon, enclosed = set(), set()
    for i in big:
        x0, y0, x1, y1 = boxes[i]
        inner = [j for j, b in enumerate(boxes) if b and j != i and b[0] > x0 and b[1] > y0 and b[2] < x1 and b[3] < y1]
        if inner: icon.add(i); enclosed.update(inner)
    points, regions, ignore = [], [], []
    for i, b in enumerate(boxes):
        if b is None or (b[2] - b[0] < 5 or b[3] - b[1] < 5) and i not in enclosed: continue
        if i in icon or i in enclosed: ignore.append(b); continue
        if i in big: regions.append((b, 2 if max(b[2] - b[0], b[3] - b[1]) >= 36 else 1))
        else: points.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    return points, regions, ignore


def box_size_500():
    P = all_paths(); acc = collections.defaultdict(list)
    for s in test_stems():
        if s not in P: continue
        g, _ = load_raw(P[s]); H, W = g.shape
        for line in (LABEL_DIR / f"{s}.txt").read_text().strip().splitlines():
            _, _, _, w, h = map(float, line.split()); acc[machine_of(P[s])].append((w * W, h * H))
    return {m: tuple(np.median(np.array(v), 0).tolist()) for m, v in acc.items()}


def near_copies(P, T, cand):
    """500장 중 하나와 거의 같은 사진: 64×64로 줄였을 때 평균 밝기 차이 1 미만이고, 이물(색 사각형) 위치가 모두 2픽셀 안으로 같은 사진.
    이 데이터는 같은 시험용 제품을 여러 날 반복해서 찍어, 날짜가 달라도 놓인 위치까지 거의 같은 사진이 있다."""
    small = lambda s: cv2.resize(load_raw(P[s])[0], (64, 64), interpolation=cv2.INTER_AREA).reshape(-1)
    cen = lambda s: sorted(((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in marker_boxes(load_raw(P[s])[1]))
    out = set()
    for m in ("1호기", "2호기", "3호기"):
        a = [s for s in T if machine_of(P[s]) == m]; b = [s for s in cand if machine_of(P[s]) == m]
        if not a or not b: continue
        A = np.stack([small(s) for s in a]); B = np.stack([small(s) for s in b]); CA = {s: cen(s) for s in a}
        for i, s in enumerate(b):
            d = np.abs(A - B[i]).mean(1)
            for j in np.where(d < 1.0)[0]:
                ct, ca = cen(s), CA[a[j]]
                if len(ct) == len(ca) and ct and max(np.hypot(x - u, y - v) for (x, y), (u, v) in zip(ct, ca)) <= 2: out.add(s); break
    return out
