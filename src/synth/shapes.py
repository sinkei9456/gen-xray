"""새 형상 그리기: 점이 아닌 이물(곧은 선·휜 선·꺾인 선·작은 조각)을 프로그램으로 그려 투과율 패치로 만든다(합성 이미지에 점 패치 대신 넣음).
- 크기: 선은 굵기 1~2픽셀·길이 6~30픽셀, 조각은 3~8픽셀. 방향은 무작위. 크기는 테스트 이미지를 재지 않고 정했다.
- 어두운 정도: 같은 장비의 점 패치 하나의 가장 어두운 투과율. 흐림 σ: develop/measure_shapes.py 가 실제 점 이물에서 잰 값(material_500/shapes_500.json).
- 레이블 상자: 형상을 감싸는 상자(가장 어두운 곳의 20% 이상 어두운 픽셀) + 사방 3픽셀.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, cv2

KINDS = ("line", "curve", "bent", "flake")   # 곧은 선, 휜 선, 꺾인 선, 작은 조각
S = 8          # 8배로 크게 그린 뒤 줄여 가장자리를 부드럽게
N = 45         # 형상을 그리는 정사각형 이미지의 한 변(픽셀)
MARGIN = 3     # 레이블 상자 여유(사방)


def _disc_ratios(D, sig):
    c = np.zeros((21 * S, 21 * S), np.float32); cv2.circle(c, (21 * S // 2, 21 * S // 2), int(round(D * S / 2)), 1.0, -1, lineType=cv2.LINE_AA)
    c = cv2.resize(c, (21, 21), interpolation=cv2.INTER_AREA)
    if sig > 0: c = cv2.GaussianBlur(c, (0, 0), sig)
    m = 10; v = c[m, m]
    return np.array([(c[m - 1, m] + c[m + 1, m] + c[m, m - 1] + c[m, m + 1]) / 4 / v, (c[m - 1, m - 1] + c[m + 1, m + 1] + c[m - 1, m + 1] + c[m + 1, m - 1]) / 4 / v,
                     (c[m - 2, m] + c[m + 2, m] + c[m, m - 2] + c[m, m + 2]) / 4 / v])


def load_blur(material):
    """material_500/shapes_500.json 의 흐림 σ"""
    return json.load(open(Path(material) / "shapes_500.json"))["blur_sigma"]


def _polyline(kind, rng):
    """→ 점 목록(픽셀 단위, 중심 근처), 굵기"""
    if kind == "flake":
        D = rng.uniform(3, 8); k = int(rng.integers(5, 9)); ang = np.sort(rng.uniform(0, 2 * np.pi, k)); rad = rng.uniform(0.5, 1.0, k) * D / 2
        return np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1), None
    L = rng.uniform(6, 30); w = rng.uniform(1, 2)
    if kind == "line": pts = np.array([[-L / 2, 0.0], [L / 2, 0.0]])
    elif kind == "curve":   # 한쪽으로 완만하게 휜 선(2차 베지어)
        c = rng.choice([-1, 1]) * rng.uniform(0.15, 0.35) * L; t = np.linspace(0, 1, 40)[:, None]
        p0, p1, p2 = np.array([-L / 2, 0.0]), np.array([0.0, 2 * c]), np.array([L / 2, 0.0]); pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    else:                   # 한 번 꺾인 선
        t = rng.uniform(0.3, 0.7); turn = np.deg2rad(rng.uniform(30, 120)) * rng.choice([-1, 1])
        a = np.array([0.0, 0.0]); b = a + [t * L, 0.0]; c = b + (1 - t) * L * np.array([np.cos(turn), np.sin(turn)]); pts = np.stack([a, b, c])
    th = rng.uniform(0, np.pi); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); pts = pts @ R.T
    return pts - (pts.min(0) + pts.max(0)) / 2, w


def draw_shape(kind, rng, sigma, tmin):
    """→ (투과율 패치(정사각형, 홀수 크기, 가운데 = 레이블 상자 가운데), 레이블 폭, 레이블 높이)"""
    pts, w = _polyline(kind, rng); cv = np.zeros((N * S, N * S), np.float32); P = np.round((pts + N / 2) * S).astype(np.int32)
    if kind == "flake": cv2.fillPoly(cv, [P], 1.0, lineType=cv2.LINE_AA)
    else: cv2.polylines(cv, [P], False, 1.0, thickness=max(1, int(round(w * S))), lineType=cv2.LINE_AA)
    c = np.clip(cv2.resize(cv, (N, N), interpolation=cv2.INTER_AREA), 0, 1)
    if sigma > 0: c = cv2.GaussianBlur(c, (0, 0), sigma)
    d = c / max(c.max(), 1e-6) * (1 - tmin)                       # 가장 어두운 곳 = 점 패치의 가장 어두운 투과율
    ys, xs = np.where(d >= 0.2 * d.max()); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2; nz_y, nz_x = np.where(d > 0.01 * d.max())
    r = int(max(abs(nz_y - cy).max(), abs(nz_x - cx).max())) + 1
    pad = np.pad(d, r); sub = pad[cy:cy + 2 * r + 1, cx:cx + 2 * r + 1]
    return (1 - sub).astype(np.float32), float(x1 - x0 + 1 + 2 * MARGIN), float(y1 - y0 + 1 + 2 * MARGIN)
