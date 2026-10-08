"""지시어 해석(parse_prompt)과 Spec(이미지 한 장의 설계도).

지금 결과(synth/compose.py)는 Spec 의 machine(장비), product(막대 수), defects 개수(이물 수), rotation_deg 만 쓴다.
이물 위치는 레이블 500장의 실제 위치 분포를, 이물 모양은 점 패치와 새 형상을 쓰므로 Defect 의 size·contrast·shape·along·zone 과
Spec 의 placement 는 이전 방법(절차적 생성기)용이다.
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field, asdict
from typing import Optional, List

ZONES = ("stick_in", "stick_edge", "packaging", "prod_edge")
MACHINES = ("1호기", "2호기", "3호기")
PRODUCTS = ("stick1", "stick3", "dense_bag", "empty")     # 1스틱 / 3스틱 / 스틱 없는 균일 봉지(7/27형) / 봉지 없음
PLACEMENTS = ("center", "left_cut", "right_cut", "top_cut", "off_frame")


@dataclass
class Defect:
    zone: str = "stick_in"          # ZONES 중 하나
    size: float = 1.0               # 기준 이물 대비 배율 (0.5 ≈ 2px, 1.8 ≈ 7px), 이전 방법용
    contrast: float = 1.0           # 어두워지는 양 배율 (1.0 = 기준 이물), 이전 방법용
    shape: str = "ball"             # ball | wire | flake  (이전 방법의 절차적 이물용)
    stick_index: Optional[int] = None   # 3스틱일 때 몇 번째 스틱(0,1,2). None=랜덤
    along: str = "random"           # 막대 축 위치: end | center | random (이전 방법용. 지금 합성은 레이블 500장의 실제 위치 분포를 쓴다)


@dataclass
class Spec:
    machine: str = "random"                  # 1호기/2호기/3호기/random → 장비별 크기·밝기·잡음 설정값
    product: str = "stick1"                  # PRODUCTS
    placement: str = "center"                # PLACEMENTS (봉지 잘림 등, 이전 방법용)
    background: str = "procedural"           # procedural | real | diffusion (diffusion = 배치도를 조건으로 학습한 Diffusion 모델, 지금 결과)
    defects: List[Defect] = field(default_factory=list)
    rotation_deg: Optional[float] = None     # None=랜덤 ±25°
    note: str = ""                           # 원문 지시어 등

    @property
    def n_sticks(self) -> int:
        return {"stick1": 1, "stick3": 3}.get(self.product, 0)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def spec_from_dict(d: dict) -> Spec:
    d = dict(d)
    d["defects"] = [Defect(**x) if isinstance(x, dict) else x for x in d.get("defects", [])]
    return Spec(**d)


# ---------------------------------------------------------------- 자연어 파서 (규칙 기반, 한/영)
_NUM = {"한": 1, "하나": 1, "일": 1, "1": 1, "두": 2, "둘": 2, "2": 2, "세": 3, "셋": 3, "3": 3, "네": 4, "넷": 4, "4": 4, "다섯": 5, "5": 5}


def _count(text: str, nouns: str) -> Optional[int]:
    """'막대 3개', '이물 두 개', 'three sticks' 등에서 개수 추출"""
    m = re.search(rf"({'|'.join(nouns.split('|'))})\s*(?:를|을|이|가)?\s*(\d+|한|하나|두|둘|세|셋|네|넷|다섯)\s*(?:개|장|point|points|pcs)?", text)
    if m:
        return _NUM.get(m.group(2))
    m = re.search(rf"(\d+|one|two|three|a single|single)\s+({'|'.join(nouns.split('|'))})", text, re.I)
    if m:
        w = m.group(1).lower(); return {"one": 1, "a single": 1, "single": 1, "two": 2, "three": 3}.get(w, int(w) if w.isdigit() else None)
    return None


def parse_prompt(text: str) -> Spec:
    """자연어 지시문 → Spec. 규칙 기반이라 오프라인·재현 가능.

    예)
      "봉지 하나에 검은 막대 3개, 그중 하나에 작은 이물 한 개"
      "막대가 없는 봉지"  /  "봉지가 화면 왼쪽에 잘린 이미지"  /  "빈 프레임"
      "2호기, 스틱 1개, 이물 2개는 스틱 가장자리에, 대비 약하게"
    """
    t = text.strip(); low = t.lower()
    spec = Spec(note=t)

    # 장비
    for m in MACHINES:
        if m in t or m.replace("호기", "") + "호" in t:
            spec.machine = m
    mm = re.search(r"machine\s*([123])", low)
    if mm: spec.machine = f"{mm.group(1)}호기"

    # 봉지·막대를 그리는 방법(spec.background: real = 실제 봉지 사진, diffusion = Diffusion 모델). 지금 결과는 compose.make_one 이 diffusion 으로 정함
    if re.search(r"실제\s*(배경|봉지)|real\s*(background|bag)", low): spec.background = "real"
    if re.search(r"diffusion|\bsd\b|생성\s*모델", low): spec.background = "diffusion"

    # 제품 / 스틱 수
    n_stick = _count(t, "막대|스틱|막대기|stick|sticks|bar|bars")
    if re.search(r"(막대|스틱|막대기|stick|bar)[^.。]*?(없|비어|안 보|missing|without|no\s)", low) or re.search(r"(no|without)\s+(stick|bar)", low):
        spec.product = "dense_bag"
    elif re.search(r"빈\s*(프레임|화면|이미지)|empty\s*(frame|image)|봉지\s*(가|도)?\s*없", low):
        spec.product = "empty"
    elif n_stick == 3: spec.product = "stick3"
    elif n_stick == 1: spec.product = "stick1"
    elif n_stick == 0: spec.product = "dense_bag"
    if re.search(r"균일|어두운 봉지|dense", low): spec.product = "dense_bag"

    # 봉지 배치(잘림 등, 이전 방법용)
    if re.search(r"왼쪽.*(잘|걸|벗어)|left.*(cut|crop|edge)", low): spec.placement = "left_cut"
    elif re.search(r"오른쪽.*(잘|걸|벗어)|right.*(cut|crop)", low): spec.placement = "right_cut"
    elif re.search(r"위쪽.*(잘|걸)|top.*(cut|crop)", low): spec.placement = "top_cut"
    elif re.search(r"화면\s*밖|프레임\s*밖|off[- ]?frame|out of frame", low): spec.placement = "off_frame"

    # 회전
    mr = re.search(r"(-?\d+)\s*(도|deg)", low)
    if mr: spec.rotation_deg = float(mr.group(1))

    # 이물
    n_def = _count(t, "이물|이물질|결함|defect|defects|foreign|contaminant|particle|particles")
    if re.search(r"이물[^.。]*?(없|no\s|without)|(no|without)\s+(defect|foreign)", low): n_def = 0
    if n_def is None:
        n_def = 1 if re.search(r"이물|결함|defect|foreign|particle", low) else 0
    if spec.product in ("dense_bag", "empty") and not re.search(r"포장|packaging", low):
        n_def = 0 if not re.search(r"이물", low) else n_def
    zone = "stick_in"
    if re.search(r"스틱\s*가장자리|막대\s*가장자리|stick\s*edge|edge of (the )?stick", low): zone = "stick_edge"
    elif re.search(r"포장(재)?\s*(위|에)|packaging|film|seal", low): zone = "packaging"
    elif re.search(r"(제품|봉지)\s*가장자리|bag\s*edge|product\s*edge|edge of (the )?(bag|product)", low): zone = "prod_edge"
    if spec.product in ("dense_bag", "empty") and zone in ("stick_in", "stick_edge"): zone = "packaging"
    size = 1.0
    if re.search(r"아주 작|매우 작|tiny", low): size = 0.5
    elif re.search(r"작은|작게|small", low): size = 0.7
    elif re.search(r"큰|크게|large|big", low): size = 1.5
    ms = re.search(r"(?:크기|size)\s*[=:]?\s*([\d.]+)", low)
    if ms: size = float(ms.group(1))
    contrast = 1.0
    if re.search(r"희미|약하게|약한|faint|low[- ]contrast|weak", low): contrast = 0.4
    elif re.search(r"뚜렷|강하게|진하게|strong|high[- ]contrast", low): contrast = 1.3
    mc = re.search(r"(?:대비|contrast)\s*[=:]?\s*([\d.]+)", low)
    if mc: contrast = float(mc.group(1))
    shape = "ball"
    if re.search(r"철사|와이어|wire|선형", low): shape = "wire"
    elif re.search(r"조각|파편|유리|flake|fragment", low): shape = "flake"
    stick_idx = None
    mi = re.search(r"(\d)\s*번째?\s*(막대|스틱)", t)
    if mi: stick_idx = int(mi.group(1)) - 1
    along = "random"
    if re.search(r"(스틱|막대)\s*(의)?\s*(끝|한쪽|끄트머리)|stick\s*end|end of (the )?stick", low): along = "end"
    elif re.search(r"(스틱|막대)\s*(의)?\s*(가운데|중앙|중간)|stick\s*center|middle of (the )?stick", low): along = "center"
    # 미리 정한 배치: "실제 배치|프로토콜" 같은 말이 있으면 막대마다 이물 1개, 막대 끝, 기본 크기·대비(이전 방법용)
    if re.search(r"실제\s*배치|프로토콜|protocol", low) and spec.n_sticks:
        spec.defects = [Defect(zone="stick_in", size=size, contrast=contrast, shape=shape, stick_index=i, along="end") for i in range(spec.n_sticks)]
        return spec
    spec.defects = [Defect(zone=zone, size=size, contrast=contrast, shape=shape, stick_index=stick_idx, along=along) for _ in range(n_def)]
    return spec
