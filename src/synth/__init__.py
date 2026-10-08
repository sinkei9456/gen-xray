"""synth — 지시어 → 배치도 → 합성 X-ray 이미지 + 레이블. 레이블은 생성 모델이 아니라 배치도와 이물을 넣은 위치에서 나온다.

layout.py: 지시어 해석(parse_prompt)과 배치도, render.py: 장비별 설정값·대회 이미지 읽기(load_raw)·검사 장비 표시 찾기·배치도의 봉지·막대 위치 만들기(procedural_background),
shapes.py: 새 형상(선·조각) 그리기, compose.py: 합성 이미지 한 장 만들기(develop/make_synth.py, release/generate.py 가 씀).
generate.py·Renderer: 이전 방법(절차적 생성기)으로 지금 결과에는 쓰지 않는다.
"""
from .layout import Spec, Defect, parse_prompt, spec_from_dict
from .render import Renderer
from .generate import generate

__all__ = ["Spec", "Defect", "parse_prompt", "spec_from_dict", "Renderer", "generate"]
