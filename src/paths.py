"""폴더 위치와 모델 이름(develop·release 공통). 프로젝트 폴더와 제출 폴더의 구조가 같아 이 값 그대로 쓴다.

data/material_500/   Diffusion 학습·합성 재료(레이블 500장에서 만든 봉지·막대 이미지, 이물 패치, 장비별 설정값)
data/v2_real500/ … data/v5_synth5000/   모델별 학습셋·검증셋
models/diffusion/, models/v1_filter/ … models/v5_synth5000/   학습된 모델(제출본에 들어 있음)
runs/   학습·평가 결과(실행하면 생김), output/   v5 결과 그림
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, RUNS, MODELS, OUTPUT = ROOT / "data", ROOT / "runs", ROOT / "models", ROOT / "output"
MATERIAL = DATA / "material_500"            # prepare_material.py·measure_shapes.py 결과
DIFFUSION = MODELS / "diffusion"            # Diffusion 모델(EMA 가중치)
PRETRAINED = MODELS / "yolo26s.pt"          # YOLO 학습을 시작한 COCO 사전학습 가중치(Ultralytics 배포본)

# 모델 번호 → (폴더 이름, 문서 표기, 그림 표기). v1은 학습 없는 필터, v2~v5는 YOLO26s
NAMES = {1: ("v1_filter", "v1 학습 없는 필터", "v1 filter"),
         2: ("v2_real500", "v2 실제 500장", "v2 real500"),
         3: ("v3_synth500", "v3 합성 500장", "v3 synth500"),
         4: ("v4_synth2000", "v4 합성 2,000장", "v4 synth2000"),
         5: ("v5_synth5000", "v5 합성 5,000장(최종)", "v5 synth5000 (final)")}


def model_dir(n): return MODELS / NAMES[n][0]          # 예: models/v5_synth5000 (best.pt, args.yaml, results.csv, thresholds.json)
def data_dir(n): return DATA / NAMES[n][0]            # 예: data/v5_synth5000 (train/, val/, meta.csv)
def train_dir(n): return RUNS / f"v{n}_train"         # 학습 기록(Ultralytics: train/weights/best.pt)
def eval_dir(n): return RUNS / f"v{n}_eval"           # 평가 결과(thresholds.json, test1_*.csv, set/ = 표시 지운 테스트셋)
