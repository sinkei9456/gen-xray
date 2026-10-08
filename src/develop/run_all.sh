#!/usr/bin/env bash
# 전체 재현(시드 고정): 학습 재료 → Diffusion 학습 → 새 형상 흐림 재기 → 합성 5,000장 → v5 학습·평가·오류 분석
#   → v2 데이터 → v1 크기·기준값 정하기·평가 → v2 학습·평가 → v3·v4 데이터 → v3·v4 학습·평가 → v1~v5 비교표 → 테스트셋 2 비교 → 보고서 그림
# 결과가 이미 있는 단계는 건너뛴다: 학습 재료·학습 데이터는 data/, 학습한 모델은 models/ 를 본다(YOLO 학습은 runs/v번호_train/train_done 도 본다). v1 크기·기준값 정하기와 평가·분석·비교·그림은 매번 다시 한다.
# 제출본에는 data/·models/ 가 모두 들어 있어 학습·생성 단계를 모두 건너뛴다. 처음부터 다시 만들려면 README 5-2절의 "지울 것"(data/·models/ 의 결과 폴더, runs/, output/ 등)을 지운 뒤 실행한다(data/dataset/·models/yolo26s.pt 는 남김)
#   (처음부터 GB10 기준 약 8~10시간). 단계 사이마다 메모리 정리(mem_reset.py).
# 대회 원본(X-ray BMP·사람 레이블)은 data/dataset/ 에 들어 있다(README 3절). Diffusion·YOLO 학습은 GPU 결정적 계산 모드로 돌린다(같은 기계·같은 버전에서 처음부터 끝까지 다시 돌려 가중치·결과가 모두 같음을 확인, 10월 7~8일).
set -euo pipefail
cd "$(dirname "$0")/.."          # src/
PY=${PY:-python}
reset() { $PY -u mem_reset.py "$1" ../data --min-avail-gb 40 || true; }
[ -f ../data/material_500/levels_500.json ] || { reset "before 재료"; $PY -u develop/prepare_material.py; }
[ -f ../models/diffusion/diffusion_pytorch_model.safetensors ] || { reset "before Diffusion 학습"; $PY -u develop/train_diffusion.py; }
[ -f ../data/material_500/shapes_500.json ] || { reset "before 흐림 재기"; $PY -u develop/measure_shapes.py; }
[ -f ../data/v5_synth5000/meta.csv ] || { reset "before 합성 5,000장"; $PY -u develop/make_synth.py; }
[ -f ../models/v5_synth5000/best.pt ] || { reset "before v5 학습"; $PY -u develop/v5_train.py; }
reset "before v5 평가"; $PY -u develop/v5_eval.py
reset "before 분석"; $PY -u develop/analyze.py
[ -d ../data/v2_real500/all ] || { reset "before v2 데이터"; $PY -u develop/prepare_real500.py; }
reset "before v1"; $PY -u develop/v1_train.py; $PY -u develop/v1_eval.py   # v1 필터 크기·기준값 정하기(1분 미만, 같은 값이 나옴, 그림이 튜닝 기록을 씀) → 평가
[ -f ../models/v2_real500/best.pt ] || { reset "before v2 학습"; $PY -u develop/v2_train.py; }
reset "before v2 평가"; $PY -u develop/v2_eval.py
{ [ -d ../data/v3_synth500 ] && [ -d ../data/v4_synth2000 ]; } || $PY -u develop/make_subsets.py
for m in v3_synth500 v4_synth2000; do
  n=${m:1:1}
  [ -f ../models/$m/best.pt ] || { reset "before v$n 학습"; $PY -u develop/v${n}_train.py; }
  reset "before v$n 평가"; $PY -u develop/v${n}_eval.py
done
reset "before 비교"; $PY -u develop/compare.py; $PY -u develop/test2_compare.py
reset "before 그림"; for f in data testprep gen compare val test errors shapes fp models blackhat test2models fpmodels missed; do $PY develop/make_figures.py $f; done
reset "after all"
echo ALL_DONE
