#!/usr/bin/env bash
# 학습 없이 제출한 모델(models/)로 v1~v5 평가와 비교를 다시 낸다(GB10 기준 약 7분). 대회 원본(X-ray BMP·사람 레이블)은 data/dataset/ 에 들어 있다(README 3절).
# v5 평가가 테스트셋 1·2(검사 장비 표시를 지운 이미지)를 만들고, v1 평가와 테스트셋 2 비교가 그것을 같이 쓴다.
set -euo pipefail
cd "$(dirname "$0")/.."          # src/
PY=${PY:-python}
reset() { $PY -u mem_reset.py "$1" ../data --min-avail-gb 40 || true; }
reset "before v5 평가"; $PY -u develop/v5_eval.py          # 검증셋으로 α·β → 테스트셋 1 정량 평가 → 테스트셋 2 판정, 결과 그림(output/)
reset "before 분석"; $PY -u develop/analyze.py              # v5 오류 분석, 합성·실제 비교
[ -d ../data/v2_real500/all ] || { reset "before v2 데이터"; $PY -u develop/prepare_real500.py; }
reset "before v1"; $PY -u develop/v1_train.py; $PY -u develop/v1_eval.py   # v1 필터: 크기·기준값 정하기(1분 미만) → 평가
reset "before v2 평가"; $PY -u develop/v2_eval.py
{ [ -d ../data/v3_synth500 ] && [ -d ../data/v4_synth2000 ]; } || $PY -u develop/make_subsets.py   # v3·v4 데이터(없을 때만) = 합성 5,000장의 앞 500·2,000장
for n in 3 4; do reset "before v$n 평가"; $PY -u develop/v${n}_eval.py; done
reset "before 비교"; $PY -u develop/compare.py; $PY -u develop/test2_compare.py   # → runs/compare/, runs/test2_compare/
reset "after"
echo EVAL_DONE
