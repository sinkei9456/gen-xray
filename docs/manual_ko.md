# Gen X-ray 사용 설명서(자세한 판)

[README.ko.md](../README.ko.md)(한국어 소개) | [README.md](../README.md)(English)

코드 주석의 "README N절"은 이 설명서의 절 번호다. 아래 경로는 모두 저장소 맨 위 기준이다.

제6회 K-인공지능 제조데이터 분석 경진대회 문제 ④. 대회 X-ray 이미지에는 검사 장비가 이물 위치에 bounding box를 그려 두어, 이 이미지로 바로 학습할 수 없다. 그래서 레이블이 있는 500장에서 봉지·막대 이미지와 이물 패치를 분리하고, Diffusion 모델로 합성 이미지 5,000장을 만들어 YOLO26s를 학습했다. 합성 이물은 500장에서 떼어 낸 점 패치(70%)와 프로그램으로 그린 새 형상(곧은 선·휜 선·꺾인 선·작은 조각, 30%)이다. 평가는 학습 재료와 겹치지 않는 실제 이미지로 했다. 비교를 위해 학습 없는 필터, 표시를 지운 실제 500장으로 학습한 모델, 합성 양을 바꾼 모델을 같은 테스트셋으로 평가했다(5-1절).

이 저장소는 대회 제출 zip과 코드가 같다. 다른 점은 데이터와 큰 파일이다.
- 대회 원본(KAMP 데이터)과 원본에서 만든 실제 이미지(학습 재료, 표시를 지운 500장, 테스트셋, 결과 그림)는 넣지 않았다. 원본은 KAMP에서 받아 넣고(3절), 나머지는 코드가 다시 만든다.
- Diffusion 모델 가중치, v2–v4 가중치, 사전학습 가중치, 합성 5,000장은 [Releases](https://github.com/sinkei9456/gen-xray/releases)에 있다(1절). v5 가중치는 저장소에 들어 있다.
- 코드 주석의 "제출본", "제출 폴더"는 대회 제출 zip을 말한다.

모델 이름은 보고서와 같다.

| 번호 | 모델 | 폴더 |
|---|---|---|
| v1 | 학습 없는 필터(black-hat 연산, 필터 크기·기준값만 레이블 500장에서 고름) | `models/v1_filter/` |
| v2 | 실제 500장: 레이블 500장에서 검사 장비 표시를 지운 실제 이미지로 학습한 YOLO26s | `models/v2_real500/` |
| v3 | 합성 500장: 합성 5,000장의 앞 500장으로 학습한 YOLO26s | `models/v3_synth500/` |
| v4 | 합성 2,000장: 합성 5,000장의 앞 2,000장으로 학습한 YOLO26s | `models/v4_synth2000/` |
| **v5** | **합성 5,000장(최종 모델)** | `models/v5_synth5000/` |

처음 한 번 준비(2·3절):

```bash
conda env create -f environment.yml          # 가상환경 venv 만들기(이 폴더 맨 위에서, 처음 한 번, 2절)
conda activate venv                          # 가상환경 켜기(새 터미널마다)
# KAMP 원본을 data/dataset/ 에 넣고(3절), Releases 파일을 받아 넣은 뒤(1절)
cd src
python develop/prepare_material.py           # 학습 재료 다시 만들기(1분 미만, 원본에서 같은 파일이 나옴)
python develop/measure_shapes.py             # 새 형상 흐림 재기
```

바로 해 볼 수 있는 것(학습 없음, 4절):

```bash
python release/generate.py --prompt "2호기, 스틱 3개, 이물 3개" --n 5 --out ../runs/generate   # Diffusion 모델로 합성 이미지와 레이블 만들기
python release/predict.py --src "../data/dataset/test1/yolov3/X선이물검출기(06.23_09.22)/2호기(2020.09.22)" --recursive --out ../runs/predict_raw   # v5로 이미지 판정(불량 / 재검사 / 정상)
bash develop/run_all.sh                                                                        # 학습은 건너뛰고 v1–v5 평가·비교·그림 다시 내기(5-1절)
```

## 1. 폴더 구성

저장소에 들어 있는 것:

```
data/
  test_set_counts.json      전체 2,529장의 쓰임별 장수
docs/figures/               보고서 그림 일부
models/
  diffusion/                Diffusion 모델 설정(config.json)과 생성 설정(sample.json: 50단계, 빈 화면 출발, t = 999). 가중치는 Releases
  v1_filter/params.json     v1: 레이블 500장에서 고른 필터 크기(5픽셀)·기준값(16)
  v2_real500/               v2 YOLO26s: args.yaml(학습 설정), results.csv(epoch별 기록), thresholds.json(검증셋에서 정한 α·β). best.pt(가장 좋은 epoch의 가중치)는 Releases
  v3_synth500/              v3 YOLO26s(같은 파일 구성)
  v4_synth2000/             v4 YOLO26s(같은 파일 구성)
  v5_synth5000/             v5 YOLO26s, 최종 모델(같은 파일 구성 + best.pt, 89번째 epoch)
predictions/                v5(최종 모델)의 예측결과와 비교·분석 결과(CSV·JSON, 대회 제출본과 같음)
  analysis/                 v5 오류 분석(develop/analyze.py): 미검출·오검출 목록(test1_fp.csv 의 case = 오검출 경우, 9절), 합성과 실제 비교, 검증셋 임계값 표
  comparison/               v1–v5 비교: compare.csv(테스트셋 1), fp_spots.csv·fp_by_case.csv(v5의 오검출 위치에서 다른 모델도 오검출이 나왔나),
                            v1_params.json·v1_val_grid.csv·v1_test1_*.csv(v1), v2–v4 의 α·β와 결과(v2_thresholds.json, v2_test1_result.csv 등),
                            테스트셋 2 판정 비교(test2_summary.csv, test2_per_image.csv, test2_boxes.json)
  test1/test1_answers.csv   테스트셋 1 정답(검사 장비 bounding box에서 만든 것)
  test1/test1_boxes.csv     이미지별 검출 상자: α 이상은 평가에 쓴 상자(α 이상끼리 합침, 검출·오검출 표시), β–α는 재검사 범위 상자
  test1/test1_metrics.csv   테스트셋 1 정밀도·재현율·F1(전체, 장비별)
  test1/test1_per_image.csv 이미지별 정답 수, 검출(hit), 오검출(false_detection), 미검출(missed), 판정
  test2/test2_boxes.csv     이미지별 검출 상자
  test2/test2_per_image.csv 이미지별 판정
  thresholds.json           α(불량)·β(재검사) 임계값과 정한 규칙
src/                        소스코드(6절): 공통 함수, develop/(결과 재현용), release/(학습 없이 바로 쓰는 코드)
environment.yml             conda 가상환경(venv) 생성 파일
LICENSE                     AGPL-3.0
README.md                   영문 소개
README.ko.md                한국어 소개(README.md 와 같은 내용)
docs/manual_ko.md           사용 설명서(이 파일)
requirements-lock.txt       분석에 쓴 conda 가상환경의 pip freeze 전체
requirements.txt            사용 패키지와 버전(설치 순서는 2절)
```

[Releases](https://github.com/sinkei9456/gen-xray/releases)에서 받아 넣을 것:

| 파일 | 풀 곳 | 내용 |
|---|---|---|
| `diffusion_model.zip` | `models/` | `models/diffusion/`: Diffusion 모델 가중치(EMA)와 설정 |
| `yolo_v2_v3_v4_weights.zip` | `models/` | v2·v3·v4의 `best.pt` |
| `yolo26s.pt` | `models/` | YOLO 학습을 시작한 COCO 사전학습 가중치(Ultralytics 배포본, 2절) |
| `synthetic_v5_5000.zip` | `data/` | `data/v5_synth5000/`: 합성 5,000장. `train/images, labels/` 학습셋 4,500장(이물 10,410개, 그중 새 형상 3,139개), `val/images, labels/` 검증셋 500장(이물 1,138개, 그중 새 형상 343개), `generation.json`(빈 화면 출발 단계 t = 999, 시드, 새 형상 설정), `meta.csv`(합성 이미지별 지시어, 장비, 막대 수, 이물 수, 새 형상 수, 이물 출처) |

코드가 다시 만드는 것(대회 원본에서, 3절):

```
data/
  material_500/             학습 재료(develop/prepare_material.py, measure_shapes.py): 이물과 검사 장비 표시를 지운 봉지·막대 이미지 500장(bags/),
                            Diffusion 학습 입력(diffusion_cache_500.npz), 밝기·기울기(levels_500.json), 이물 패치 1,147개(patches.npz),
                            장비별 설정값(profiles_500.json), 지시어 비율(prompt_stats.json), 새 형상 흐림(shapes_500.json, σ 0.6픽셀)
  v2_real500/               v2의 학습 데이터(develop/prepare_real500.py): 표시를 지운 레이블 500장(all/), 학습셋 400장·검증셋 100장(train/, val/)
  test1/images/             테스트셋 1: 실제 1,870장(검사 장비 표시를 지운 이미지, develop/v5_eval.py)
  test2/images/             테스트셋 2: 실제 107장(검사 장비 bounding box가 없던 이미지, 표시를 지움)
  v3_synth500/, v4_synth2000/   v3·v4의 학습 데이터 = 합성 5,000장의 앞부분(develop/make_subsets.py, 몇 초)
```

코드를 실행하면 이 폴더 안에 `runs/`(학습·평가 결과), `output/`(v5 결과 그림)도 생기고, 5-2절의 `run_all.sh`는 `doc/fig/`·`report/slides_assets/`(보고서·발표 그림)도 만든다.

결과 그림 색: 초록 = 정답(점선 = 겹친 bounding box 영역), 빨강 = confidence α 이상, 주황 = β 이상 α 미만, 숫자 = confidence.

## 2. 실행 환경

conda 가상환경 **venv**를 쓴다(분석에 쓴 것과 같은 패키지 버전).

| 항목 | 버전 |
|---|---|
| OS, CPU | Ubuntu 24.04.4 LTS, aarch64 (ARM 64비트) |
| 장비 | NVIDIA DGX Spark (GB10 Grace Blackwell 슈퍼칩, CPU 와 GPU 가 메모리 121GB 하나를 같이 씀) |
| GPU | GB10, 드라이버 580.126.09, CUDA 13.0 |
| 환경 관리 | conda 25.9.1, Python 3.11.16 가상환경 |
| 주요 패키지 | torch 2.14.0 (CUDA 13.0 휠), torchvision 0.29.0, ultralytics 8.4.157, diffusers 0.40.0, transformers 5.17.0, accelerate 1.15.0, opencv-python-headless 5.0.0.93, numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, matplotlib 3.11.2, pillow 12.3.0 |
| 전체 목록 | `requirements-lock.txt` (그 환경의 pip freeze) |

설치(둘 중 하나):

```bash
# 방법 1: environment.yml
conda env create -f environment.yml
conda activate venv

# 방법 2: 직접 설치
conda create -n venv python=3.11.16 -y
conda activate venv
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130   # 다른 GPU는 그 CUDA에 맞는 torch 2.14.0 휠
pip install -r requirements.txt
```

YOLO 학습은 `models/yolo26s.pt`(COCO로 학습된 Ultralytics 배포본, 분석에 쓴 것과 같은 파일, md5 372e5c34064f37eb45dd7ef5cbbe60aa)에서 시작한다. 배포본이 바뀌어도 같은 결과가 나오도록 Releases에 함께 올렸다. 이 파일이 없으면 Ultralytics가 다른 배포본을 받아 올 수 있다.

## 3. 대회 원본 데이터(KAMP에서 받아 넣기)

원본은 KAMP(https://www.kamp-ai.kr/)의 **X-ray 검사장비 AI 데이터셋**이다. 재배포하지 않으므로 KAMP에서 받아, 코드가 읽는 두 폴더를 원래 경로 그대로 `data/dataset/`에 넣는다(대회 zip `4. X-ray 검사장비 AI 데이터셋.zip` 안의 `dataset/` 폴더를 `data/dataset/`에 둔 것과 같은 경로). 4절(`src/release/`)은 학습 재료(`data/material_500/`)만 있으면 돌아가고, 5절(`src/develop/`)은 이 원본에서 학습 재료와 테스트셋·정답을 다시 만든다(테스트셋 1의 정답은 원본에 그려진 검사 장비 표시에서 만든다).

```
data/dataset/test1/yolov3/X선이물검출기(06.23_09.22)/1호기(2020.09.22)/<날짜 폴더>/*.bmp   (1,031장)
data/dataset/test1/yolov3/X선이물검출기(06.23_09.22)/2호기(2020.09.22)/<날짜 폴더>/*.bmp   (920장)
data/dataset/test1/yolov3/X선이물검출기(06.23_09.22)/3호기(2020.09.22)/<날짜 폴더>/*.bmp   (858장)
data/dataset/라벨링 6종 세트/labels/*.txt                                             (레이블 500개)
```

원본을 넣은 뒤 학습 재료를 다시 만든다(같은 기계·환경에서 대회 제출본과 파일이 모두 같았다):

```bash
conda activate venv
cd src
python develop/prepare_material.py      # → data/material_500/ (1분 미만)
python develop/measure_shapes.py        # → data/material_500/shapes_500.json
```

인용(KAMP 가이드북의 출처 표기 양식): 중소벤처기업부, Korea AI Manufacturing Platform(KAMP), X-ray 검사장비 AI 데이터셋, KAIST(㈜임픽스, 한양대학교 산학협력단, ㈜아큐라소프트), 2020.12.14., www.kamp-ai.kr

파일 2,809개 중 이름이 같은 파일이 280쌍이다. 277쌍은 같은 장비의 하루 차이 나는 날짜 폴더 두 곳에 같은 이미지(픽셀까지 같음)가 들어 있는 것이고, 3쌍(`002_20200916_122650(1)`, `002_20200918_043338(2)`, `002_20200922_163053(8)`)은 같은 날짜의 1호기·2호기 폴더에 이름만 같고 내용이 다른 이미지다. 이름 기준으로 한 장씩 남겨 2,529장을 쓴다(이 3쌍은 1호기 이미지가 남고, 2호기 이미지 3장은 쓰지 않는다).

| 쓰임 | 장수 |
|---|---:|
| 학습 재료(레이블 있는 500장) | 500 |
| 테스트셋 1(검사 장비 bounding box 있음) | 1,870 |
| 테스트셋 2(검사 장비 bounding box 없음) | 107 |
| 뺀 이미지: 500장과 같은 장비로 60초 안에 연속 촬영 | 33 |
| 뺀 이미지: 날짜는 다르지만 500장과 거의 같음 | 19 |

## 4. 바로 쓰기(학습 없음): `src/release/`

### 4-1. Diffusion 모델로 합성 이미지 만들기(`release/generate.py`)

```bash
conda activate venv                # 가상환경 켜기(만들기는 2절)
cd src
python release/generate.py --prompt "2호기, 스틱 3개, 이물 3개" --n 5 --out ../runs/generate
python release/generate.py --prompt "1호기, 스틱 1개, 이물 하나" --prompt "3호기, 스틱 3개, 이물 3개" --n 2 --out ../runs/generate2
python release/generate.py --random 20 --out ../runs/generate_random      # 지시어를 레이블 500장의 장비·막대 수 비율대로 무작위로 뽑음
```

- 필요한 것: 학습 재료 `data/material_500/`(3절에서 만듦)와 Diffusion 모델 가중치(Releases의 `diffusion_model.zip`).
- 지시어: "N호기, 스틱 K개, 이물 M개". 장비는 1·2·3호기, 막대는 1개 또는 3개(레이블 500장에 있는 제품)다. 이물은 막대마다 차례로 하나씩 넣는다. 장비는 꼭 적는다(`--random`은 장비까지 500장의 비율대로 뽑는다).
- 만드는 방법은 학습 데이터(`develop/make_synth.py`)와 같다: 지시어 → 배치도(봉지·막대 위치) → Diffusion 모델이 빈 화면에서 봉지·막대를 그림(256×256으로 그린 뒤 장비 크기로 되돌림) → 레이블 500장 중 한 장의 밝기에 맞춤 → 이물을 막대 안에 넣음(점 패치, `--shape-frac` 비율만큼 새 형상. 기본 0.3 = v5 학습 데이터와 같음, 0이면 점 패치만).
- 결과(`--out` 폴더): `images/gen_00000.png …`(장비 크기 회색조: 1호기 352×332, 2호기 316×332, 3호기 576×444), `labels/gen_00000.txt …`(YOLO 형식: 0 중심x 중심y 폭 높이, 0–1), `meta.csv`(지시어·장비·막대 수·이물 수·새 형상 수·이물 출처), `preview.png`(앞 8장과 레이블 상자), `generation.json`(시드 등).
- 같은 시드(`--seed`, 기본 0)면 같은 이미지가 나온다. GB10에서 한 장에 약 1.5초(처음에 모델을 읽는 데 몇 초 더).

### 4-2. v5로 이미지 판정하기(`release/predict.py`)

```bash
conda activate venv                # 가상환경 켜기(만들기는 2절)
cd src
python release/predict.py --src ../data/test1/images --out ../runs/predict_test1          # 5-1절을 돌리면 생기는 테스트셋 1(표시를 지운 PNG)
python release/predict.py --src "../data/dataset/test1/yolov3/X선이물검출기(06.23_09.22)/2호기(2020.09.22)" --recursive --out ../runs/predict_raw   # 대회 원본 BMP
```

- 입력: 이미지 파일 하나 또는 폴더(.png .bmp .jpg). 대회 원본 BMP처럼 검사 장비 표시(색 픽셀)가 있으면 평가 때와 같은 방법(OpenCV 인페인팅)으로 지운 뒤 넣는다. 원본 폴더에 이름과 내용이 같은 이미지가 두 번 있으면 한 번만, 이름만 같고 내용이 다르면 이름 뒤에 `__2`를 붙여 따로 판정한다.
- 처리는 5절의 평가와 같다: YOLO 입력 640×640(7절) → confidence α(0.40) 이상 상자끼리 겹치거나 맞닿으면 하나로 합쳐 불량 상자로, 그 밖의 β(0.05)–α 상자는 재검사 상자로(만드는 방법은 8절) → 이미지에서 가장 높은 confidence가 α(0.40) 이상이면 불량, β(0.05) 이상이면 재검사, 아니면 정상(8절). α·β는 `models/v5_synth5000/thresholds.json`(v5 검증셋에서 정한 값)이다.
- 결과(`--out` 폴더): `boxes.csv`(상자별 이미지·좌표(원래 이미지 픽셀)·confidence·level: defect = α 이상, recheck = β–α), `images.csv`(이미지별 크기·지운 표시 픽셀 수·불량 상자 수·재검사 상자 수·가장 높은 confidence·판정 defect / recheck / normal), `vis/`(결과 그림, 파일 이름 앞에 판정), `inputs/`(모델에 넣은 이미지).
- 5-1절을 돌려 만든 테스트셋 1(`data/test1/images/`, 1,870장)을 넣으면 상자와 판정이 `predictions/`의 v5 결과와 같다(불량 1,869장, 재검사 1장, 정상 0장).

## 5. 결과 재현: `src/develop/`

### 5-1. 학습 없이 공개 모델로 v1–v5 평가와 비교 다시 내기

Releases 파일을 모두 넣고 학습 재료를 만든 뒤(1·3절) 실행한다. `run_all.sh`는 결과가 이미 있는 학습·생성 단계를 건너뛰고, 원본에서 v2 데이터·테스트셋·정답을 다시 만든 뒤 평가·비교·그림을 낸다. `run_eval.sh`는 평가와 비교만 한다(GB10에서 약 7분, v2 데이터는 없으면 만든다).

```bash
conda activate venv                # 가상환경 켜기(만들기는 2절)
cd src
bash develop/run_all.sh           # 학습 건너뜀 → 평가·비교·그림(학습 재료가 없으면 먼저 만듦)
bash develop/run_eval.sh          # 평가와 비교만. 다른 python 을 쓰려면 PY=<python 경로> bash develop/run_eval.sh
```

| 순서 | 코드 | 하는 일 | 결과 |
|---:|---|---|---|
| 1 | `develop/v5_eval.py` | 대회 원본에서 테스트셋 1·2를 다시 만들고(검사 장비 표시 지우기, 정답 만들기), v5 검증셋으로 α·β를 정한 뒤 평가, 결과 그림 | `runs/v5_eval/`, `output/` |
| 2 | `develop/analyze.py` | v5 오류 분석, 합성과 실제 비교 | `runs/v5_eval/` |
| 3 | `develop/v1_train.py`, `v1_eval.py` | v1 필터 크기·기준값 정하기(레이블 500장, 1분 미만) → 평가 | `models/v1_filter/params.json`(같은 값), `runs/v1_train/`, `runs/v1_eval/` |
| 4 | `develop/v2_eval.py` | v2 평가(α·β는 v2 검증셋에서) | `runs/v2_eval/` |
| 5 | `develop/make_subsets.py` | v3·v4의 학습셋·검증셋(없을 때만) | `data/v3_synth500/`, `data/v4_synth2000/` |
| 6 | `develop/v3_eval.py`, `v4_eval.py` | v3·v4 평가 | `runs/v3_eval/`, `runs/v4_eval/` |
| 7 | `develop/compare.py`, `test2_compare.py` | v1–v5 비교표·오검출 위치 비교, 테스트셋 2 비교 | `runs/compare/`, `runs/test2_compare/` |

나오는 결과(`predictions/`와 같다):

**v1–v5 학습 데이터 나누기**

| 모델 | 학습셋(가중치 학습) | 검증셋(early stopping, 임계값 결정) | 평가셋 |
|---|---|---|---|
| v1 학습 없는 필터 | 없음 | 레이블 500장 전부(표시 지움): 필터 크기·기준값 튜닝 | 테스트셋 1, 2 |
| v2 실제 500장 | 레이블 500장 중 400장(표시 지움) | 나머지 100장 | 테스트셋 1, 2 |
| v3 합성 500장 | 합성 5,000장의 1–400번째 | 401–500번째 | 테스트셋 1, 2 |
| v4 합성 2,000장 | 1–1,800번째 | 1,801–2,000번째 | 테스트셋 1, 2 |
| v5 합성 5,000장(최종) | 1–4,500번째 | 4,501–5,000번째 | 테스트셋 1, 2 |

- v2의 400 / 100: 같은 장비·같은 날에 앞 이미지와 60초 안에 찍힌 이미지를 한 묶음으로 보고(500장 = 114묶음), 묶음 순서를 시드 0으로 섞어 앞에서부터 묶음째 담아 검증셋을 정확히 100장(24묶음)으로 만든다(100장을 넘게 하는 묶음은 건너뜀, `develop/prepare_real500.py`). 연속 촬영 이미지가 학습셋과 검증셋에 갈라지지 않는다.
- v3·v4·v5: 시드 0으로 한 번 만든 합성 5,000장의 앞부분을 잘라 쓴다(`develop/make_subsets.py`). 차이는 양뿐이다.

**테스트셋 1(이물 3,194개, α 이상만 셈)**

| 모델 | 임계값 | 검출 | 미검출 | 오검출 | 정밀도 | 재현율 | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 학습 없는 필터 | 필터 5픽셀, 기준 16 | 2,977 | 217 | 560 | 0.842 | 0.932 | 0.885 |
| v2 실제 500장 | α 0.40, β 0.05 | 3,189 | 5 | 29 | 0.991 | 0.998 | 0.995 |
| v3 합성 500장 | α 0.55, β 0.05 | 3,129 | 65 | 30 | 0.991 | 0.980 | 0.985 |
| v4 합성 2,000장 | α 0.45, β 0.05 | 3,190 | 4 | 32 | 0.990 | 0.999 | 0.994 |
| **v5 합성 5,000장(최종)** | **α 0.40, β 0.05** | **3,191** | **3** | **32** | **0.990** | **0.999** | **0.995** |

v5 이미지 판정: 불량 1,869장, 재검사 1장, 정상 0장(테스트셋 1은 모든 이미지에 이물이 있으므로, 이물을 놓치고 정상으로 내보낸 이미지가 없다는 뜻). 오검출 위치 비교(`comparison/fp_by_case.csv`, 9절): v2·v3·v4의 오검출은 모두 v5의 오검출과 같은 위치에서 나왔다. v2의 오검출이 v5보다 3개 적은 것 중 2개는 검사 장비가 표시하지 않은 이물을 v2가 찾지 못한 것이다.

**테스트셋 2(정답 없음, 눈에 보이는 이물 5개, α 이상만 셈)**

| 모델 | 검출 | 미검출 | 오검출(봉지가 잘리거나 없는 104장) | 오검출이 나온 이미지(장) |
|---|---:|---:|---:|---:|
| v1 학습 없는 필터 | 5 | 0 | 48 | 42 |
| v2 실제 500장 | 4 | 1 | 0 | 0 |
| v3·v4·v5 | 5 | 0 | 0 | 0 |

v5의 테스트셋 2 판정: 불량 3장, 재검사 0장, 정상 104장(봉지가 잘리거나 없는 이미지, 검출 없음).

**최종 모델 선정 규칙(결과를 보기 전에 정함, v3·v4·v5 중에서)**: ① 테스트셋 1 F1이 가장 높은 모델 ② 차이가 0.001 이하면 미검출 + 오검출이 적은 모델 ③ 그것도 같으면 테스트셋 2 재검사가 적은 모델. v5 F1 0.9945, v4 0.9944 → ②에서 v5 35개 < v4 36개로 v5를 골랐다.

### 5-2. 처음부터 다시 만들기(GB10 기준 약 8–10시간)

`develop/run_all.sh`는 결과가 이미 있는 단계를 건너뛴다(학습 재료·학습 데이터는 `data/`, 학습한 모델은 `models/`의 가중치 파일, YOLO 학습 기록은 `runs/`를 본다). 처음부터 다시 만들려면 Releases 파일 중 `yolo26s.pt`만 넣고(나머지는 넣지 않음), 저장소에 들어 있는 v5 가중치 `models/v5_synth5000/best.pt`를 지운 뒤 실행한다. 하나라도 남아 있으면 그 단계는 건너뛰고 공개 모델·데이터를 그대로 쓰므로 처음부터 만든 것이 아니게 된다.

| 구분 | 폴더·파일(저장소 맨 위 기준) | 이유 |
|---|---|---|
| 지울 것 | `models/v5_synth5000/best.pt` | 4단계가 다시 학습한다 |
| 넣지 않을 것(넣었다면 지움) | `models/diffusion/diffusion_pytorch_model.safetensors`, `models/v2_real500/best.pt`, `models/v3_synth500/best.pt`, `models/v4_synth2000/best.pt`, `data/v5_synth5000/`, `data/v3_synth500/`, `data/v4_synth2000/` | 2·3·7·8단계가 다시 만든다 |
| 지울 것 | `runs/`, `output/`, `doc/`, `report/`(전에 돌린 결과가 있으면) | 남아 있으면 새 결과와 섞인다. `runs/v번호_train/train_done`이 남아 있으면 그 모델의 YOLO 학습을 건너뛴다 |
| 남길 것 | `data/dataset/`(3절), `models/yolo26s.pt`(Releases) | 대회 원본, YOLO 학습을 시작하는 사전학습 가중치 |
| 남길 것 | `models/*/args.yaml`·`results.csv`·`thresholds.json`, `models/v1_filter/`, `models/diffusion/config.json`·`sample.json` | 학습·평가가 다시 쓰므로 그대로 둬도 된다 |

```bash
# 저장소 맨 위에서
rm -f models/v5_synth5000/best.pt
rm -rf runs output doc report
conda activate venv                # 가상환경 켜기(만들기는 2절)
cd src && bash develop/run_all.sh 2>&1 | tee ../run_all.log   # 약 8–10시간. 터미널을 닫거나 접속이 끊겨도 계속 돌도록 tmux 등에서 실행
```

| 단계 | 코드 | 결과 | 걸린 시간(실측: 10월 3–4일 / 10월 7–8일 다시 실행) |
|---|---|---|---|
| 1. 500장 → 봉지·막대 이미지, 이물 패치, 설정값 | `develop/prepare_material.py` | `data/material_500/` | 1분 미만 |
| 2. Diffusion 모델 학습(8,000 스텝, 배치 16) | `develop/train_diffusion.py` | `runs/diffusion/` → `models/diffusion/` | 5시간 46분 / 3시간 48분 |
| 3. 새 형상 흐림 재기 → 합성 5,000장(새 형상 30%) | `develop/measure_shapes.py`, `develop/make_synth.py` | `data/material_500/shapes_500.json`, `data/v5_synth5000/`, `models/diffusion/sample.json` | 2시간 6분 / 2시간 8분 |
| 4. v5 YOLO26s 학습(최대 100 epoch, early stopping 20) | `develop/v5_train.py` | `runs/v5_train/` → `models/v5_synth5000/` | 약 73분 |
| 5. v5 평가(α·β 결정, 테스트셋 1·2), 오류 분석 | `develop/v5_eval.py`, `develop/analyze.py` | `runs/v5_eval/`, `output/` | 약 5분 |
| 6. v2 데이터, v1 필터 정하기·평가 | `develop/prepare_real500.py`, `v1_train.py`, `v1_eval.py` | `data/v2_real500/`, `models/v1_filter/`, `runs/v1_*` | 약 1분 |
| 7. v2 학습(36 epoch에서 early stopping)·평가 | `develop/v2_train.py`, `v2_eval.py` | `models/v2_real500/`, `runs/v2_*` | 약 4분 |
| 8. v3·v4 데이터, 학습·평가 | `develop/make_subsets.py`, `v3_*`, `v4_*` | `data/v3_synth500/`, `data/v4_synth2000/`, `models/v3_*`, `models/v4_*`, `runs/v3_*`, `runs/v4_*` | 약 35분 |
| 9. v1–v5 비교표, 테스트셋 2 비교, 보고서·발표 그림 | `develop/compare.py`, `test2_compare.py`, `make_figures.py` | `runs/compare/`, `runs/test2_compare/`, `doc/fig/`, `report/slides_assets/` | 약 2분 |

- 단계 사이마다 이전 프로세스 종료, 파일 캐시 비우기, 가용 메모리(40GB) 확인을 한다(`mem_reset.py`).
- 끝나면 `run_all.log` 마지막 줄에 `ALL_DONE`이 나온다. 새 결과는 `runs/compare/compare.csv`(v1–v5 테스트셋 1 비교표), `runs/v5_eval/test1_result.csv`(v5 장비별), `runs/v번호_eval/thresholds.json`(α·β) 등에 있고, 5-1절의 표와 비교한다. 공개한 결과(`predictions/`, `models/*/thresholds.json`)와도 비교할 수 있다.
- 중간에 멈추면 같은 명령을 다시 실행한다. 끝난 단계는 건너뛰고, YOLO 학습은 멈춘 epoch부터 이어 하며(`runs/v번호_train/train/weights/last.pt`), Diffusion 학습과 합성은 처음부터 다시 한다. 이어 한 YOLO 학습은 한 번에 끝까지 돌린 것과 결과가 똑같다고 보장하지 않는다.
- 모델 하나만 다시 학습하려면 그 모델의 `models/v번호_…/` 폴더와 학습 기록 `runs/v번호_train/`을 함께 지우고 `run_all.sh`를 돌리거나, `python develop/v4_train.py`처럼 직접 실행한다(`runs/v번호_train/train_done`이 남아 있으면 학습을 건너뛴다). 학습 스크립트는 끝나면 가장 좋은 가중치를 `models/`의 그 모델 폴더로 복사한다(`--no-export`를 주면 복사하지 않음).
- 다른 GPU 작업이 함께 돌면 시간이 더 걸린다.

**시드와 재현 범위**

- 모든 단계의 난수 시드를 고정했다(학습·합성·나누기는 시드 0, 생성 시작 깊이를 고를 때 시험으로 그려 보는 이미지만 시드 1).
  - numpy·random·torch 시드
  - Diffusion 학습: `torch.use_deterministic_algorithms(True)`, cuDNN 고정 설정
  - 합성 이미지마다 쓰는 torch 난수도 numpy 난수에서 뽑아 고정(새 형상도 같은 난수열에서 뽑음)
  - YOLO: `seed=0, deterministic=True`
- 같은 기계·같은 환경(GB10, 2절의 패키지 버전)에서 2026년 10월 3–8일에 실제로 돌려 확인한 것:
  - 1단계 결과와 새 형상 흐림 측정값은 다시 만들어도 파일이 모두 같았다.
  - Diffusion 학습 30스텝을 두 번, YOLO 학습 2 epoch를 두 번 하니 가중치가 완전히 같았다.
  - 학습 스크립트(`v1_train.py`–`v5_train.py`, `train_diffusion.py`)를 임시 폴더에서 짧게(YOLO는 학습셋의 5%로 1 epoch, Diffusion은 20스텝) 돌려 끝까지 실행되고, YOLO 학습 설정이 제출 모델의 `args.yaml`과 epoch·데이터 비율·폴더 경로 말고 모두 같음을 확인했다.
  - 대회 제출 폴더만 새 위치에 복사해(대회 원본은 제출 zip에 들어 있던 `data/dataset/`을 씀) 4절·5-1절의 명령과 합성 점검(`python develop/make_synth.py --limit 40 --out <임시 폴더>`, 5,000장의 앞 40장)을 돌려, 학습 재료, 다섯 모델의 평가(지표·α·β·이미지별 판정)와 비교표, 오검출 위치 비교, 테스트셋 2 비교, v2의 400 / 100 나누기, v3·v4의 학습셋·검증셋, 합성 이미지(최대 픽셀 차이 0), `release/predict.py`의 판정이 제출본과 모두 같음을 확인했다.
  - 10월 7–8일에는 대회 제출 zip만 새 폴더에 풀고 가상환경(2절)을 새로 만든 뒤, 학습 재료·학습 데이터·모델·실행 결과 폴더를 지우고 `run_all.sh`로 처음부터 끝까지 다시 돌렸다(7시간 55분). 학습 재료(507개 파일), Diffusion 가중치(8,000스텝), 합성 5,000장의 이미지·레이블, YOLO v2–v5의 가중치, 평가 결과(48개 파일), 결과 그림(1,977장), 보고서 그림이 제출본과 모두 같았다.
- GPU 종류나 torch·CUDA 버전이 다르면 소수점 계산 차이로 결과가 조금 달라질 수 있다. 공개 결과와 똑같은 숫자가 필요하면 공개한 모델과 데이터(Releases)를 쓴다.
- 같은 기계·같은 환경에서는 처음부터 다시 만들어도 결과가 같았다(위). 다른 GPU·버전에서 결과가 조금이라도 달라지면 최종 모델 선정(5-1절)이 바뀔 수 있다. 제출 결과에서 v5와 v4의 테스트셋 1 F1 차이는 0.0001(0.9945 대 0.9944)이고 미검출 + 오검출은 35개 대 36개라, 이물 1–2개 차이로 v4가 뽑힐 수 있다.

## 6. 소스코드

```
src/                         공통 함수
├─ paths.py                  폴더 위치와 모델 이름(v1–v5)
├─ splits.py                 대회 이미지 목록(이름 기준 중복 제거), 장비 구분, 500장과 연속 촬영된 이미지 찾기
├─ test_answers.py           검사 장비 표시 → 테스트셋 1 정답(점·영역·"T" 표시 제외)
├─ mem_reset.py              단계 사이 메모리 정리
├─ detect.py                 상자 합치기·짝짓기·판정·표시 지우기·결과 그림·묶음 추론, YOLO 학습·평가
├─ sd/                       Diffusion 모델 정의·생성
├─ synth/                    지시어 해석·배치도·이물 넣기·새 형상 그리기
├─ develop/                  결과 재현용
│  ├─ prepare_material.py    500장 → 봉지·막대 이미지·이물 패치
│  ├─ measure_shapes.py      새 형상 흐림 재기
│  ├─ train_diffusion.py     Diffusion 모델 학습
│  ├─ make_synth.py          합성 5,000장 + 학습·검증 나누기
│  ├─ make_subsets.py        v3·v4용 앞 500·2,000장
│  ├─ prepare_real500.py     v2용 표시 지운 실제 500장
│  ├─ v1_train.py  v1_eval.py   학습 없는 필터(500장에서 필터 크기·기준값 정하기 / 평가)
│  ├─ v2_train.py  v2_eval.py   실제 500장
│  ├─ v3_train.py  v3_eval.py   합성 500장
│  ├─ v4_train.py  v4_eval.py   합성 2,000장
│  ├─ v5_train.py  v5_eval.py   합성 5,000장(최종)
│  ├─ analyze.py  compare.py  test2_compare.py  make_figures.py  export_submission.py
│  └─ run_all.sh(전체 재현)  run_eval.sh(학습 없이 제출 모델로 v1–v5 평가·비교)
└─ release/                  사용용(학습 없음)
   ├─ generate.py            Diffusion 합성 이미지 생성
   └─ predict.py             v5로 추론 → 결과 파일
```

| 파일 | 하는 일 |
|---|---|
| `develop/v2_train.py`–`v5_train.py` | YOLO26s 학습. 설정(최대 100 epoch, early stopping 20, 입력 640×640, 배치 32, 시드 0, 증강)은 `detect.train_yolo`에 있고 네 모델이 같다. 다른 것은 학습 데이터뿐 |
| `develop/v2_eval.py`–`v5_eval.py` | 평가 절차는 `detect.evaluate`(네 모델 같음): 검증셋에서 α·β → 테스트셋 1 정량 평가(정답 규칙: `test_answers.py`) → 테스트셋 2 판정(v3–v5. v2의 테스트셋 2 판정은 `test2_compare.py`가 낸다). v5는 결과 그림(`output/`)도 만든다. 정한 α·β는 모델 폴더의 `thresholds.json`에도 저장 |
| `develop/analyze.py` | v5 오류 분석: 미검출·오검출 분류, 오검출을 같은 봉지의 다른 사진과 비교해 경우 나누기(9절), 합성과 실제의 밝기·잡음 비교 |
| `develop/compare.py` | v1–v5 비교표(테스트셋 1), v5의 오검출 위치(같은 이미지, 상자 중심 6픽셀 안)에서 다른 모델도 오검출이 나왔는지 |
| `develop/test2_compare.py` | 테스트셋 2에서 모델별 판정·검출 위치 비교(학습·임계값 선택 없음) |
| `develop/make_figures.py` | 보고서·발표 그림(`doc/fig/`, `report/slides_assets/`) |
| `develop/export_submission.py` | 대회 제출 폴더 만들기(모델 파일·학습 기록 안의 서버 경로를 지움, 가중치는 그대로) |

## 7. 이미지 크기

| 단계 | 크기(가로×세로, 픽셀) |
|---|---|
| 대회 원본(중복 뺀 2,529장) | 1호기 316×332(431장)·352×332(392장)·412×332(120장), 2호기 316×332(802장), 3호기 576×444(784장) |
| Diffusion 모델 | 256×256으로 학습하고 그린다. 이미지를 긴 변 길이의 정사각형으로 채운 뒤 줄이고, 그린 그림은 다시 늘려 채운 부분을 잘라 장비 크기로 되돌린다 |
| 합성 이미지 | 장비 크기 그대로: 1호기 352×332, 2호기 316×332, 3호기 576×444(합성 5,000장은 1,577·1,757·1,666장). 이물은 장비 크기로 되돌린 뒤에 넣는다 |
| YOLO 입력 | 학습·평가 모두 640×640(imgsz 640). 비율을 지킨 채 긴 변을 640에 맞추고 남는 곳은 회색(114)으로 채운다: 576×444 → 640×493, 352×332 → 640×604, 316×332 → 609×640, 412×332 → 640×516. 평가 때는 Ultralytics의 rect(같은 크기끼리 묶이면 회색을 덜 채움)를 끄고 늘 640×640으로 넣는다. 찾은 상자는 원래 이미지 좌표로 되돌려 정답과 비교한다 |

## 8. 판정 기준

이미지에서 가장 높은 confidence(겹친 상자를 합친 뒤)가 α(0.40) 이상이면 불량, β(0.05) 이상 α 미만이면 재검사(사람이 다시 확인), β 미만이면 정상(이물이 없다고 판정해 그대로 다음 공정으로 보냄)이다. α·β는 검증셋(합성 500장)에서만 정했다.
- α: F1이 가장 높은 값(같은 F1이 여러 개면 그 범위의 가운데 값)
- β: α에서 0.05씩 내려가며, 그 기준 이상인 상자를 검증셋 정답과 짝지어(중심이 정답 상자 안 또는 IoU 0.3 이상) 잰 정밀도 = 검출 ÷ (검출 + 오검출)이 0.9 아래로 처음 떨어지기 직전의 값(0.9 이상이 이어지는 동안 계속 내려감). 합성 검증셋에서는 0.05까지 내려도 정밀도가 0.999라 후보의 하한 0.05에서 멈췄다(비교 모델도 모두 0.05). 현장에서는 실제 이미지로 재검사 양을 보고 다시 정해야 한다.
- 테스트셋의 검출·미검출·오검출은 α 기준으로만 센다. β는 이미지 판정(재검사)에만 쓴다.
- 결과 파일·그림의 상자: 불량 상자(빨강)는 α 이상 상자끼리 합친 것으로 점수를 셀 때의 상자와 같다. 재검사 상자(주황)는 두 단계로 만든다. ① confidence가 β 이상인 상자를 α 이상인 것까지 모두 골라, 겹치거나 맞닿은 것끼리 같은 방법으로 하나로 합친다. ② 합친 상자 가운데 α 이상 상자가 하나도 들어 있지 않은 것만 재검사 상자로 남긴다. 그래서 불량 상자와 겹친 약한 상자(α 미만)는 ①에서 불량 상자 쪽과 한 묶음이 되어 ②에서 빠진다. 불량 상자는 α 이상끼리만 합친 것이라 이 약한 상자 때문에 커지지 않는다.

## 9. 오검출 경우(`predictions/analysis/test1_fp.csv` 의 case)

테스트셋 1의 정답은 검사 장비가 그린 bounding box라, 장비가 표시하지 않은 이물을 모델이 찾으면 오검출로 센다. 이 데이터는 같은 봉지를 몇 초 간격으로 여러 번 찍었으므로(연속 촬영 묶음: 같은 장비·같은 날, 앞 사진과 60초 안), 오검출이 나온 사진을 같은 봉지의 다른 사진과 비교해 나눴다(`develop/analyze.py`). 점수는 바꾸지 않는다.

| case | 뜻 | v5 |
|---|---|---:|
| `boxed_in_other_shot` | 같은 봉지의 다른 사진에서는 검사 장비가 모델이 찾은 수만큼 bounding box를 그림(그 사진 = `boxed_shot`, 모델 검출 위치와 장비 표시 위치를 회전·이동으로 맞춘 뒤 점 사이 거리의 제곱평균 = `layout_diff_px`, 최대 4.4픽셀) | 23 |
| `found_in_all_shots` | 같은 봉지의 모든 사진에서 모델은 같은 수를 찾았고, 장비는 모든 사진에서 그보다 적게 표시 | 7 |
| `beside_missed` | 오검출 상자가 같은 이미지의 미검출 정답과 겹침(정답 위치가 이물에서 살짝 벗어나 같은 이물을 미검출 1 + 오검출 1로 셈) | 1 |
| `this_shot_only` | 그 밖(모델이 같은 봉지의 다른 사진보다 이 사진에서 더 많이 찾음). v5에서는 흐린 점 1개 | 1 |

`predictions/comparison/fp_by_case.csv`: v5의 오검출 32개를 위의 경우로 나누고, 같은 위치(같은 이미지, 상자 중심 6픽셀 안)에서 v2·v3·v4도 오검출이 나왔는지 센 표다. 숫자는 각 모델의 오검출 수다. `fp_spots.csv`는 v5의 오검출 하나하나에 대한 각 모델의 결과와 그 위치의 가장 높은 confidence다.

| case | v2 | v3 | v4 | v5 |
|---|---:|---:|---:|---:|
| `boxed_in_other_shot` | 22 | 22 | 23 | 23 |
| `found_in_all_shots` | 6 | 7 | 7 | 7 |
| `beside_missed` | 1 | 1 | 1 | 1 |
| `this_shot_only` | 0 | 0 | 1 | 1 |
| 합계(각 모델의 오검출 수) | 29 | 30 | 32 | 32 |

v5에는 있고 v2에는 없는 오검출 3개 중 2개는 장비가 표시하지 않은 이물을 v2가 α 아래 confidence(0.195, 0.054)로만 본 것이고, 1개는 흐린 점이다. v3는 v5보다 2개 적은데, 흐린 점을 잡지 않았고 같은 봉지의 다른 사진에서 장비가 표시한 이물 1개를 v3의 α(0.55) 바로 아래(0.549)로 봤기 때문이다.
