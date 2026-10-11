# Gen X-ray

**Diffusion 모델로 만든 합성 이미지만으로 학습한 X-ray 이물 탐지**

[English](README.md) | 한국어 | 자세한 사용 설명서: [docs/manual_ko.md](docs/manual_ko.md)

식품 포장공정의 이 X-ray 데이터에는 검사 장비가 찾은 이물마다 색 사각형(bounding box)을 이미지 위에 직접 그려 두었다. 이 이미지로 검출 모델을 학습하면 모델은 3–4픽셀 크기의 이물이 아니라 사각형을 배운다. Gen X-ray는 실제 이미지를 학습에 한 장도 쓰지 않는다. 사람 레이블이 있는 500장은 재료로만 쓰고, 봉지와 막대를 그리는 조건부 Diffusion 모델을 학습한 뒤, 실제 이물 패치와 프로그램으로 그린 형상을 X-ray 감쇠(곱셈)대로 넣어 레이블을 저절로 얻는다. 이렇게 만든 합성 이미지 5,000장으로 학습한 YOLO26s는, 학습과 임계값 결정에 쓰지 않은 실제 테스트 이미지 1,870장(이물 3,194개)에서 **F1 0.995**를 냈다.

![원본 이미지의 검사 장비 표시](docs/figures/fig1_data_problem.png)

*원본 이미지(위)와 확대(아래). 작은 색 사각형 = X-ray 검사 장비가 이물마다 그린 표시, 큰 노란 사각형 = 아래에 확대한 곳.*

## 방법

1. **분리**: 레이블 500장을 봉지·막대 이미지(검사 장비 표시와 이물을 지움)와 이물 패치 1,147개로 나눈다. 이물 패치는 투과율(이물 위치의 밝기 ÷ 주변 밝기)로 저장한다.
2. **조건부 Diffusion 모델 학습**: 봉지·막대 이미지 500장으로 U-Net(4,077만 파라미터, 8,000 스텝, 256×256)을 학습한다. 조건은 배치도(봉지·막대 위치)와 장비 번호다.
3. **합성**: 지시어(예: `"2호기, 스틱 3개, 이물 3개"`) → 배치도 → Diffusion 모델이 빈 화면에서 봉지·막대를 그림(DDIM, 50단계) → 실제 이미지 한 장의 밝기에 맞춤 → 투과율을 곱해 이물을 넣음(실제 점 패치 70%, 새 형상 30%: 곧은 선·휜 선·꺾인 선·작은 조각) → YOLO 레이블. 합성 5,000장, 이물 11,548개.
4. **검출과 판정**: COCO 사전학습 YOLO26s를 합성 4,500장으로 학습하고, early stopping과 임계값은 나머지 합성 500장(검증셋)에서 정한다. 이미지마다 가장 높은 confidence로 판정한다: α(0.40) 이상이면 불량, β(0.05) 이상 α 미만이면 재검사(사람이 다시 확인), β 미만이면 정상(이물이 없다고 보고 다음 공정으로 보냄).

![합성 과정](docs/figures/fig4_generation.png)

*지시어 → 배치도(회색 = 봉지, 흰색 = 막대) → 빈 화면 → Diffusion 모델 출력 → 이물과 레이블(초록 = 점 패치, 빨강 = 새 형상).*

## 결과

테스트셋 1: 실제 1,870장(이물 3,194개). 정답은 검사 장비가 그린 사각형에서 만들었고, confidence α 이상만 센다. 이 이미지들은 학습과 임계값 결정에 쓰지 않았다.

| 모델 | 학습 데이터 | 미검출(개) | 오검출(개) | 정밀도 | 재현율 | F1 |
|---|---|---:|---:|---:|---:|---:|
| v1 학습 없는 필터(black-hat 연산) | 없음(필터 크기·기준값만 레이블 500장에서 고름) | 217 | 560 | 0.842 | 0.932 | 0.885 |
| v2 YOLO26s, 실제 500장 | 실제 400장(검사 장비 표시를 지움) | 5 | 29 | 0.991 | 0.998 | 0.995 |
| v3 YOLO26s, 합성 500장 | 합성 400장 | 65 | 30 | 0.991 | 0.980 | 0.985 |
| v4 YOLO26s, 합성 2,000장 | 합성 1,800장 | 4 | 32 | 0.990 | 0.999 | 0.994 |
| **v5 YOLO26s, 합성 5,000장(최종)** | **합성 4,500장** | **3** | **32** | **0.990** | **0.999** | **0.995** |

- 실제 이미지를 한 장도 학습하지 않은 v5가 실제 이미지로 학습한 v2와 같은 F1을 냈고, 미검출은 더 적다(3개 대 5개). 장비별 F1은 1호기 0.998, 2호기 1.000, 3호기 0.986이다. IoU 0.5 기준 F1(10픽셀 상자)은 0.992다.
- 테스트셋 1의 이미지 판정(모든 이미지에 이물이 있음): 불량 1,869장, 재검사 1장, **정상 0장**. 이물이 있는데 정상으로 판정해 내보낸 이미지가 없다.
- 테스트셋 2(검사 장비 사각형이 없는 107장): 이물이 보이는 3장은 모두 불량, 봉지가 잘리거나 없는 104장은 검출 없이 모두 정상이다.
- 오검출 32개 중 30개는 검사 장비가 표시하지 않은 이물로 보인다. 그중 23개는 같은 봉지를 몇 초 간격으로 다시 찍은 다른 사진에서는 장비가 표시했다. 그래서 이 데이터에서는 오검출이 적다고 더 좋은 모델이 아니다.

![테스트셋 1 예시](docs/figures/fig8_test_examples.png)

*테스트셋 1 예시: 초록 = 정답(검사 장비가 표시한 위치), 빨강 = YOLO가 α 이상으로 찾은 상자, 숫자 = confidence.*

그림은 [`docs/figures/`](docs/figures/)에 더 있다.

### 한계

- 데이터셋이 하나이고 실제 이물도 한 종류(3–4픽셀 점)다. 테스트셋 1에서 유일하게 점이 아닌 이물(철사 모양)은 v5 confidence가 0.29로 α 아래라, 그 이미지는 재검사로 넘어간다.
- 정답이 검사 장비 표시라, 장비가 놓친 이물을 모델이 찾으면 오검출로 센다.
- 모델마다 한 번씩만 학습했다(시드 0). 최종 모델은 미리 정한 규칙으로 테스트셋 1 결과를 보고 v3–v5 중에서 골랐고, v5와 v4의 차이는 이물 1개다.
- 합성 이미지의 잡음은 실제와 다르다. 합성 이미지 봉지 잡음의 90% 값은 실제의 1.6–3.6배이고, 3호기는 합성 이미지 중간값이 실제보다 매끈하다. 3호기에 오검출 32개 중 28개가 몰렸지만, 이것이 원인인지는 확인하지 않았다.

## 폴더 구성

```
data/                       KAMP 원본을 넣을 곳(들어 있지 않음, 아래 "데이터"). 합성 데이터는 Releases
docs/figures/               보고서 그림
models/                     v1 필터 값, Diffusion 설정, YOLO 학습 기록과 α·β,
                            v5 가중치(models/v5_synth5000/best.pt). 나머지 가중치는 Releases
predictions/                v5와 비교 결과(CSV·JSON, 대회 제출본과 같음)
src/                        소스코드: 공통 함수, develop/(결과 재현), release/(합성 이미지 생성, 판정)
environment.yml, requirements.txt, requirements-lock.txt
```

코드 주석은 한국어다. 주석의 "제출본", "제출 폴더"는 대회 제출 zip을 말하고, "README N절"은 [docs/manual_ko.md](docs/manual_ko.md)의 절 번호와 같다.

## 실행 환경

분석(Diffusion 학습, 합성, YOLO 학습, 평가)에 쓴 환경은 다음과 같다.

| 항목 | 버전 |
|---|---|
| OS, CPU | Ubuntu 24.04.4 LTS, aarch64 (ARM 64비트) |
| 장비 | NVIDIA DGX Spark (GB10 Grace Blackwell 슈퍼칩, CPU 와 GPU 가 메모리 121GB 하나를 같이 씀) |
| GPU | GB10, 드라이버 580.126.09, CUDA 13.0 |
| 환경 관리 | conda 25.9.1, Python 3.11.16 가상환경 |
| 주요 패키지 | torch 2.14.0 (CUDA 13.0 휠), torchvision 0.29.0, ultralytics 8.4.157, diffusers 0.40.0, transformers 5.17.0, accelerate 1.15.0, opencv-python-headless 5.0.0.93, numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, matplotlib 3.11.2, pillow 12.3.0 |
| 전체 목록 | `requirements-lock.txt` (그 환경의 pip freeze) |

```bash
conda env create -f environment.yml      # conda 가상환경 "venv" 만들기
conda activate venv
```

또는 직접 설치한다.

```bash
conda create -n venv python=3.11.16 -y && conda activate venv
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130   # 다른 GPU는 그 CUDA에 맞는 torch 2.14.0 휠
pip install -r requirements.txt
```

`environment.yml`로 새로 만든 환경도 패키지 버전이 같다. 아래 "실행"의 처음부터 전부 재현은 이렇게 새로 만든 환경에서(대회 제출본으로, 같은 기계에서) 돌렸고, 가중치와 결과가 모두 같았다.

## 데이터

원본은 KAMP(Korea AI Manufacturing Platform, https://www.kamp-ai.kr/)의 **X-ray 검사장비 AI 데이터셋**이다. 재배포하지 않으므로 KAMP에서 받아, 코드가 읽는 두 폴더를 원래 경로 그대로 `data/dataset/`에 넣는다.

```
data/dataset/test1/yolov3/X선이물검출기(06.23_09.22)/{1,2,3}호기(2020.09.22)/<날짜 폴더>/*.bmp   (BMP 2,809장)
data/dataset/라벨링 6종 세트/labels/*.txt                                                  (레이블 500개)
```

그다음 레이블 500장에서 학습 재료를 다시 만든다(항상 같은 결과, 1분 정도).

```bash
cd src
python develop/prepare_material.py      # 봉지·막대 이미지, 이물 패치, 장비별 설정값 → data/material_500/
python develop/measure_shapes.py        # 실제 점 이물에서 잰 새 형상의 흐림 → data/material_500/shapes_500.json
```

## Releases 파일

[Releases](https://github.com/sinkei9456/gen-xray/releases) 페이지에서 받는다.

| 파일 | 풀 곳(복사할 곳) | 내용 |
|---|---|---|
| `diffusion_model.zip` | `models/` | Diffusion 모델(EMA 가중치, 설정, `sample.json`) |
| `yolo_v2_v3_v4_weights.zip` | `models/` | v2·v3·v4의 `best.pt` |
| `yolo26s.pt` | `models/` | YOLO 학습을 시작한 COCO 사전학습 YOLO26s(분석에 쓴 Ultralytics 파일 그대로) |
| `synthetic_v5_5000.zip` | `data/` | 합성 이미지 5,000장과 레이블(학습셋 4,500장 / 검증셋 500장) |

## 실행

명령은 모두 가상환경을 켠 상태에서 `src/`에서 실행한다.

```bash
# Diffusion 모델로 합성 이미지와 레이블 만들기(data/material_500 과 diffusion_model.zip 필요)
python release/generate.py --prompt "2호기, 스틱 3개, 이물 3개" --n 5 --out ../runs/generate
python release/generate.py --random 20 --out ../runs/generate_random

# v5로 이미지 판정(원본 BMP의 검사 장비 표시는 평가 때처럼 먼저 지움)
python release/predict.py --src "../data/dataset/test1/yolov3/X선이물검출기(06.23_09.22)/2호기(2020.09.22)" --recursive --out ../runs/predict_raw
```

**학습 없이 평가 재현**(Releases 파일을 모두 넣은 상태): `bash develop/run_all.sh`. 결과가 이미 있는 학습·생성 단계는 모두 건너뛰고, 원본에서 테스트셋과 정답을 다시 만든 뒤 v1–v5 평가, 비교, 그림을 다시 낸다. 결과는 `predictions/`와 같아야 한다(같은 하드웨어·패키지 버전에서는 완전히 같았다). `bash develop/run_eval.sh`는 평가와 비교만 한다(GB10에서 약 7분). Releases 파일과 `data/material_500`(위 "데이터")이 필요하다.

**처음부터 전부 재현**(GB10에서 약 8–10시간): Releases 파일은 `yolo26s.pt`만 넣고, `models/v5_synth5000/best.pt`를 지운 뒤 `bash develop/run_all.sh`를 실행한다. Diffusion 모델 학습, 합성 5,000장 생성, v2–v5 학습과 평가를 모두 한다. 모든 시드를 고정하고 결정적 계산 모드로 학습하며, 같은 기계에서 처음부터 다시 돌렸을 때 Diffusion·YOLO 가중치와 모든 결과가 완전히 같았다. 다른 GPU나 버전에서는 소수점 차이가 날 수 있고, v5와 v4의 차이가 이물 1개라 최종 모델 선택이 바뀔 수 있다.

단계별 결과 파일, 판정 기준, 오검출 경우 나누기 등 자세한 내용은 [docs/manual_ko.md](docs/manual_ko.md)에 있다.

## 데이터 인용

중소벤처기업부, Korea AI Manufacturing Platform(KAMP), X-ray 검사장비 AI 데이터셋, KAIST(㈜임픽스, 한양대학교 산학협력단, ㈜아큐라소프트), 2020.12.14., www.kamp-ai.kr

이 작업은 제6회 K-인공지능 제조데이터 분석 경진대회(2026) 문제 ④로 했다.

## 라이선스

AGPL-3.0([LICENSE](LICENSE)). 이 코드가 쓰는 Ultralytics YOLO와 같은 조건이며, 검출 모델은 Ultralytics의 COCO 사전학습 가중치에서 학습했다. KAMP 원본은 들어 있지 않으며 KAMP 이용 조건을 따른다.
