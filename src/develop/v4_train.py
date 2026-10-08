"""v4 합성 2,000장 YOLO26s 학습: data/v4_synth2000(합성 5,000장의 앞 2,000장: 학습 1,800 / 검증 200) → runs/v4_train/, 가장 좋은 가중치를 models/v4_synth2000/ 로 복사.
설정(최대 100 epoch, early stopping 20, 입력 640×640, 배치 32, 시드 0, 증강)은 detect.train_yolo 에 있고 v2~v5가 같다. 다른 것은 학습 데이터뿐이다.
사용: python develop/v4_train.py
     짧은 실행 확인: python develop/v4_train.py --out <임시 폴더> --epochs 1 --fraction 0.05 --no-export
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (공통 함수)
from paths import data_dir, train_dir, model_dir
from detect import train_yolo

N = 4

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=str(data_dir(N))); ap.add_argument("--out", default=str(train_dir(N)))
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--fraction", type=float, default=1.0, help="학습셋 중 쓸 비율(짧은 실행 확인용)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no-export", action="store_true", help="models/ 로 복사하지 않음(짧은 실행 확인용)"); a = ap.parse_args()
    train_yolo(a.data, a.out, epochs=a.epochs, seed=a.seed, fraction=a.fraction, export_dir=None if a.no_export else model_dir(N))
