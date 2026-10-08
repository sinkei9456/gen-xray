"""v4 합성 2,000장 평가: 검증셋 200장에서 α·β를 정하고 → 테스트셋 1(1,870장) 정량 평가 → 테스트셋 2(107장) 판정.
평가 절차는 detect.evaluate(v2~v5 공통). α·β는 models/ 의 모델 폴더에도 thresholds.json 으로 저장한다(release/predict.py 가 씀).
사용: python develop/v4_eval.py → runs/v4_eval/   (다른 가중치: --weights 경로)
"""
import argparse, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (공통 함수)
from paths import data_dir, eval_dir, model_dir, OUTPUT
from detect import evaluate

N = 4

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", default=str(model_dir(N) / "best.pt")); ap.add_argument("--val", default=str(data_dir(N) / "val" / "images"))
    ap.add_argument("--out", default=str(eval_dir(N))); a = ap.parse_args()
    evaluate(a.weights, a.val, a.out)
    if Path(a.weights).resolve() == (model_dir(N) / "best.pt").resolve(): shutil.copy(Path(a.out) / "thresholds.json", model_dir(N) / "thresholds.json")
