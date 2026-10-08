"""합성 5,000장(v5, 시드 0 한 번 생성)에서 앞부분을 잘라 v3·v4의 학습셋·검증셋을 만든다. v5는 5,000장 전체(학습 4,500 / 검증 500)를 그대로 쓴다.
- v3 앞 500장: 1~400번째 학습셋, 401~500번째 검증셋
- v4 앞 2,000장: 1~1,800번째 학습셋, 1,801~2,000번째 검증셋
세 세트는 같은 생성 결과의 앞부분이라 차이는 양뿐이다. 모델마다 자기 학습셋과 검증셋은 겹치지 않는다(v3의 검증셋은 v4·v5의 학습셋에 들어 있음).
사용: python develop/make_subsets.py → data/v3_synth500, data/v4_synth2000
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (공통 함수)
from paths import DATA, data_dir

SETS = {data_dir(3).name: (500, 100), data_dir(4).name: (2000, 200)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default=str(data_dir(5))); ap.add_argument("--out", default=str(DATA), help="만들 위치"); a = ap.parse_args()
    src = Path(a.src); meta = pd.read_csv(src / "meta.csv"); root = Path(a.out)
    for name, (n, n_val) in SETS.items():
        out = root / name
        if out.exists(): shutil.rmtree(out)
        sub = meta.iloc[:n].copy(); sub["split"] = ["val" if i >= n - n_val else "train" for i in range(n)]
        for sp in ("train", "val"): (out / sp / "images").mkdir(parents=True); (out / sp / "labels").mkdir(parents=True)
        for _, r in sub.iterrows():
            old = meta.loc[meta.image == r.image, "split"].iloc[0]
            shutil.copy(src / old / "images" / f"{r.image}.png", out / r.split / "images" / f"{r.image}.png")
            shutil.copy(src / old / "labels" / f"{r.image}.txt", out / r.split / "labels" / f"{r.image}.txt")
        sub.to_csv(out / "meta.csv", index=False); g = json.load(open(src / "generation.json")); g.update(subset_of="합성 5,000장(v5_synth5000)", first_n=n, n_val=n_val)
        json.dump(g, open(out / "generation.json", "w"), ensure_ascii=False, indent=1)
        print(name, sub.groupby("split").agg(images=("image", "size"), defects=("n_defects", "sum"), shapes=("n_shapes", "sum")).to_dict("index"))


if __name__ == "__main__":
    main()
