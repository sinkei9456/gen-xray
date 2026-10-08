"""generate(): 이전 방법(절차적 생성기, Renderer)으로 지시어/Spec → n장 생성. 지금 결과에는 쓰지 않는다(지금은 synth/compose.py)."""
from __future__ import annotations
import os, json, argparse
from pathlib import Path
from typing import Union, List, Optional
import numpy as np, cv2, pandas as pd
from .layout import Spec, parse_prompt, spec_from_dict
from .render import Renderer, yolo_lines, CLS_DEFECT, CLS_STICK, CLS_BAG


def generate(prompt_or_spec: Union[str, Spec, dict, List], n: int = 10, out_dir: Union[str, Path] = "synthetic_out", seed: int = 0,
             renderer: Optional[Renderer] = None, use_real_dots: bool = False, label_classes=(CLS_DEFECT,), prefix: str = "syn") -> pd.DataFrame:
    """label_classes: YOLO txt에 쓸 클래스. (0,)=이물만, (0,1,2)=이물+막대+봉지(이전 실험용)."""
    out = Path(out_dir); (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(parents=True, exist_ok=True)
    items = prompt_or_spec if isinstance(prompt_or_spec, list) else [prompt_or_spec]
    specs = [parse_prompt(x) if isinstance(x, str) else (spec_from_dict(x) if isinstance(x, dict) else x) for x in items]
    R = renderer or Renderer(seed=seed)
    rows = []; k = 0
    for i in range(n):
        spec = specs[i % len(specs)]
        img, labels, meta = R.render(spec, use_real_dots=use_real_dots)
        H, W = img.shape; name = f"{prefix}_{seed:03d}_{k:05d}"; k += 1
        cv2.imwrite(str(out / "images" / f"{name}.png"), img)
        with open(out / "labels" / f"{name}.txt", "w") as f:
            f.write("\n".join(yolo_lines(labels, W, H, label_classes)) + ("\n" if labels else ""))
        base = dict(image=name, W=W, H=H, **meta, spec=spec.to_json().replace("\n", " "))
        defects = [L for L in labels if L["cls"] == CLS_DEFECT]
        if defects:
            for L in defects:
                x0, y0, x1, y1 = L["box"]; rows.append(dict(base, **{kk: vv for kk, vv in L.items() if kk not in ("box", "cls")}, x0=x0, y0=y0, x1=x1, y1=y1))
        else:
            rows.append(dict(base))
    df = pd.DataFrame(rows); df.to_csv(out / "meta.csv", index=False, mode="a", header=not (out / "meta.csv").exists())
    return df


def main():
    ap = argparse.ArgumentParser(description="자연어 지시 → X-ray 합성 이미지+라벨")
    ap.add_argument("prompt", nargs="+", help="지시문 (여러 개 가능, 순환)")
    ap.add_argument("-n", type=int, default=10); ap.add_argument("-o", "--out", default="synthetic_out"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real-dots", action="store_true"); ap.add_argument("--all-classes", action="store_true", help="스틱/봉지 박스도 라벨에 포함")
    a = ap.parse_args()
    df = generate(a.prompt, n=a.n, out_dir=a.out, seed=a.seed, use_real_dots=a.real_dots, label_classes=(0, 1, 2) if a.all_classes else (0,))
    print(df.groupby(["product", "placement"]).size().to_string()); print("→", a.out)


if __name__ == "__main__":
    main()
