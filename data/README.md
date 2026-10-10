# data/

이 폴더에는 데이터가 들어 있지 않다(`test_set_counts.json`만 있음). / No data is included here except `test_set_counts.json`.

| 넣을 것 / What to add | 어디서 / Source | 자세히 / Details |
|---|---|---|
| `dataset/` (KAMP 원본 X-ray BMP 2,809장, 레이블 500개) | KAMP, https://www.kamp-ai.kr/ (재배포하지 않음 / not redistributed) | docs/manual_ko.md 3절 / README.md "Data" |
| `v5_synth5000/` (합성 5,000장 / 5,000 synthetic images) | Releases `synthetic_v5_5000.zip` | docs/manual_ko.md 1절 / README.md "Release files" |

`material_500/`, `v2_real500/`, `test1/`, `test2/`, `v3_synth500/`, `v4_synth2000/` 는 코드가 원본·합성 데이터에서 다시 만든다. / These are rebuilt by the code from the raw and synthetic data.
