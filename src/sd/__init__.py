"""sd — Diffusion 모델(배치도·장비 번호를 조건으로 받는 U-Net, 처음부터 학습).

model.py: 모델 구조·잡음 일정·샘플링, data.py: 정사각형 채우기·배치도 그리기, sample.py: 합성 이미지의 봉지·막대 그리기(빈 화면에서 출발, 256×256으로 그린 뒤 장비 크기로 되돌림).
학습 재료: 레이블 500장의 봉지·막대 이미지(이물·검사 장비 표시를 지운 것)와 배치도(data/material_500/diffusion_cache_500.npz). 학습은 develop/train_diffusion.py.
"""
