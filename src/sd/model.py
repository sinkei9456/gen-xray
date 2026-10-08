from __future__ import annotations
import torch, numpy as np
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler


def build_unet(size: int = 256) -> UNet2DModel:
    return UNet2DModel(sample_size=size, in_channels=3, out_channels=1, layers_per_block=2,
                       block_out_channels=(64, 128, 256, 384),
                       down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),   # 어텐션은 32² 에서만 (64² 어텐션은 비용 4배)
                       up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"), norm_num_groups=32)


def make_cond(layout_u8: np.ndarray | torch.Tensor, machine_idx) -> torch.Tensor:
    """배치도(그 밖 0, 봉지 128, 막대 255) + 장비 → cond 텐서 (B,2,H,W) in [-1,1]"""
    lay = torch.as_tensor(np.asarray(layout_u8)).float() / 127.5 - 1.0
    if lay.ndim == 2: lay = lay[None]
    m = torch.as_tensor(np.asarray(machine_idx)).float().view(-1, 1, 1, 1).expand(-1, 1, *lay.shape[-2:]) - 1.0   # 0,1,2 -> -1,0,1
    return torch.cat([lay[:, None], m], 1)


def train_scheduler(): return DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")
def sample_scheduler(): return DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=True)


@torch.no_grad()
def sample(unet: UNet2DModel, cond: torch.Tensor, steps: int = 50, generator=None, device="cuda",
           init: torch.Tensor | None = None, t_start: int = 700) -> np.ndarray:
    """cond (B,2,H,W) → 이미지 uint8 (B,H,W).

    init 이 주어지면 순수 잡음이 아니라 init 에 t_start 만큼 잡음을 섞은 곳에서 시작한다(img2img/SDEdit 방식).
    지금 결과는 init = 장비 바탕 밝기만 채운 빈 화면, t_start = 999(가장 깊게: 빈 화면의 밝기는 거의 남지 않고 봉지·막대는 배치도만 보고 그린다).
    init: (B,1,H,W) in [-1,1]
    """
    sch = sample_scheduler(); sch.set_timesteps(steps)
    B, _, H, W = cond.shape; cond = cond.to(device)
    if init is None:
        x = torch.randn((B, 1, H, W), generator=generator, device=device); ts = list(sch.timesteps)
    else:
        ts = [t for t in sch.timesteps if int(t) <= t_start]; t0 = torch.full((B,), int(ts[0]), device=device)
        x = train_scheduler().add_noise(init.to(device), torch.randn((B, 1, H, W), generator=generator, device=device), t0)
    for t in ts:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            eps = unet(torch.cat([x, cond], 1), t).sample.float()
        x = sch.step(eps, t, x).prev_sample
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte()[:, 0].cpu().numpy()
