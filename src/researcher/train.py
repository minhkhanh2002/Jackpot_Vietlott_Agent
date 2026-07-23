"""Huấn luyện Agent RL (PPO) trên VietlottEnv — môi trường về bản chất là nhiễu thuần tuý.

Mục tiêu NGHIÊN CỨU (không phải tối ưu lợi nhuận, bất khả thi về mặt toán học): xem Agent
có tự học ra thiên vị/"mê tín" hay không khi lặp lại nhiều lượt trên cùng một chuỗi dữ liệu
mà không có tín hiệu thật để học — đo bằng policy_analysis.py sau khi train xong.

Chọn PPO vì hỗ trợ sẵn action space MultiDiscrete của env này. DQN (dự tính ban đầu trong kế
hoạch gốc) KHÔNG tương thích: SB3 DQN chỉ hỗ trợ action space Discrete đơn, trong khi
VietlottEnv cần 6 lựa chọn độc lập mỗi bước (MultiDiscrete([55]*6)) — phát hiện khi bắt tay
code Phase 4, không phải lỗi thiết kế env.

Chạy: python -m src.researcher.train --timesteps 50000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from stable_baselines3 import PPO

from src.data_engine.loader import load_results
from src.envs.prize_calculator import DrawOutcome
from src.envs.vietlott_gym_env import VietlottEnv

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "ppo_power655.zip"


def draws_from_dataframe(df: pd.DataFrame) -> list[DrawOutcome]:
    return [
        DrawOutcome(
            numbers=[row.num_1, row.num_2, row.num_3, row.num_4, row.num_5, row.num_6],
            bonus_number=row.bonus_number,
            jackpot1_amount=row.jackpot1_amount,
            jackpot2_amount=row.jackpot2_amount,
        )
        for row in df.itertuples()
    ]


def train(
    timesteps: int = 50_000,
    window_size: int = 10,
    seed: int = 0,
    save_path: Optional[Path] = None,
) -> PPO:
    df = load_results()
    draws = draws_from_dataframe(df)
    env = VietlottEnv(draws, window_size=window_size)

    model = PPO("MlpPolicy", env, seed=seed, verbose=1)
    model.learn(total_timesteps=timesteps)

    save_path = save_path or DEFAULT_MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent trên VietlottEnv (Power 6/55).")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(timesteps=args.timesteps, window_size=args.window_size, seed=args.seed)


if __name__ == "__main__":
    main()
