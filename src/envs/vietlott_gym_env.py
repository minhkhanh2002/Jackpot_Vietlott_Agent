"""Power 6/55 Gym Env — môi trường Gymnasium mô phỏng mua vé qua chuỗi kỳ quay lịch sử.

Dùng để huấn luyện/phân tích Agent RL (Phase 4) trong một môi trường về bản chất là
nhiễu thuần tuý (pure noise) — mục tiêu nghiên cứu là xem Agent có tự học ra "sự mê tín"
hay không, KHÔNG phải để tối ưu lợi nhuận thực sự (bất khả thi về mặt toán học, xem README).

Env nhận dữ liệu qua tham số `draws` (danh sách DrawOutcome) thay vì tự đọc DB — tách
biệt khỏi Neon để có thể test với dữ liệu synthetic, và tái dùng được với dữ liệu thật
qua loader.py khi cần.
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

from src.envs.prize_calculator import DrawOutcome, compute_reward, load_prize_rules

NUM_POOL = 55  # Power 6/55: 6 số chính + 1 số Power, mỗi số trong [1, 55]


class VietlottEnv(gym.Env):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        draws: list[DrawOutcome],
        window_size: int = 10,
        rules: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if len(draws) <= window_size:
            raise ValueError("Cần số kỳ quay > window_size để có đủ lịch sử khởi tạo state")

        self.num_pool = NUM_POOL
        self.draws = draws
        self.window_size = window_size
        self.rules = rules or load_prize_rules()
        self.ticket_price = self.rules["ticket_price"]

        # 6 lựa chọn độc lập trong [0, num_pool) — trùng số giữa 6 lựa chọn = vé không hợp lệ
        # (bị compute_reward phạt), đây là hành vi CẦN quan sát khi phân tích Agent (Phase 4),
        # không che giấu bằng action space ràng buộc cứng.
        self.action_space = gym.spaces.MultiDiscrete([self.num_pool] * 6)

        obs_len = self.window_size * self.num_pool + 1  # + jackpot hiện tại đã chuẩn hoá
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

        self._cursor = 0  # index trong self.draws của kỳ SẮP TỚI mà agent đang đặt cược

    def _build_observation(self) -> np.ndarray:
        history = self.draws[self._cursor - self.window_size : self._cursor]
        multi_hot = np.zeros((self.window_size, self.num_pool), dtype=np.float32)
        for i, draw in enumerate(history):
            for n in draw.numbers:
                multi_hot[i, n - 1] = 1.0

        jackpot = self.draws[self._cursor].jackpot1_amount or 0
        # Chuẩn hoá thô theo mốc 100 tỷ VNĐ (jackpot Power 6/55 thực tế quan sát chưa vượt mốc này).
        jackpot_norm = np.array([min(jackpot / 1e11, 1.0)], dtype=np.float32)
        return np.concatenate([multi_hot.flatten(), jackpot_norm])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._cursor = self.window_size
        return self._build_observation(), {}

    def step(self, action):
        ticket_numbers = [int(a) + 1 for a in action]  # action 0-indexed -> số thật 1..num_pool
        draw = self.draws[self._cursor]

        reward = compute_reward(ticket_numbers, draw, self.rules)
        match_count = len(set(ticket_numbers) & set(draw.numbers))

        self._cursor += 1
        terminated = self._cursor >= len(self.draws)
        truncated = False
        obs = (
            self._build_observation()
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )
        info = {"ticket_numbers": ticket_numbers, "match_count": match_count}
        return obs, float(reward), terminated, truncated, info
