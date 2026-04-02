"""
realtime_phase_predictor_tecno.py
==================================
将 maincb 每帧获得的机器人状态包装成 TeCNO（Multi-Stage TCN）网络所需的输入格式，
并以在线增量方式执行外科手术阶段（phase）预测，同时基于 MC Dropout 输出不确定性感知的安全衰减系数。

特征向量格式（与 Dataset/transform_state.py 完全一致，16 维）
... (略)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
from collections import deque
from scipy.signal import savgol_filter

# ── 将 Trajectory 目录加到搜索路径 ────────────────────────────────────────────
_TRAJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trajectory')
if _TRAJ_DIR not in sys.path:
    sys.path.insert(0, _TRAJ_DIR)

from Trajectory.TeCNO_seg_train import (
    TeCNO,
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES,
    NUM_STAGES, KERNEL_SIZE, DROPOUT,
)

from Trajectory.load_data import (
    VELSCALAR_COLS, POS_COL_GROUPS, VEL3_COL_GROUPS,
    load_demonstrations_state, _scale_demos, ratio,
)

PHASE_NAMES = [
    'P0 Right Move',
    'P1 Pick Needle',
    'P2 Right Move2',
    'P3 Pass Needle',
    'P4 Left Move',
    'P5 Left Pick',
    'P6 Pull Thread',
]

class _SGFilter:
    """单通道实时 Savitzky-Golay 平滑器。"""
    def __init__(self, window_length: int = 9, polyorder: int = 3):
        assert window_length % 2 == 1, "window_length 必须为奇数"
        assert polyorder < window_length
        self.wl = window_length
        self.po = polyorder
        self.buf: deque = deque(maxlen=window_length)

    def update(self, value: float) -> float:
        self.buf.append(float(value))
        n = len(self.buf)
        if n <= self.po + 1:
            return value
        wl = n if n % 2 == 1 else n - 1
        wl = min(wl, self.wl)
        smoothed = savgol_filter(np.array(self.buf), wl, self.po, mode='interp')
        return float(smoothed[-1])

class TeCNOPhasePredictor:
    """
    基于 TeCNO（Multi-Stage TCN）与 MC Dropout 的不确定性感知实时预测器。
    """

    def __init__(
        self,
        model_path: str,
        device: str | torch.device | None = None,
        min_frames: int = 10,
        sg_window: int = 9,
        sg_poly:   int = 3,
        max_history: int = 512,
        mc_samples: int = 15,          # [NEW] MC Dropout 采样次数
        entropy_lambda: float = 2.0,   # [NEW] 熵到衰减系数的敏感度超参数
        demo_id_list_for_scalers=None,
    ):
        self.device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                       if device is None else torch.device(device))
        self.min_frames = min_frames
        self.mc_samples = mc_samples
        self.entropy_lambda = entropy_lambda
        self._max_entropy = np.log(NUM_CLASSES)  # 最大理论熵值 ln(7) ≈ 1.946

        # 加载模型
        self.model, self.without_quat, self._scalers = \
            self._load_checkpoint(model_path, demo_id_list_for_scalers)
        self.model.eval()

        self._feat_dim = 8 if self.without_quat else 16

        # 速度滤波器
        self._l_vel3_f = [_SGFilter(sg_window, sg_poly) for _ in range(3)]
        self._r_vel3_f = [_SGFilter(sg_window, sg_poly) for _ in range(3)]
        self._l_spd_f  = _SGFilter(sg_window, sg_poly)
        self._r_spd_f  = _SGFilter(sg_window, sg_poly)

        self._prev_L_pos: np.ndarray | None = None
        self._prev_R_pos: np.ndarray | None = None
        self._history: deque = deque(maxlen=max_history)

        self.last_phase: int = -1
        self.last_probs: np.ndarray = np.zeros(NUM_CLASSES)
        self.last_alpha: float = 1.0  # [NEW] 最近一次的衰减系数

        print(f"[TeCNOPhasePredictor] 模型已加载  "
              f"device={self.device}  max_history={max_history}  mc_samples={mc_samples}")

        print("[TeCNOPhasePredictor] 正在进行 CUDA 预热...")
        dummy_input = torch.zeros((1, 10, self._feat_dim), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            self._enable_dropout()
            _ = self.model(dummy_input, [10])
        print("[TeCNOPhasePredictor] CUDA 预热完成，模型已准备就绪！")

    def _enable_dropout(self):
        """[NEW] 强制开启模型中的所有 Dropout 层，以支持 MC Dropout 推理。"""
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def update(
        self,
        L_pos3:    np.ndarray,
        R_pos3:    np.ndarray,
        L_gripper: float,
        R_gripper: float,
    ) -> tuple[int, np.ndarray, float]:
        """
        处理一帧新数据并返回预测结果及安全衰减系数。

        Returns
        -------
        phase_label : int
            预测的手术阶段标签 (0-6)。
        phase_probs : np.ndarray
            MC Dropout 得到的预测平均概率分布。
        alpha : float
            安全衰减系数 [0, 1]。系统越混乱（熵越大），alpha 越趋近于 0。
        """
        raw_feat = self._compute_raw_feature(L_pos3, R_pos3, L_gripper, R_gripper)
        scaled   = self._scale_feature(raw_feat)
        self._history.append(scaled)

        if len(self._history) < self.min_frames:
            self.last_phase = -1
            self.last_probs = np.zeros(NUM_CLASSES)
            self.last_alpha = 1.0
            return -1, self.last_probs, 1.0

        phase, probs, alpha = self._infer()
        self.last_phase = phase
        self.last_probs = probs
        self.last_alpha = alpha
        return phase, probs, alpha

    def reset(self):
        self._history.clear()
        self._prev_L_pos = None
        self._prev_R_pos = None
        for f in self._l_vel3_f + self._r_vel3_f:
            f.buf.clear()
        self._l_spd_f.buf.clear()
        self._r_spd_f.buf.clear()
        self.last_phase = -1
        self.last_probs = np.zeros(NUM_CLASSES)
        self.last_alpha = 1.0

    @property
    def phase_name(self) -> str:
        if self.last_phase < 0:
            return 'Warming up…'
        return PHASE_NAMES[self.last_phase] if self.last_phase < len(PHASE_NAMES) else f'Phase {self.last_phase}'

    # _compute_raw_feature 和 _scale_feature 方法保持原样...
    def _compute_raw_feature(self, L_pos3, R_pos3, L_gripper, R_gripper):
        L_pos3 = np.asarray(L_pos3, dtype=float)
        R_pos3 = np.asarray(R_pos3, dtype=float)
        if self._prev_L_pos is None:
            L_vel3_raw = np.zeros(3)
            R_vel3_raw = np.zeros(3)
        else:
            L_vel3_raw = L_pos3 - self._prev_L_pos
            R_vel3_raw = R_pos3 - self._prev_R_pos
        self._prev_L_pos = L_pos3.copy()
        self._prev_R_pos = R_pos3.copy()
        L_spd_raw = float(np.linalg.norm(L_vel3_raw))
        R_spd_raw = float(np.linalg.norm(R_vel3_raw))
        L_vel3 = np.array([f.update(v) for f, v in zip(self._l_vel3_f, L_vel3_raw)])
        R_vel3 = np.array([f.update(v) for f, v in zip(self._r_vel3_f, R_vel3_raw)])
        L_spd  = max(0.0, self._l_spd_f.update(L_spd_raw))
        R_spd  = max(0.0, self._r_spd_f.update(R_spd_raw))
        feat_16 = np.concatenate([
            L_pos3, L_vel3, [L_spd], [float(L_gripper)],
            R_pos3, R_vel3, [R_spd], [float(R_gripper)]
        ])
        if self.without_quat:
            return feat_16[[0, 1, 2, 7, 8, 9, 10, 15]]
        return feat_16

    def _scale_feature(self, feat):
        out = feat.copy().reshape(1, -1)
        sc = self._scalers
        out[:, VELSCALAR_COLS] = sc['vel_scalar'].transform(out[:, VELSCALAR_COLS])
        for group, s in zip(POS_COL_GROUPS, sc['pos']):
            out[:, group] = s.transform(out[:, group])
        for group, s in zip(VEL3_COL_GROUPS, sc['vel3']):
            out[:, group] = s.transform(out[:, group])
        return out.flatten()

    def _infer(self) -> tuple[int, np.ndarray, float]:
        """
        [NEW] 执行 MC Dropout 多次采样，计算预测均值与预测熵，并输出安全衰减系数。
        """
        seq = np.stack(list(self._history), axis=0)
        seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        lengths = [seq.shape[0]]

        self.model.eval()        # 确保 BatchNorm 等层处于 eval 模式
        self._enable_dropout()   # 单独强制开启 Dropout

        probs_list = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits = self.model(seq_t, lengths)
                probs = torch.softmax(logits[0, -1, :], dim=0)
                probs_list.append(probs)

        # 1. 计算预测平均分布
        probs_tensor = torch.stack(probs_list, dim=0)        # (K, C)
        mean_probs = probs_tensor.mean(dim=0).cpu().numpy()  # (C,)
        
        # 2. 计算香农熵 (加入 eps 防止 log(0))
        eps = 1e-10
        entropy = -np.sum(mean_probs * np.log(mean_probs + eps))
        
        # 3. 计算安全衰减系数 alpha (指数衰减映射)
        alpha = np.exp(-self.entropy_lambda * (entropy / self._max_entropy))
        
        # 4. 获取最终预测标签
        label = int(mean_probs.argmax())

        return label, mean_probs, float(alpha)

    # _load_checkpoint 和 _fit_scalers_from_training 保持原样...
    def _load_checkpoint(self, path, demo_id_list_for_scalers):
        ckpt = torch.load(path, map_location=self.device)
        cfg  = ckpt['model_config']
        without_quat = ckpt.get('without_quat', False)
        model = TeCNO(
            input_size  = cfg['input_size'], hidden_size = cfg['hidden_size'],
            num_layers  = cfg['num_layers'], num_classes = cfg['num_classes'],
            num_stages  = cfg.get('num_stages',  NUM_STAGES),
            kernel_size = cfg.get('kernel_size', KERNEL_SIZE),
            dropout     = cfg.get('dropout',     DROPOUT),
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(self.device)
        if 'scalers' in ckpt:
            scalers = ckpt['scalers']
        else:
            scalers = self._fit_scalers_from_training(demo_id_list_for_scalers)
        return model, without_quat, scalers

    @staticmethod
    def _fit_scalers_from_training(demo_id_list=None):
        excluded = [80, 81, 92, 109, 112, 117, 122, 144, 145]
        if demo_id_list is None:
            demo_id_list = np.delete(np.arange(148), excluded)
        from Trajectory.config import without_quat as _wq, resample as _rs
        raw_all = load_demonstrations_state(shuffle=True, without_quat=_wq, resample=_rs, demo_id_list=demo_id_list)
        from Trajectory.load_data import ratio
        bound = round(ratio * len(raw_all))
        _, scalers = _scale_demos(raw_all[:bound])
        return scalers