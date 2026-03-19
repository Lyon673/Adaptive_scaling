"""
realtime_phase_predictor.py
============================
将 maincb 每帧获得的机器人状态包装成 LSTM / LSTM-CRF 网络所需的输入格式，
并以在线增量方式执行外科手术阶段（phase）预测。

特征向量格式（与 Dataset/transform_state.py 完全一致，16 维）：
  [0:3]  left  position  (x, y, z)
  [3:6]  left  velocity3 (vx, vy, vz)  —— Savitzky-Golay 平滑
  [6]    left  velocity  (scalar speed) —— Savitzky-Golay 平滑
  [7]    left  gripper   (0 / 1)
  [8:11] right position  (x, y, z)
  [11:14]right velocity3 (vx, vy, vz)  —— Savitzky-Golay 平滑
  [14]   right velocity  (scalar speed) —— Savitzky-Golay 平滑
  [15]   right gripper   (0 / 1)

归一化：与训练时相同的 scaler，从 checkpoint 中读取或回退到从训练集重新拟合。

用法（在 main.py 中）：
    # 初始化（仅一次）
    self.phase_predictor = PhasePredictor(model_path='Trajectory/LSTM_model/lstm_sequence_model.pth')

    # 每帧调用（在 maincb 中）
    phase_label, phase_probs = self.phase_predictor.update(
        L_pos3    = Lpsm_position3,
        R_pos3    = Rpsm_position3,
        L_gripper = 1.0 if Lgripper_edge_list[-1] == 1 else 0.0,
        R_gripper = 1.0 if Rgripper_edge_list[-1] == 1 else 0.0,
    )
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

from LSTM_seg_train import (
    SequenceLabelingLSTM,
    SequenceLabelingLSTM_CRF,
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES,
)
from load_data import (
    VELSCALAR_COLS, POS_COL_GROUPS, VEL3_COL_GROUPS,
    load_demonstrations_state, _scale_demos, ratio,
)

# ── 手术阶段名称 ───────────────────────────────────────────────────────────────
PHASE_NAMES = [
    'P0 Right Move',
    'P1 Pick Needle',
    'P2 Right Move2',
    'P3 Pass Needle',
    'P4 Left Move',
    'P5 Left Pick',
    'P6 Pull Thread',
]


# ─────────────────────────────────────────────────────────────────────────────
# 实时 Savitzky-Golay 速度滤波器（与 transform_state.py 一致）
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# PhasePredictor
# ─────────────────────────────────────────────────────────────────────────────

class PhasePredictor:
    """
    实时手术阶段预测器。

    Parameters
    ----------
    model_path : str
        LSTM / LSTM-CRF checkpoint 路径（由 LSTM_seg_train.save_model 生成）。
    device : str | torch.device
        推理设备，默认自动选择 CUDA / CPU。
    min_frames : int
        累积至少多少帧后才开始返回网络预测（之前返回 -1 表示"热身期"）。
    sg_window : int
        速度平滑 SG 窗口长度（奇数，默认 9）。
    sg_poly : int
        SG 多项式阶数（默认 3）。
    demo_id_list_for_scalers : array-like | None
        若 checkpoint 中不含 scaler，则用此 demo 列表重新拟合。
        为 None 时使用默认的 139 个有效 demo。
    """

    def __init__(
        self,
        model_path: str,
        device: str | torch.device | None = None,
        min_frames: int = 30,
        sg_window: int = 9,
        sg_poly:   int = 3,
        demo_id_list_for_scalers=None,
    ):
        self.device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                       if device is None else torch.device(device))
        self.min_frames = min_frames

        # ── 加载模型 ─────────────────────────────────────────────────────────
        self.model, self.is_crf, self.without_quat, self._scalers = \
            self._load_checkpoint(model_path, demo_id_list_for_scalers)
        self.model.eval()

        # ── 特征维度 ──────────────────────────────────────────────────────────
        self._feat_dim = 8 if self.without_quat else 16

        # ── 速度滤波器（与 transform_state.py 一致）──────────────────────────
        # per-axis velocity3 滤波器：左(3) + 右(3)
        self._l_vel3_f = [_SGFilter(sg_window, sg_poly) for _ in range(3)]
        self._r_vel3_f = [_SGFilter(sg_window, sg_poly) for _ in range(3)]
        # scalar speed 滤波器
        self._l_spd_f  = _SGFilter(sg_window, sg_poly)
        self._r_spd_f  = _SGFilter(sg_window, sg_poly)

        # ── 上一帧位置（用于差分求速度）─────────────────────────────────────
        self._prev_L_pos: np.ndarray | None = None
        self._prev_R_pos: np.ndarray | None = None

        # ── 特征历史（每帧追加，整体送入 LSTM）───────────────────────────────
        self._history: list[np.ndarray] = []

        # ── 最近预测结果 ──────────────────────────────────────────────────────
        self.last_phase: int = -1
        self.last_probs: np.ndarray = np.zeros(NUM_CLASSES)

        print(f"[PhasePredictor] 模型已加载  is_crf={self.is_crf}  "
              f"without_quat={self.without_quat}  device={self.device}")

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def update(
        self,
        L_pos3:    np.ndarray,
        R_pos3:    np.ndarray,
        L_gripper: float,
        R_gripper: float,
    ) -> tuple[int, np.ndarray]:
        """
        处理一帧新数据并返回预测结果。

        Parameters
        ----------
        L_pos3    : shape (3,)  左臂末端三维位置 (x, y, z)
        R_pos3    : shape (3,)  右臂末端三维位置 (x, y, z)
        L_gripper : float  左夹爪状态 (0.0 / 1.0)
        R_gripper : float  右夹爪状态 (0.0 / 1.0)

        Returns
        -------
        phase_label : int
            预测的手术阶段标签 (0-6)；热身期（< min_frames）返回 -1。
        phase_probs : np.ndarray shape (NUM_CLASSES,)
            各阶段的 softmax 概率（LSTM-CRF 返回全零向量）。
        """
        raw_feat = self._compute_raw_feature(L_pos3, R_pos3, L_gripper, R_gripper)
        scaled   = self._scale_feature(raw_feat)
        self._history.append(scaled)

        if len(self._history) < self.min_frames:
            self.last_phase = -1
            self.last_probs = np.zeros(NUM_CLASSES)
            return -1, self.last_probs

        phase, probs = self._infer()
        self.last_phase = phase
        self.last_probs = probs
        return phase, probs

    def reset(self):
        """重置历史缓冲区（切换任务时调用）。"""
        self._history.clear()
        self._prev_L_pos = None
        self._prev_R_pos = None
        for f in self._l_vel3_f + self._r_vel3_f:
            f.buf.clear()
        self._l_spd_f.buf.clear()
        self._r_spd_f.buf.clear()
        self.last_phase = -1
        self.last_probs = np.zeros(NUM_CLASSES)

    @property
    def phase_name(self) -> str:
        """当前预测阶段的名称字符串。"""
        if self.last_phase < 0:
            return 'Warming up…'
        return PHASE_NAMES[self.last_phase] if self.last_phase < len(PHASE_NAMES) else f'Phase {self.last_phase}'

    # ── 内部：特征计算 ────────────────────────────────────────────────────────

    def _compute_raw_feature(
        self,
        L_pos3:    np.ndarray,
        R_pos3:    np.ndarray,
        L_gripper: float,
        R_gripper: float,
    ) -> np.ndarray:
        """
        按照与 transform_state.py 完全相同的方式计算原始（未归一化）16 维特征。
        """
        L_pos3 = np.asarray(L_pos3, dtype=float)
        R_pos3 = np.asarray(R_pos3, dtype=float)

        # ── 差分速度（第一帧视为静止）────────────────────────────────────────
        if self._prev_L_pos is None:
            L_vel3_raw = np.zeros(3)
            R_vel3_raw = np.zeros(3)
        else:
            L_vel3_raw = L_pos3 - self._prev_L_pos
            R_vel3_raw = R_pos3 - self._prev_R_pos
        self._prev_L_pos = L_pos3.copy()
        self._prev_R_pos = R_pos3.copy()

        # ── 标量速度（L2 模长）────────────────────────────────────────────────
        L_spd_raw = float(np.linalg.norm(L_vel3_raw))
        R_spd_raw = float(np.linalg.norm(R_vel3_raw))

        # ── SG 平滑 ───────────────────────────────────────────────────────────
        L_vel3 = np.array([f.update(v) for f, v in zip(self._l_vel3_f, L_vel3_raw)])
        R_vel3 = np.array([f.update(v) for f, v in zip(self._r_vel3_f, R_vel3_raw)])
        L_spd  = max(0.0, self._l_spd_f.update(L_spd_raw))
        R_spd  = max(0.0, self._r_spd_f.update(R_spd_raw))

        # ── 拼接 16 维特征 ────────────────────────────────────────────────────
        feat_16 = np.concatenate([
            L_pos3,               # [0:3]
            L_vel3,               # [3:6]
            [L_spd],              # [6]
            [float(L_gripper)],   # [7]
            R_pos3,               # [8:11]
            R_vel3,               # [11:14]
            [R_spd],              # [14]
            [float(R_gripper)],   # [15]
        ])

        # ── without_quat=True → 只取 8 列 ────────────────────────────────────
        if self.without_quat:
            return feat_16[[0, 1, 2, 7, 8, 9, 10, 15]]
        return feat_16

    def _scale_feature(self, feat: np.ndarray) -> np.ndarray:
        """应用与训练时相同的归一化 scaler。"""
        out = feat.copy().reshape(1, -1)   # (1, D)

        sc = self._scalers
        # velocity scalar
        out[:, VELSCALAR_COLS] = sc['vel_scalar'].transform(out[:, VELSCALAR_COLS])
        # position
        for group, s in zip(POS_COL_GROUPS, sc['pos']):
            out[:, group] = s.transform(out[:, group])
        # velocity3
        for group, s in zip(VEL3_COL_GROUPS, sc['vel3']):
            out[:, group] = s.transform(out[:, group])

        return out.flatten()

    # ── 内部：推理 ────────────────────────────────────────────────────────────

    def _infer(self) -> tuple[int, np.ndarray]:
        """
        将完整历史送入网络，返回最后一帧的预测标签和概率。
        LSTM-CRF 使用 Viterbi decode；标准 LSTM 使用 argmax。
        """
        seq = np.stack(self._history, axis=0)           # (T, D)
        seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        lengths = [seq.shape[0]]

        with torch.no_grad():
            
            paths = self.model.decode(seq_t, lengths)
            label = int(paths[0][-1])
        
            logits = self.model(seq_t, lengths)      # (1, T, C)
            probs  = torch.softmax(logits[0, -1, :], dim=0).cpu().numpy()

        return label, probs

    # ── 内部：加载 checkpoint ─────────────────────────────────────────────────

    def _load_checkpoint(self, path: str, demo_id_list_for_scalers):
        ckpt = torch.load(path, map_location=self.device)
        cfg  = ckpt['model_config']

        is_crf       = any(k.startswith('crf.') for k in ckpt['model_state_dict'])
        without_quat = ckpt.get('without_quat', False)

        if is_crf:
            from torchcrf import CRF   # noqa: F401 (imported for side-effects)
            model = SequenceLabelingLSTM_CRF(
                cfg['input_size'], cfg['hidden_size'],
                cfg['num_layers'], cfg['num_classes'],
            )
        else:
            model = SequenceLabelingLSTM(
                cfg['input_size'], cfg['hidden_size'],
                cfg['num_layers'], cfg['num_classes'],
            )

        model.load_state_dict(ckpt['model_state_dict'])
        model.to(self.device)

        # ── scaler：优先从 checkpoint 加载，否则回退到重新拟合 ─────────────
        if 'scalers' in ckpt:
            scalers = ckpt['scalers']
            print("[PhasePredictor] Scalers 从 checkpoint 加载成功。")
        else:
            print("[PhasePredictor] Checkpoint 中无 scalers，从训练数据重新拟合 …")
            scalers = self._fit_scalers_from_training(demo_id_list_for_scalers)

        return model, is_crf, without_quat, scalers

    @staticmethod
    def _fit_scalers_from_training(demo_id_list=None) -> dict:
        """从训练集数据重新拟合 scaler（回退方案）。"""
        excluded = [80, 81, 92, 109, 112, 117, 122, 144, 145]
        if demo_id_list is None:
            demo_id_list = np.arange(148)
            demo_id_list = np.delete(demo_id_list, excluded)

        from Trajectory.config import without_quat as _wq, resample as _rs  # noqa
        raw_all = load_demonstrations_state(
            shuffle=True, without_quat=_wq,
            resample=_rs, demo_id_list=demo_id_list,
        )
        bound = round(ratio * len(raw_all))
        _, scalers = _scale_demos(raw_all[:bound])
        return scalers
