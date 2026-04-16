"""
多模型综合性能对比与表格可视化脚本
vis_metrics_table.py

功能:
1. 在统一的测试集上运行多个不同的模型 (全面兼容基础 LSTM/TCN、无CRF消融模型 以及 高级Attn-CRF)。
2. 收集每个序列的 Acc@7, Acc@3, Acc@0, BoundF1@7, Edit↑ Score, F1@0.70, OSE。
3. 计算各项指标的 均值 ± 标准差 (Mean ± Std)。
4. 渲染并导出一张符合 IEEE RAL 规范的高清学术三线表 (突出显示最后一行结果)。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm

# =========================================================================
# 模块导入区域 (请确保路径与您的工程结构一致)
# =========================================================================
from config import resample, without_quat
from load_data import load_test_state, load_test_label, get_test_demo_id_list, load_specific_test_state, load_specific_test_label
from segmentation_metrics import SegmentationEvaluator

# 1. 基础 LSTM 模型
from LSTM_seg_train import SequenceLabelingLSTM, SequenceLabelingLSTM_CRF
# 2. 带有注意力机制的高级模型 (带 CRF)
from visualize_predictions import AdaptiveSequenceLabelingLSTM_CRF 

# 3. 多阶段 TCN 模型 (TeCNO)
try:
    from TeCNO_seg_train import TeCNO
except ImportError:
    TeCNO = None

# 4. 基础 TCN 模型
try:
    from TCN_seg_train import SequenceLabelingTCN
except ImportError:
    SequenceLabelingTCN = None

# 设置字体族为标准无衬线字体 (接近学术论文中的 Arial/Helvetica)
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =========================================================================
# 0. 【新增】内联定义无 CRF 版本的注意力模型 (避免循环导入或文件缺失)
# =========================================================================
class AdaptiveBimanualSpatialAttention(nn.Module):
    def __init__(self, input_dim=8, proj_dim=16, use_causal_conv=False):
        super(AdaptiveBimanualSpatialAttention, self).__init__()
        self.use_causal_conv = use_causal_conv
        if self.use_causal_conv:
            self.shared_causal_conv = nn.Conv1d(in_channels=input_dim, out_channels=proj_dim, kernel_size=3)
            self.layer_norm = nn.LayerNorm(16)
        else:
            self.shared_proj = nn.Linear(input_dim, proj_dim)
            
        self.dominance_mlp = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim), nn.ReLU(), nn.Linear(proj_dim, 2)
        )

    def forward(self, x):
        X_L, X_R = x[:, :, :8], x[:, :, 8:]
        if self.use_causal_conv:
            H_L = F.relu(self.shared_causal_conv(F.pad(X_L.transpose(1, 2), (2, 0)))).transpose(1, 2)
            H_R = F.relu(self.shared_causal_conv(F.pad(X_R.transpose(1, 2), (2, 0)))).transpose(1, 2)
        else:
            H_L, H_R = F.relu(self.shared_proj(X_L)), F.relu(self.shared_proj(X_R))

        H_cat = torch.cat([H_L, H_R], dim=-1)
        alphas = F.softmax(self.dominance_mlp(H_cat), dim=-1)
        tilde_X = torch.cat([alphas[:, :, 0].unsqueeze(-1) * X_L, alphas[:, :, 1].unsqueeze(-1) * X_R], dim=-1)
        return self.layer_norm(x + tilde_X) if self.use_causal_conv else tilde_X, alphas

class AdaptiveSequenceLabelingLSTM_Attn(nn.Module):
    """消融实验专用：带双臂注意力，但无 CRF 约束的纯分类 LSTM"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, proj_dim=16, use_causal_conv=False):
        super(AdaptiveSequenceLabelingLSTM_Attn, self).__init__()
        self.spatial_attention = AdaptiveBimanualSpatialAttention(input_dim=8, proj_dim=proj_dim, use_causal_conv=use_causal_conv)
        self.lstm = nn.LSTM(input_size=16, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths, return_alphas=False):
        tilde_X, alphas = self.spatial_attention(x)
        packed_input = pack_padded_sequence(tilde_X, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        logits = self.fc(lstm_out)
        return (logits, alphas) if return_alphas else logits


# =========================================================================
# 1. 测试集定义
# =========================================================================
class TestDataset(Dataset):
    def __init__(self):
        demo_id_list = [35, 34, 65, 138, 4, 3, 71, 90, 131, 54, 140, 43, 80, 81, 92, 109, 112, 117, 122, 144, 145, 200]
        states = load_specific_test_state(shuffle=False, without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
        labels = load_specific_test_label(demo_id_list=demo_id_list)
        
        self.samples = []
        for state_seq, label_seq in zip(states, labels):
            state_tensor = torch.tensor(state_seq, dtype=torch.float32)
            label_tensor = torch.tensor(label_seq, dtype=torch.long)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(1)
            self.samples.append((state_tensor, label_tensor))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# =========================================================================
# 2. 动态模型加载器
# =========================================================================
def load_model_dynamically(filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    cfg = checkpoint.get('model_config', {})
    state_dict = checkpoint['model_state_dict']
    
    is_tecno = 'model_type' in checkpoint and 'TeCNO' in checkpoint['model_type']
    has_spatial_attn = any(k.startswith('spatial_attention.') for k in state_dict.keys())
    has_crf = any(k.startswith('crf.') for k in state_dict.keys())
    has_base_tcn = any(k.startswith('tcn.') for k in state_dict.keys()) 
    
    if is_tecno and TeCNO is not None:
        model = TeCNO(
            input_size=cfg.get('input_size', 16), hidden_size=cfg.get('hidden_size', 64),
            num_layers=cfg.get('num_layers', 6), num_classes=cfg.get('num_classes', 7), num_stages=cfg.get('num_stages', 2)
        )
    elif has_spatial_attn and has_crf:
        # 带注意力且带 CRF
        use_causal_conv = "spatial_attention.shared_causal_conv.weight" in state_dict.keys()
        model = AdaptiveSequenceLabelingLSTM_CRF(
            input_size=cfg.get('input_size', 16), hidden_size=cfg.get('hidden_size', 256),
            num_layers=cfg.get('num_layers', 3), num_classes=cfg.get('num_classes', 7),
            proj_dim=cfg.get('proj_dim', 16), use_causal_conv=use_causal_conv
        )
    elif has_spatial_attn and not has_crf:
        # 【新增】带注意力但无 CRF (消融模型)
        use_causal_conv = "spatial_attention.shared_causal_conv.weight" in state_dict.keys()
        model = AdaptiveSequenceLabelingLSTM_Attn(
            input_size=cfg.get('input_size', 16), hidden_size=cfg.get('hidden_size', 256),
            num_layers=cfg.get('num_layers', 3), num_classes=cfg.get('num_classes', 7),
            proj_dim=cfg.get('proj_dim', 16), use_causal_conv=use_causal_conv
        )
    elif has_crf:
        model = SequenceLabelingLSTM_CRF(
            input_size=cfg.get('input_size', 16), hidden_size=cfg.get('hidden_size', 256),
            num_layers=cfg.get('num_layers', 3), num_classes=cfg.get('num_classes', 7)
        )
    elif has_base_tcn:
        if SequenceLabelingTCN is None: raise ImportError("缺失 SequenceLabelingTCN")
        num_channels = cfg.get('num_channels', [64, 64, 64, 64, 128])
        try:
            model = SequenceLabelingTCN(input_size=cfg.get('input_size', 16), num_channels=num_channels, num_classes=cfg.get('num_classes', 7))
        except TypeError:
            model = SequenceLabelingTCN() 
    else:
        model = SequenceLabelingLSTM(
            input_size=cfg.get('input_size', 16), hidden_size=cfg.get('hidden_size', 256),
            num_layers=cfg.get('num_layers', 3), num_classes=cfg.get('num_classes', 7)
        )
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# =========================================================================
# 3. 核心评估与聚合逻辑 
# =========================================================================
def evaluate_single_model(model, dataset, device):
    evaluator = SegmentationEvaluator()
    
    metrics = {
        'Acc@7f↑': [], 'Acc@3f↑': [], 'Acc@0f↑': [], 
        'Bound-F1@10f↑': [], 'IoU-F1@50%↑': [], 'Edit↑': [], 'OSE↓': []
    }
    
    with torch.no_grad():
        for sequence, true_labels in tqdm(dataset, desc="Evaluating", leave=False):
            seq_tensor = sequence.to(device)
            lengths = [seq_tensor.shape[0]]
            
            # 严格判断是否含有 CRF 约束层
            if hasattr(model, 'crf'):
                decode_out = model.decode(seq_tensor.unsqueeze(0), lengths)
                pred_path = decode_out[0][0] if isinstance(decode_out, tuple) else decode_out[0]
                pred_np = np.array(pred_path, dtype=int)
            else:
                outputs = model(seq_tensor.unsqueeze(0), lengths)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                pred_np = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
                
            true_np = true_labels.numpy()
            min_len = min(len(true_np), len(pred_np))
            true_np, pred_np = true_np[:min_len], pred_np[:min_len]
            valid_idx = np.where(true_np != -1)[0]
            if len(valid_idx) == 0: continue
            
            start_idx, end_idx = valid_idx[0], valid_idx[-1]
            true_np = true_np[start_idx:end_idx+1]
            pred_np = pred_np[start_idx:end_idx+1]
            if np.all(true_np == -1): continue
            
            res = evaluator.evaluate(true_np, pred_np, tau=10, segmental_thresholds=[0.50])
            
            metrics['Acc@7f↑'].append(res.frame_acc_tau7 * 100)
            metrics['Acc@3f↑'].append(res.frame_acc_tau3 * 100)
            metrics['Acc@0f↑'].append(res.frame_accuracy * 100)
            metrics['Bound-F1@10f↑'].append(res.boundary_f1.f1 * 100)
            metrics['IoU-F1@50%↑'].append(res.segmental_f1.f1_at_k[0.5] * 100)
            metrics['Edit↑'].append(res.edit_score.score)
            metrics['OSE↓'].append(res.oversegmentation_err)
            
    return metrics

def format_mean_std(metrics_dict):
    formatted = {}
    for key, values in metrics_dict.items():
        if len(values) == 0:
            formatted[key] = "N/A"
            continue
        mean_val = np.mean(values)
        std_val = np.std(values)
        if key == 'OSE↓': formatted[key] = f"{mean_val:.3f} ± {std_val:.3f}"
        else: formatted[key] = f"{mean_val:.1f} ± {std_val:.1f}"
    return formatted


# =========================================================================
# 4. 学术三线表绘制 (严格 RAL 期刊规范)
# =========================================================================
def plot_comparison_table(results_data, columns, save_path):
    model_names = [item[0] for item in results_data]
    cell_data = []
    for _, metrics in results_data:
        row = [metrics.get(col, "N/A") for col in columns]
        cell_data.append(row)
        
    full_cols = ['Model'] + columns
    full_data = [[name] + row for name, row in zip(model_names, cell_data)]
    n_rows, n_cols = len(full_data), len(full_cols)
    
    fig, ax = plt.subplots(figsize=(n_cols * 1.5, n_rows * 0.4 + 0.5))
    ax.axis('off')
    
    tbl = ax.table(cellText=full_data, colLabels=full_cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8) 
    
    for (i, j), cell in tbl.get_celld().items():
        cell.set_facecolor('white')
        cell.set_text_props(color='black')
        cell.visible_edges = ''
        
        if i == 0:
            cell.visible_edges = 'BT'
            cell.set_linewidth(1.5)
            cell.set_text_props(fontweight='bold')
        elif i == n_rows:
            cell.visible_edges = 'B'
            cell.set_linewidth(1.5)
            cell.set_text_props(fontweight='bold')
            
        if j == 0 and i > 0:
            cell.set_text_props(fontweight='bold')
                
    plt.tight_layout(pad=0.1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n📊 标准学术三线表已生成并保存至: {save_path}")


# =========================================================================
# 5. 主程序入口
# =========================================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dir_path = os.path.dirname(__file__)
    
    # 【全面收录】加入了所有的消融版本与 Baseline
    MODELS_ZOO = [
        {
            "name": "LSTM",
            "path": os.path.join(dir_path, "LSTM_model", "lstm_sequence_model.pth") 
        },
        {
            "name": "TCN",
            "path": os.path.join(dir_path, "TCN_model", "tcn_sequence_model.pth")
        },
        {
            "name": "Multi-stage TCN",
            "path": os.path.join(dir_path, "TeCNO_model", "tecno_sequence_model.pth")
        },
        {
            "name": "BiWeight-LSTM",
            "path": os.path.join(dir_path, "LSTM_model", "spatial_attn_lstmnocrf_sequence_model.pth")
        },
        {
            "name": "LSTM-CRF",
            "path": os.path.join(dir_path, "LSTM_model", "lstmcrf_sequence_model.pth")
        },
        
        {
            "name": "BiWeight-LSTM-CRF",
            "path": os.path.join(dir_path, "LSTM_model", "spatial_attn_lstmcrf_sequence_model2.pth")
        }
    ]
    
    print("加载测试数据集...")
    dataset = TestDataset()
    print(f"有效序列总数: {len(dataset)}")
    
    columns = ['Acc@7f↑', 'Acc@3f↑', 'Acc@0f↑', 'Bound-F1@10f↑', 'IoU-F1@50%↑', 'Edit↑', 'OSE↓']
    aggregated_results = []
    
    print("\n🚀 启动多模型性能综合评估...\n")
    
    for m_info in MODELS_ZOO:
        m_name = m_info["name"]
        m_path = m_info["path"]
        
        if not os.path.exists(m_path):
            print(f"⚠️ 文件缺失，跳过模型: {m_name}")
            continue
            
        print(f"评估中: {m_name}")
        model = load_model_dynamically(m_path, device)
        
        raw_metrics = evaluate_single_model(model, dataset, device)
        formatted_metrics = format_mean_std(raw_metrics)
        aggregated_results.append((m_name, formatted_metrics))
        

    if not aggregated_results:
        print("❌ 未捕获有效评估数据，请核对 MODELS_ZOO 权重路径配置。")
        exit()

    save_path = os.path.join(dir_path, os.pardir, "Essay_image_results", "Metrics_Comparison_Table.png")
    plot_comparison_table(aggregated_results, columns, save_path)