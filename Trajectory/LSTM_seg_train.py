import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from load_data import (load_train_state, load_train_label,
                       load_demonstrations_state, _scale_demos,
                       get_shuffled_demo_ids)
from sklearn.metrics import classification_report, accuracy_score
import os
from torchcrf import CRF
from torch.utils.tensorboard import SummaryWriter   # 引入tensorboard
import datetime
import time
from config import resample, without_quat
from torch.optim.lr_scheduler import StepLR

# --- 1. 定义超参数 (Hyperparameters) ---
INPUT_SIZE = 16     # 每个时间点的特征数 (14 kinematic variables from both arms)4
HIDDEN_SIZE = 256     # LSTM 隐藏层的大小
NUM_LAYERS = 3      # LSTM 层数
NUM_CLASSES = 7      # 标签的类别数量 (0-6, total 7 different actions)
BATCH_SIZE = 16
NUM_EPOCHS = 800
LEARNING_RATE = 0.001

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# --- 2. 准备数据 (Data Preparation) ---
# 在实际项目中，您需要从文件中加载您的demonstrations
# 这里我们创建一个虚拟数据集来模拟变长序列

class KinematicDataset(Dataset):
    def __init__(self, num_samples=100, is_train=True):
        self.samples = []
        
        # # 使用配对加载确保 state 和 label 完全匹配
        # all_states, all_labels = load_demonstrations_paired(shuffle=True, without_quat=True)
        
        # # 划分训练集和测试集
        # bound = round(ratio * len(all_states))
        # if is_train:
        #     demonstrations_state = all_states[:bound]
        #     demonstrations_label = all_labels[:bound]
        # else:
        #     demonstrations_state = all_states[bound:]
        #     demonstrations_label = all_labels[bound:]
        
        # # 标准化
        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler()
        # all_data_stacked = np.vstack(demonstrations_state)
        # scaler.fit(all_data_stacked)
        # demonstrations_state = [scaler.transform(arr) for arr in demonstrations_state]

        demo_id_list = np.arange(148)
        demo_id_list = np.delete(demo_id_list, [80, 81, 92, 109, 112, 117, 122, 144, 145])

        demonstrations_state = load_train_state(without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
        demonstrations_label = load_train_label(resample=resample, demo_id_list=demo_id_list)
        
        # 确保每个演示都是一个序列（2D数组）
        for state_seq, label_seq in zip(demonstrations_state, demonstrations_label):
            if len(state_seq) == 0 or len(label_seq) == 0:
                print("⚠️ 警告: 发现长度为 0 的空序列，已自动跳过！")
                continue
            # 转换为张量
            state_tensor = torch.tensor(state_seq, dtype=torch.float32)
            label_tensor = torch.tensor(label_seq, dtype=torch.long)
            
            # 确保是2D张量 (seq_len, feature_size)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(1)  # 如果是1D，添加特征维度
            
            self.samples.append((state_tensor, label_tensor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class SmoothPhaseLoss(nn.Module):
    def __init__(self, num_classes=7, transition_weight=1.0, ignore_index=-1):
        """
        包含标准交叉熵和阶段跳变惩罚的自定义损失函数
        :param transition_weight: 跳变惩罚的权重 (lambda)，需要根据实际情况调参
        """
        super(SmoothPhaseLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.transition_weight = transition_weight
        self.ignore_index = ignore_index
        
        # 1. 构建惩罚矩阵 M (7x7)
        M = torch.zeros((num_classes, num_classes))


        for i in range(num_classes):
            for j in range(num_classes):
                dist = abs(i - j)
                if dist > 1:
                    # 距离大于 1 时产生惩罚。距离越远，惩罚呈线性（或平方）增长
                    # 例如：0到2惩罚1，0到3惩罚2...
                    M[i, j] = float(dist - 1)  
                    # M[i, j] = float((dist - 1) ** 2) # 如果你希望跳跃惩罚极其严厉，可以加上平方

        # for i in range(num_classes):
        #     for j in range(num_classes):
        #         if j == i:
        #             M[i, j] = 0.0              # 1. 允许：维持当前阶段 (如 1 -> 1)
        #         elif j == i + 1:
        #             M[i, j] = 0.0              # 2. 允许：单步正常推进 (如 1 -> 2)
        #         elif j < i:
        #             # 3. 严惩：任何形式的倒退！
        #             # 倒退得越狠，惩罚越大。例如 2->1 惩罚 1.0； 5->1 惩罚 4.0
        #             M[i, j] = float(i - j)  
        #             # M[i, j] = 5.0 # (或者你也可以给所有倒退都设一个极大的常数死刑，比如 5.0)
        #         elif j > i + 1:
        #             # 4. 惩罚：向前的越级跳跃
        #             # 跳得越远，惩罚越大。例如 1->3 惩罚 1.0
        #             M[i, j] = float(j - i - 1)
                    
        # 使用 register_buffer 确保矩阵会随着模型一起移动到 CPU/GPU
        self.register_buffer('penalty_matrix', M)

    def forward(self, logits, labels):
        """
        :param logits: LSTM 输出的未激活分数，形状 (Batch, Seq_Len, Num_Classes)
        :param labels: 真实标签，形状 (Batch, Seq_Len)
        """
        # --- 第一部分：标准的交叉熵损失 (Cross Entropy) ---
        ce = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # --- 第二部分：基于软概率的转移惩罚 (Transition Penalty) ---
        # 1. 将 logits 转换为概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 2. 错位获取 t-1 时刻 和 t 时刻的概率分布
        p_t_minus_1 = probs[:, :-1, :]  # 形状: (B, T-1, C)
        p_t = probs[:, 1:, :]           # 形状: (B, T-1, C)
        
        # 3. 计算转移期望: P_{t-1} * M * P_{t}
        # p_t_minus_1 乘矩阵 M -> 形状 (B, T-1, C)
        intermediate = torch.matmul(p_t_minus_1, self.penalty_matrix)
        # 再与 p_t 做点积并求和 -> 形状 (B, T-1)
        step_penalties = torch.sum(intermediate * p_t, dim=-1) 
        
        # 4. 生成 Mask，忽略 Padding（-1） 带来的无效转移
        mask_t_minus_1 = (labels[:, :-1] != self.ignore_index)
        mask_t = (labels[:, 1:] != self.ignore_index)
        valid_transitions = mask_t_minus_1 & mask_t  # 只有前后两帧都有效，才计算转移
        
        # 5. 求平均惩罚
        if valid_transitions.sum() > 0:
            trans_loss = (step_penalties * valid_transitions).sum() / valid_transitions.sum()
        else:
            trans_loss = 0.0
            
        # --- 总损失 ---
        total_loss = ce + self.transition_weight * trans_loss
        
        # 返回 total_loss 用于反向传播，顺便返回 ce 和 trans_loss 方便你打印监控
        return total_loss, ce, trans_loss


# 自定义 collate_fn 函数来处理变长序列的批处理
# 这是处理变长序列的关键步骤
def collate_fn(batch):
    # batch 是一个列表，列表的每个元素是 (sequence, labels)
    sequences, labels = zip(*batch)
    
    # 获取每个序列的实际长度
    lengths = [len(seq) for seq in sequences]
    
    # 对序列进行填充 (padding)
    # batch_first=True 表示返回的张量形状为 (batch_size, seq_len, feature_size)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1) # 使用-1作为填充标签

    return padded_sequences, padded_labels, lengths

# 创建 Dataset 和 DataLoader 实例
train_dataset = KinematicDataset(num_samples=200, is_train=True)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn, # 使用我们自定义的函数
    drop_last=False
)


# --- 3. 定义LSTM模型 (LSTM Model Definition) ---
class SequenceLabelingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SequenceLabelingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层: batch_first=True 表示输入和输出的张量将以 (batch, seq, feature) 的形式
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        
        # 线性层: 将LSTM的输出映射到类别空间
        # 因为是双向LSTM，所以输入特征维度是 hidden_size * 2
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        # 打包填充后的序列 (Pack padded sequence)
        # 这可以告诉LSTM忽略填充部分，提高计算效率
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # 前向传播 LSTM
        # h_0 和 c_0 默认为零
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        
        # 解包序列 (Unpack sequence)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # 将LSTM的输出通过全连接层
        # lstm_out shape: (batch_size, seq_len, hidden_size * 2)
        logits = self.fc(lstm_out)
        # logits shape: (batch_size, seq_len, num_classes)
        return logits

# class SequenceLabelingLSTM_CRF(nn.Module): # 建议重命名以区分
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(SequenceLabelingLSTM_CRF, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # LSTM层保持单向
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        
#         # 全连接层将LSTM输出映射到类别空间
#         # 输出的不再是logits，而是送入CRF的"发射分数"(emission scores)
#         self.fc = nn.Linear(hidden_size, num_classes)
        
#         # --- 新增：定义CRF层 ---
#         self.crf = CRF(num_classes, batch_first=True)

#     def forward(self, x, labels, lengths):
#         """
#         用于训练的前向传播，直接返回CRF计算的损失
#         """
#         # 创建一个mask来屏蔽填充部分
#         # mask中padding的位置为False，非padding为True
#         mask = (labels != -1) # 形状: (batch_size, seq_len)
        
#         # LSTM和FC层的前向传播，得到发射分数
#         lstm_out = self._get_lstm_features(x, lengths) # 形状: (batch_size, seq_len, num_classes)
        
#         # --- 修改：计算CRF的对数似然损失 ---
#         # CRF层需要3个输入: 发射分数, 真实标签, mask
#         # 返回的是对数似然，所以我们需要取负数作为损失
#         loss = -self.crf(lstm_out, labels, mask=mask, reduction='mean')
#         return loss

#     def decode(self, x, lengths):
#         """
#         用于预测/评估的解码方法，返回最优标签路径
#         """
#         # 创建一个mask，这次基于输入长度
#         # 注意：这里我们不能依赖labels，因为预测时没有labels
#         max_len = x.size(1)
#         mask = torch.arange(max_len, device=x.device)[None, :] < torch.tensor(lengths, device=x.device)[:, None]
        
#         # 得到发射分数
#         lstm_out = self._get_lstm_features(x, lengths)
        
#         # --- 修改：使用CRF的decode方法 ---
#         # decode方法会使用Viterbi算法找到最优路径
#         best_path = self.crf.decode(lstm_out, mask=mask)
#         return best_path

#     def _get_lstm_features(self, x, lengths):
#         """
#         一个辅助函数，封装了获取发射分数的逻辑
#         """
#         packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
#         packed_output, (h_n, c_n) = self.lstm(packed_input)
#         lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
#         emissions = self.fc(lstm_out)
#         return emissions      


class SequenceLabelingLSTM_CRF(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SequenceLabelingLSTM_CRF, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 【修改】：加入 dropout 参数 (例如 0.5)
        # 注意：dropout 只有在 num_layers > 1 时才生效
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=False,
            dropout=0.2 if num_layers > 1 else 0  # 防止单层报错
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.crf = CRF(num_classes, batch_first=True)

        with torch.no_grad():
            # 1. 先将所有转移分数初始化为一个极小的负数 (例如 -100)
            # 这意味着：除非我们特批，否则任何阶段之间的乱跳（包括倒退和越级跳）都极难发生
            self.crf.transitions.fill_(0) 
            
            for i in range(num_classes):
                # 2. 赋予最高分数：允许保持在当前阶段 (i -> i)
                # 手术视频中绝大多数帧都是停留在当前阶段的
                self.crf.transitions[i, i] = 2.0
                
                # 3. 赋予次高分数：允许正常步进到下一个阶段 (i -> i+1)
                if i < num_classes - 1:
                    self.crf.transitions[i, i+1] = 2.0
                    
                # (可选) 4. 如果您的数据允许偶尔的跳步(如 0->2)，可以给个略微负的分数
                # if i < num_classes - 2:
                #     self.crf.transitions[i, i+2] = -5.0
        # ==============================================================

    def forward(self, x, lengths):
        """
        标准的前向传播：只负责提取特征，返回发射分数(Emissions)。
        符合 PyTorch 规范，方便后续推理调用。
        """
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        emissions = self.fc(lstm_out)
        return emissions

    # def compute_loss(self, x, labels, lengths):
    #     """
    #     专门用于训练的损失计算函数
    #     """
    #     mask = (labels != -1)  # 形状: (batch_size, seq_len)
        
    #     # 【核心修复】：清洗非法的标签索引
    #     # CRF 内部依靠 labels 作为索引，绝不能把 -1 送进去
    #     safe_labels = labels.clone()
    #     safe_labels[~mask] = 0  # 把 padding 的位置全部置为 0
        
    #     # 获取发射分数
    #     emissions = self.forward(x, lengths)
        
    #     # 计算 CRF 的对数似然损失 (注意这里传入 safe_labels)
    #     loss = -self.crf(emissions, safe_labels, mask=mask, reduction='mean')
    #     return loss
    def compute_loss(self, x, labels, lengths):
        """
        专门用于训练的损失计算方法。
        """
        # ================= 新增：第二重保险 =================
        # 采用 lengths 绝对安全地生成 mask，不受 labels 里脏数据的影响
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device)[None, :] < torch.tensor(lengths, device=x.device)[:, None]
        # ====================================================

        # 将 padding 的位置替换为 0，防止 CRF 内部索引越界
        safe_labels = labels.clone()
        
        # ====================================================================
        # 2. 精准清洗：只将“阶段 0 之前”的 -1 修改为 0
        # ====================================================================
        valid_labels = (labels != -1)
        # 只要还没遇到第一个有效标签，cumsum 的结果就是 0
        leading_mask = (valid_labels.cumsum(dim=1) == 0)
        
        # 仅把开头的 -1 变成 0 (Phase 0)
        safe_labels[leading_mask] = 0
        trailing_valid_mask = (safe_labels == -1) & mask
        safe_labels[trailing_valid_mask] = 6  
        
        # 最后，把超出 lengths 范围的纯 Padding 替换为 0 (反正会被 mask 丢弃，只为防越界)
        safe_labels[~mask] = 0
        
        # 获取发射分数
        emissions = self.forward(x, lengths)
        
        # 使用 safe_labels 计算损失
        loss = -self.crf(emissions, safe_labels, mask=mask, reduction='mean')
        return loss

    def decode(self, x, lengths):
        """
        用于预测/评估的解码方法，返回最优标签路径
        """
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device)[None, :] < torch.tensor(lengths, device=x.device)[:, None]
        
        emissions = self.forward(x, lengths)
        best_path = self.crf.decode(emissions, mask=mask)
        return best_path

# --- 保存训练好的模型 ---
def _build_train_scalers(demo_id_list):
    """用训练集 demo 拟合并返回归一化 scaler 字典。"""
    excluded = [80, 81, 92, 109, 112, 117, 122, 144, 145]
    if demo_id_list is None:
        import numpy as _np
        demo_id_list = _np.arange(148)
        demo_id_list = _np.delete(demo_id_list, excluded)

    from load_data import ratio
    raw_all = load_demonstrations_state(
        shuffle=True, without_quat=without_quat,
        resample=resample, demo_id_list=demo_id_list,
    )
    bound = round(ratio * len(raw_all))
    _, scalers = _scale_demos(raw_all[:bound])
    return scalers


def save_model(model, filepath, additional_info=None, demo_id_list=None):
    """
    保存模型到指定路径，同时将训练集 scaler 一起打包进 checkpoint，
    以便实时推理时无需重新加载全量训练数据。

    参数:
    model         : 训练好的模型
    filepath      : 保存路径
    additional_info: 额外的信息字典（可选）
    demo_id_list  : 训练所用的 demo_id 列表（用于拟合 scaler）
    """
    print("正在拟合并保存 scalers …")
    scalers = _build_train_scalers(demo_id_list)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': INPUT_SIZE,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'num_classes': NUM_CLASSES,
            'bidirectional': False,
        },
        'training_info': {
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
        },
        'scalers': scalers,          # dict: pos / vel_scalar / vel3
        'without_quat': without_quat,
    }

    if additional_info:
        save_dict['additional_info'] = additional_info

    torch.save(save_dict, filepath)
    print(f"模型已保存到: {filepath}")

# def train_LSTM():
#     model = SequenceLabelingLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
#     print(model)

#     dir_path = os.path.dirname(__file__)
#     log_total_dir = os.path.join(dir_path, "logs", "LSTM")
#     exp_num = len(os.listdir(log_total_dir))
#     datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_dir = os.path.join(log_total_dir, f"exp{exp_num}_{datetime_str}")
#     os.makedirs(log_dir, exist_ok=True)
#     writer = SummaryWriter(log_dir=log_dir)

#     # 定义损失函数和优化器
#     # CrossEntropyLoss 会自动处理 softmax
#     ignore_index=-1 #告诉损失函数忽略我们之前填充标签用的-1
#     # criterion = nn.CrossEntropyLoss(ignore_index=-1)
#     # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     criterion = SmoothPhaseLoss(num_classes=NUM_CLASSES, transition_weight=1.0, ignore_index=-1).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#     # 开始训练
#     train_start = time.time()
#     for epoch in range(NUM_EPOCHS):
#         epoch_start = time.time()
#         model.train()
#         total_loss = 0
#         for i, (sequences, labels, lengths) in enumerate(train_loader):
#             sequences = sequences.to(device)
#             labels = labels.to(device)

#             outputs = model(sequences, lengths)
#             #loss = criterion(outputs.view(-1, NUM_CLASSES), labels.view(-1))
#             loss, ce_loss, trans_loss = criterion(outputs, labels)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()

#         avg_loss = total_loss / len(train_loader)
#         writer.add_scalar('loss', avg_loss, epoch)

#         elapsed = time.time() - train_start
#         avg_per_epoch = elapsed / (epoch + 1)
#         eta = avg_per_epoch * (NUM_EPOCHS - epoch - 1)
#         eta_m, eta_s = divmod(int(eta), 60)
#         eta_h, eta_m = divmod(eta_m, 60)
        
#         # print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, '
#         #       f'Epoch Time: {time.time()-epoch_start:.1f}s, '
#         #       f'ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}')
#         print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, CE Loss: {ce_loss.item():.4f}, Jump Penalty: {trans_loss.item():.4f} '
#             f'Epoch Time: {time.time()-epoch_start:.1f}s, '
#             f'ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}')


#     total_time = time.time() - train_start
#     print(f"训练完成! 总耗时: {total_time/60:.1f} min")
#     writer.close()
#     return model

def train_LSTM():
    model = SequenceLabelingLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    print(model)

    dir_path = os.path.dirname(__file__)
    log_total_dir = os.path.join(dir_path, "logs", "LSTM")
    exp_num = len(os.listdir(log_total_dir)) if os.path.exists(log_total_dir) else 0
    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_total_dir, f"exp{exp_num}_{datetime_str}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # 【重要修复】：把 Loss 送入 device
    criterion = SmoothPhaseLoss(num_classes=NUM_CLASSES, transition_weight=1.0, ignore_index=-1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    
    scheduler = StepLR(optimizer, step_size=200, gamma=0.5)

    # 开始训练
    train_start = time.time()
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        for i, (sequences, labels, lengths) in enumerate(train_loader):
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences, lengths)
            loss, ce_loss, trans_loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item()

       
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('loss', avg_loss, epoch)

        elapsed = time.time() - train_start
        avg_per_epoch = elapsed / (epoch + 1)
        eta = avg_per_epoch * (NUM_EPOCHS - epoch - 1)
        eta_m, eta_s = divmod(int(eta), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, CE Loss: {ce_loss.item():.4f}, Jump Penalty: {trans_loss.item():.4f}, LR: {current_lr:.6f}, '
            f'Epoch Time: {time.time()-epoch_start:.1f}s, '
            f'ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}')

    total_time = time.time() - train_start
    print(f"训练完成! 总耗时: {total_time/60:.1f} min")
    writer.close()
    return model

# def train_LSTM_CRF():
#     model = SequenceLabelingLSTM_CRF(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
#     print(model)

#     dir_path = os.path.dirname(__file__)
#     log_dir = os.path.join(dir_path, "logs", "LSTM_CRF")
#     os.makedirs(log_dir, exist_ok=True)
#     writer = SummaryWriter(log_dir=log_dir)

#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#     for epoch in range(NUM_EPOCHS):
#         model.train()
#         total_loss = 0
#         for i, (sequences, labels, lengths) in enumerate(train_loader):
#             sequences = sequences.to(device)
#             labels = labels.to(device)
            
#             optimizer.zero_grad()
            
#             # --- 修改：直接调用模型获取损失 ---
#             # 将sequences, labels, lengths全部传入
#             loss = model(sequences, labels, lengths)
            
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()

#         avg_loss = total_loss / len(train_loader)
#         writer.add_scalar('loss', avg_loss, epoch)
#         print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}')

#     print("训练完成!")
#     writer.close()
#     return model

def train_LSTM_CRF():
    model = SequenceLabelingLSTM_CRF(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    print(model)

    dir_path = os.path.dirname(__file__)
    log_total_dir = os.path.join(dir_path, "logs", "LSTM_CRF")
    # 自动增加 exp 文件夹编号，防止覆盖之前的 log
    exp_num = len(os.listdir(log_total_dir)) if os.path.exists(log_total_dir) else 0
    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_total_dir, f"exp{exp_num}_{datetime_str}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # 差异化学习率配置
    crf_params = list(model.crf.parameters())
    base_params = list(model.lstm.parameters()) + list(model.fc.parameters())
    
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': LEARNING_RATE}, # 例如 0.001
        {'params': crf_params, 'lr': LEARNING_RATE * 5.0} # 给 CRF 更大的学习率 0.01
    ])

    
    # 每隔 200 个 Epoch，将学习率衰减为原来的 0.5 倍
    scheduler = StepLR(optimizer, step_size=200, gamma=0.5)

    train_start = time.time()
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        
        for i, (sequences, labels, lengths) in enumerate(train_loader):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(sequences, labels, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('loss', avg_loss, epoch)
        
        elapsed = time.time() - train_start
        avg_per_epoch = elapsed / (epoch + 1)
        eta = avg_per_epoch * (NUM_EPOCHS - epoch - 1)
        eta_m, eta_s = divmod(int(eta), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, '
              f'Epoch Time: {time.time()-epoch_start:.1f}s, '
              f'ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}')

    total_time = time.time() - train_start
    print(f"训练完成! 总耗时: {total_time/60:.1f} min")
    writer.close()
    return model

if __name__ == "__main__":
    # --- 4. 训练循环 (Training Loop) ---
    model = train_LSTM_CRF()

    # save model
    dir_path = os.path.dirname(__file__)
    os.makedirs(os.path.join(dir_path, "LSTM_model"), exist_ok=True)
    model_save_path = os.path.join(dir_path, "LSTM_model", "lstmcrfBAD_sequence_model.pth")
    save_model(model, model_save_path)



