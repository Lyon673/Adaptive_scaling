import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from load_data import load_train_state, load_train_label
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm # 引入tqdm来显示进度条，因为这个过程会慢很多
import os
from torchcrf import CRF
from torch.utils.tensorboard import SummaryWriter   # 引入tensorboard
import datetime
from config import resample, without_quat

# --- 1. 定义超参数 (Hyperparameters) ---
INPUT_SIZE = 16  if not without_quat else 8    # 每个时间点的特征数 (14 kinematic variables from both arms)4
HIDDEN_SIZE = 256     # LSTM 隐藏层的大小
NUM_LAYERS = 5      # LSTM 层数
NUM_CLASSES = 6      # 标签的类别数量 (0-5, total 6 different actions)
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
        demonstrations_state = load_train_state(without_quat=without_quat, resample=resample)
        demonstrations_label = load_train_label(resample=resample)
        
        # 确保每个演示都是一个序列（2D数组）
        for state_seq, label_seq in zip(demonstrations_state, demonstrations_label):
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

class SequenceLabelingLSTM_CRF(nn.Module): # 建议重命名以区分
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SequenceLabelingLSTM_CRF, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层保持单向
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        
        # 全连接层将LSTM输出映射到类别空间
        # 输出的不再是logits，而是送入CRF的"发射分数"(emission scores)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # --- 新增：定义CRF层 ---
        self.crf = CRF(num_classes, batch_first=True)

    def forward(self, x, labels, lengths):
        """
        用于训练的前向传播，直接返回CRF计算的损失
        """
        # 创建一个mask来屏蔽填充部分
        # mask中padding的位置为False，非padding为True
        mask = (labels != -1) # 形状: (batch_size, seq_len)
        
        # LSTM和FC层的前向传播，得到发射分数
        lstm_out = self._get_lstm_features(x, lengths) # 形状: (batch_size, seq_len, num_classes)
        
        # --- 修改：计算CRF的对数似然损失 ---
        # CRF层需要3个输入: 发射分数, 真实标签, mask
        # 返回的是对数似然，所以我们需要取负数作为损失
        loss = -self.crf(lstm_out, labels, mask=mask, reduction='mean')
        return loss

    def decode(self, x, lengths):
        """
        用于预测/评估的解码方法，返回最优标签路径
        """
        # 创建一个mask，这次基于输入长度
        # 注意：这里我们不能依赖labels，因为预测时没有labels
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device)[None, :] < torch.tensor(lengths, device=x.device)[:, None]
        
        # 得到发射分数
        lstm_out = self._get_lstm_features(x, lengths)
        
        # --- 修改：使用CRF的decode方法 ---
        # decode方法会使用Viterbi算法找到最优路径
        best_path = self.crf.decode(lstm_out, mask=mask)
        return best_path

    def _get_lstm_features(self, x, lengths):
        """
        一个辅助函数，封装了获取发射分数的逻辑
        """
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        emissions = self.fc(lstm_out)
        return emissions        

# --- 保存训练好的模型 ---
def save_model(model, filepath, additional_info=None):
    """
    保存模型到指定路径
    
    参数:
    model: 训练好的模型
    filepath: 保存路径
    additional_info: 额外的信息字典（可选）
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': INPUT_SIZE,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'num_classes': NUM_CLASSES,
            'bidirectional': False  # 根据你的模型设置
        },
        'training_info': {
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        }
    }
    
    if additional_info:
        save_dict['additional_info'] = additional_info
    
    torch.save(save_dict, filepath)
    print(f"模型已保存到: {filepath}")

def train_LSTM():
    model = SequenceLabelingLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    print(model)

    dir_path = os.path.dirname(__file__)
    log_total_dir = os.path.join(dir_path, "logs", "LSTM")
    exp_num = len(os.listdir(log_total_dir))
    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_total_dir, f"exp{exp_num}_{datetime_str}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # 定义损失函数和优化器
    # CrossEntropyLoss 会自动处理 softmax
    ignore_index=-1 #告诉损失函数忽略我们之前填充标签用的-1
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 开始训练
    for epoch in range(NUM_EPOCHS):
        model.train() # 设置模型为训练模式
        total_loss = 0
        for i, (sequences, labels, lengths) in enumerate(train_loader):
            # 将数据移动到指定设备
            sequences = sequences.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(sequences, lengths)

            # 计算损失
            # CrossEntropyLoss期望的输入是 (N, C) 和 (N)
            # 所以我们需要调整 outputs 和 labels 的形状
            # outputs: (batch, seq_len, num_classes) -> (batch * seq_len, num_classes)
            # labels: (batch, seq_len) -> (batch * seq_len)
            loss = criterion(outputs.view(-1, NUM_CLASSES), labels.view(-1))
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('loss', avg_loss, epoch)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}')

    print("训练完成!")
    writer.close()
    return model

def train_LSTM_CRF():
    model = SequenceLabelingLSTM_CRF(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    print(model)

    dir_path = os.path.dirname(__file__)
    log_dir = os.path.join(dir_path, "logs", "LSTM_CRF")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for i, (sequences, labels, lengths) in enumerate(train_loader):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # --- 修改：直接调用模型获取损失 ---
            # 将sequences, labels, lengths全部传入
            loss = model(sequences, labels, lengths)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('loss', avg_loss, epoch)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}')

    print("训练完成!")
    writer.close()
    return model

if __name__ == "__main__":
    # --- 4. 训练循环 (Training Loop) ---
    model = train_LSTM()

    # save model
    dir_path = os.path.dirname(__file__)
    os.makedirs(os.path.join(dir_path, "LSTM_model"), exist_ok=True)
    model_save_path = os.path.join(dir_path, "LSTM_model", "lstm_sequence_model.pth")
    save_model(model, model_save_path)



