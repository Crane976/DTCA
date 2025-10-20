# models/STEP2_train_bot_lstm.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
# ✅ 核心修正: 从 torch.utils.data 中同时导入 Dataset, DataLoader, TensorDataset
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
from config import UNIFIED_FEATURE_SET, TARGET_FIELDS_FOR_LSTM

# 导入我们在STEP1中定义的CAE模型类
# 为了方便，直接在这里重新定义
class ConditionalAE(nn.Module):
    def __init__(self, input_dim, condition_dim, encoding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim + condition_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x, c):
        x_cond = torch.cat([x, c], dim=1);
        encoded = self.encoder(x_cond)
        encoded_cond = torch.cat([encoded, c], dim=1);
        decoded = self.decoder(encoded_cond)
        return decoded, encoded


# 导入我们在之前定义的LSTM模型类
class PredictiveLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, output_dim))

    def forward(self, x):
        lstm_out, _ = self.lstm(x);
        last_time_step_out = lstm_out[:, -1, :];
        return self.fc(last_time_step_out)


# --- 1. 配置区 ---
DATA_DIR_PREPROCESSED = r'D:\DTCA\data\preprocessed'
DATA_DIR_FILTERED = r'D:\DTCA\data\filtered'
MODELS_DIR = r'D:\DTCA\models'

# --- 输入路径 ---
# STEP1训练好的风格迁移引擎
cae_model_path = os.path.join(MODELS_DIR, 'style_transfer_cae.pt')
# LSTM的训练数据源 (已处理的Bot流量)
bot_processed_path = os.path.join(DATA_DIR_PREPROCESSED, 'bot_traffic_processed.csv')
# LSTM的目标值来源 (原始的Bot流量)
raw_bot_csv = os.path.join(DATA_DIR_FILTERED, 'bot_traffic_target.csv')

# --- 输出路径 ---
lstm_model_path = os.path.join(MODELS_DIR, 'bot_pattern_lstm.pt')
target_scaler_path = os.path.join(MODELS_DIR, 'target_scaler_bot.pkl')  # Scaler for Y values

# --- 模型参数 ---
input_dim_cae = len(UNIFIED_FEATURE_SET)
encoding_dim = 5
condition_dim = 2
input_dim_lstm = encoding_dim  # LSTM的输入是CAE的编码维度

window_size = 3
batch_size = 32
epochs = 100
learning_rate = 0.001
hidden_dim = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
target_fields = TARGET_FIELDS_FOR_LSTM


# --- 数据集定义 ---
class SequenceDataset(Dataset):
    def __init__(self, x, y): self.x = torch.tensor(x, dtype=torch.float32); self.y = torch.tensor(y,
                                                                                                   dtype=torch.float32)

    def __len__(self): return len(self.x)

    def __getitem__(self, idx): return self.x[idx], self.y[idx]


# --- 主训练函数 ---
def main():
    print("=============================================");
    print("🚀 STEP 2: 开始训练'动态模式注入'LSTM引擎...");
    print("=============================================");
    print(f"使用设备: {device}")

    # 1. 加载STEP1的CAE模型
    print("正在加载'风格迁移'CAE引擎...")
    cae_model = ConditionalAE(input_dim_cae, condition_dim, encoding_dim).to(device)
    cae_model.load_state_dict(torch.load(cae_model_path))
    cae_model.eval()  # 我们只用它的encoder，所以设为评估模式

    # 2. 准备LSTM的输入 (X) 和目标 (Y)
    print("正在准备LSTM的训练数据...")
    df_bot_processed = pd.read_csv(bot_processed_path)
    X_bot_for_encoding = torch.tensor(df_bot_processed.values, dtype=torch.float32).to(device)

    # 创建Bot的条件标签 [0, 1]
    C_bot_for_encoding = torch.zeros(len(X_bot_for_encoding), condition_dim).to(device)
    C_bot_for_encoding[:, 1] = 1

    # 使用CAE编码器提取Bot的潜在表示，作为LSTM的输入特征
    with torch.no_grad():
        _, lstm_input_features = cae_model(X_bot_for_encoding, C_bot_for_encoding)
    lstm_input_features = lstm_input_features.cpu().numpy()

    # 准备LSTM的目标值Y
    df_raw = pd.read_csv(raw_bot_csv);
    df_raw.columns = df_raw.columns.str.strip()
    target_values = df_raw[target_fields].values
    target_scaler = MinMaxScaler();
    target_values_scaled = target_scaler.fit_transform(target_values)
    joblib.dump(target_scaler, target_scaler_path);
    print(f"✅ 目标值(Y)的Bot-Scaler已保存到: {target_scaler_path}")

    # 3. 构建滑动窗口序列
    print("正在构建滑动窗口序列...")
    X_seq, Y_seq = [], []
    for i in range(len(lstm_input_features) - window_size):
        X_seq.append(lstm_input_features[i:i + window_size]);
        Y_seq.append(target_values_scaled[i + window_size - 1])
    X_seq, Y_seq = np.array(X_seq), np.array(Y_seq)
    X_train, X_val, Y_train, Y_val = train_test_split(X_seq, Y_seq, test_size=0.2, random_state=42)

    # 4. 训练LSTM
    train_loader = DataLoader(SequenceDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, Y_val), batch_size=batch_size)

    output_dim_lstm = len(target_fields)
    lstm_model = PredictiveLSTM(input_dim_lstm, hidden_dim, output_dim_lstm).to(device)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate);
    criterion = nn.MSELoss()

    print("开始训练LSTM模型...")
    best_val_loss = float('inf')
    for epoch in range(epochs):
        lstm_model.train();
        total_train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = lstm_model(x_batch);
            loss = criterion(pred, y_batch)
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        lstm_model.eval();
        total_val_loss = 0
        with torch.no_grad():
            for x_val_batch, y_val_batch in val_loader:
                x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)
                val_pred = lstm_model(x_val_batch);
                val_loss = criterion(val_pred, y_val_batch)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        if (epoch + 1) % 10 == 0: print(f"  -> Epoch {epoch + 1:3d}/{epochs}, Val Loss: {avg_val_loss:.6f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss;
            torch.save(lstm_model.state_dict(), lstm_model_path)

    print("\n--- 训练完成 ---");
    print(f"表现最好的'动态模式注入'LSTM引擎已保存在: {lstm_model_path}");
    print(f"(Final Best Val Loss: {best_val_loss:.6f})")


if __name__ == "__main__":
    main()