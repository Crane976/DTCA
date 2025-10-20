# models/STEP1_train_style_transfer_cae.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
from config import UNIFIED_FEATURE_SET

# ==========================================================
# --- 1. 配置区 ---
# ==========================================================
DATA_DIR = r'D:\DTCA\data\preprocessed'
MODELS_DIR = r'D:\DTCA\models'

# --- 输入路径 ---
benign_processed_path = os.path.join(DATA_DIR, 'benign_traffic_processed.csv')
bot_processed_path = os.path.join(DATA_DIR, 'bot_traffic_processed.csv')

# --- 输出路径 ---
# 这是我们“风格迁移”引擎的核心
cae_model_path = os.path.join(MODELS_DIR, 'style_transfer_cae.pt')

# --- 模型参数 ---
input_dim = len(UNIFIED_FEATURE_SET)
encoding_dim = 5
condition_dim = 2  # ✅ 关键：现在我们有两类 (Benign, Bot)，所以是2维
epochs = 100  # 可以多训练一会儿，让模型充分学习
batch_size = 128
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 模型定义 ---
class ConditionalAE(nn.Module):
    def __init__(self, input_dim, condition_dim, encoding_dim):
        super().__init__()
        # 可以适当加深网络以学习更复杂的映射
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
        x_cond = torch.cat([x, c], dim=1)
        encoded = self.encoder(x_cond)
        encoded_cond = torch.cat([encoded, c], dim=1)
        decoded = self.decoder(encoded_cond)
        return decoded, encoded


# ==========================================================
# --- 2. 主训练函数 ---
# ==========================================================
def main():
    print("=============================================")
    print("🚀 STEP 1: 开始训练'风格迁移'CAE引擎...")
    print("=============================================")
    print(f"使用设备: {device}")

    # --- 1. 加载并合并数据 ---
    print("正在加载并准备Benign和Bot数据...")
    try:
        df_benign = pd.read_csv(benign_processed_path)
        df_bot = pd.read_csv(bot_processed_path)
    except FileNotFoundError as e:
        print(f"错误: 找不到预处理文件 - {e}");
        return

    # --- 2. 创建特征(X)和条件标签(C) ---
    X_benign = df_benign.values
    X_bot = df_bot.values

    # Benign: label 0 -> one-hot [1, 0]
    C_benign = np.zeros((len(X_benign), condition_dim))
    C_benign[:, 0] = 1

    # Bot: label 1 -> one-hot [0, 1]
    C_bot = np.zeros((len(X_bot), condition_dim))
    C_bot[:, 1] = 1

    # 合并所有数据
    X_full = np.concatenate([X_benign, X_bot], axis=0)
    C_full = np.concatenate([C_benign, C_bot], axis=0)

    # --- 3. 划分数据集并创建DataLoader ---
    X_train, X_val, C_train, C_val = train_test_split(X_full, C_full, test_size=0.2, random_state=42,
                                                      stratify=C_full.argmax(axis=1))

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(C_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_tensor_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_tensor_c = torch.tensor(C_val, dtype=torch.float32).to(device)

    # --- 4. 初始化模型并开始训练 ---
    model = ConditionalAE(input_dim, condition_dim, encoding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("\n开始训练CAE模型...")
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for x_batch, c_batch in train_loader:
            x_batch, c_batch = x_batch.to(device), c_batch.to(device)
            recon, _ = model(x_batch, c_batch)
            loss = criterion(recon, x_batch)
            optimizer.zero_grad();
            loss.backward();
            optimizer.step()

        model.eval()
        with torch.no_grad():
            recon_val, _ = model(val_tensor_x, val_tensor_c)
            val_loss = criterion(recon_val, val_tensor_x).item()
            if (epoch + 1) % 10 == 0:
                print(f"  -> Epoch {epoch + 1:3d}/{epochs}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cae_model_path)

    print("\n--- 训练完成 ---")
    print(f"表现最好的'风格迁移'CAE引擎已保存在: {cae_model_path}")
    print(f"(Final Best Val Loss: {best_val_loss:.6f})")


if __name__ == "__main__":
    main()