# models/train_cnn_hunter.py (FINAL ROBUST VERSION)
import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler  # ✅ 引入 Sampler
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

from config import DEFENDER_SET, set_seed, LogMinMaxScaler  # 引入自定义 Scaler
from models.cnn_architecture import CNN_Classifier
from models.mlp_architecture import FocalLoss

# --- Configuration ---
TRAIN_SET_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
TEST_SET_PATH = os.path.join(project_root, 'data', 'splits', 'holdout_test_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
CNN_HUNTER_MODEL_PATH = os.path.join(project_root, 'models', 'cnn_hunter.pt')

FEATURE_DIM = len(DEFENDER_SET)
EPOCHS = 100
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 2025
BEST_PARAMS = {'learning_rate': 0.0005}


# ✅ 复用 MLP 的清洗逻辑
def clean_data(df, feature_cols):
    """统一的数据清洗函数: 替换Inf并丢弃NaN"""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=feature_cols, inplace=True)
    return df


def main():
    set_seed(RANDOM_SEED)
    print("=" * 60)
    print("🚀 开始训练 1D-CNN Hunter (ResNet-like, Balanced)...")
    print("=" * 60)
    print(f"使用设备: {device}")

    # --- 1. 加载与清洗 ---
    print("\n[步骤1] 正在加载数据和Scaler...")
    df_train_full = pd.read_csv(TRAIN_SET_PATH)
    df_test = pd.read_csv(TEST_SET_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = scaler.feature_names_in_

    print("   -> 正在清洗数据...")
    df_train_full = clean_data(df_train_full, feature_names)
    df_test = clean_data(df_test, feature_names)

    # --- 2. 转换与划分 ---
    X_train_full_scaled = scaler.transform(df_train_full[feature_names])
    y_train_full = df_train_full['label'].values

    X_test_scaled = scaler.transform(df_test[feature_names])
    y_test = df_test['label'].values

    # 划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full_scaled, y_train_full, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=y_train_full
    )

    # --- 3. 均衡采样器 (关键!) ---
    print("\n[步骤2] 配置均衡采样器 (WeightedRandomSampler)...")
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train.astype(int)])
    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))

    # shuffle=False 因为用了 sampler
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False)

    val_tensor_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_tensor_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    # --- 4. 训练 ---
    benign_ratio = (y_train_full == 0).sum() / len(y_train_full)
    model = CNN_Classifier(feature_dim=FEATURE_DIM).to(device)
    # 既然用了均衡采样，Focal Loss 的 alpha 可以设为 0.5 或者干脆用 CrossEntropy
    # 这里保持 Focal Loss 以增强对难样本的挖掘
    criterion = FocalLoss(alpha=0.5, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BEST_PARAMS['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    print("\n[步骤3] 开始训练...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        # 加入 tqdm 显示进度
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for x_batch, y_batch in pbar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        model.eval()
        with torch.no_grad():
            val_logits = model(val_tensor_x)
            val_loss = criterion(val_logits, val_tensor_y).item()

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CNN_HUNTER_MODEL_PATH)

    print(f"\n✅ 训练完成，最佳验证损失: {best_val_loss:.6f}")

    # --- 5. 阈值寻优 ---
    print("\n[步骤4] 寻找最佳决策阈值...")
    final_model = CNN_Classifier(feature_dim=FEATURE_DIM).to(device)
    final_model.load_state_dict(torch.load(CNN_HUNTER_MODEL_PATH, map_location=device))
    final_model.eval()

    with torch.no_grad():
        val_probs = final_model.predict(val_tensor_x).cpu().numpy()

    best_threshold, best_f1 = 0.5, 0
    for threshold in np.arange(0.01, 1.0, 0.01):
        y_val_pred = (val_probs > threshold).astype(int)
        current_f1 = f1_score(y_val, y_val_pred, pos_label=1)
        if current_f1 > best_f1:
            best_f1, best_threshold = current_f1, threshold

    print(f"✅ 最佳阈值: {best_threshold:.2f} (F1: {best_f1:.4f})")

    # --- 6. 最终评估 ---
    print("\n--- '1D-CNN Hunter'在【留出测试集】上的真实性能报告 ---")
    with torch.no_grad():
        test_tensor_x = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        test_probs = final_model.predict(test_tensor_x).cpu().numpy()
        y_pred = (test_probs > best_threshold).astype(int)
    print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Bot (1)'], digits=4))


if __name__ == "__main__":
    main()