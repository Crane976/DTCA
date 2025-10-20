# generate/STEP3_generate_camouflage_bot.py (v2 - with Quantity Control)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import joblib
from config import UNIFIED_FEATURE_SET, TARGET_FIELDS_FOR_LSTM


# --- 1. 模型定义 ---
class ConditionalAE(nn.Module):
    def __init__(self, input_dim, condition_dim, encoding_dim):
        super().__init__();
        self.encoder = nn.Sequential(nn.Linear(input_dim + condition_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(),
                                     nn.Linear(16, encoding_dim));
        self.decoder = nn.Sequential(nn.Linear(encoding_dim + condition_dim, 16), nn.ReLU(), nn.Linear(16, 32),
                                     nn.ReLU(), nn.Linear(32, input_dim))

    def forward(self, x, c):
        x_cond = torch.cat([x, c], dim=1);
        encoded = self.encoder(x_cond);
        return encoded


class PredictiveLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__();
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2);
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, output_dim))

    def forward(self, x):
        lstm_out, _ = self.lstm(x);
        last_time_step_out = lstm_out[:, -1, :];
        return self.fc(last_time_step_out)


# --- 2. 配置区 ---
DATA_DIR_PREPROCESSED = r'D:\DTCA\data\preprocessed'
DATA_DIR_FILTERED = r'D:\DTCA\data\filtered'
DATA_DIR_GENERATED = r'D:\DTCA\data\generated'
MODELS_DIR = r'D:\DTCA\models'
os.makedirs(DATA_DIR_GENERATED, exist_ok=True)

cae_model_path = os.path.join(MODELS_DIR, 'style_transfer_cae.pt')
lstm_model_path = os.path.join(MODELS_DIR, 'bot_pattern_lstm.pt')
benign_processed_path = os.path.join(DATA_DIR_PREPROCESSED, 'benign_traffic_processed.csv')
raw_benign_csv_path = os.path.join(DATA_DIR_FILTERED, 'benign_traffic.csv')
target_scaler_path = os.path.join(MODELS_DIR, 'target_scaler_bot.pkl')
output_csv_path = os.path.join(DATA_DIR_GENERATED, 'final_camouflage_bot.csv')

# ✅ 核心修改：定义我们想要生成的伪装Bot数量
NUM_TO_GENERATE = 30000  # 生成4万条，大约是真实Bot数量的100倍

input_dim_cae = len(UNIFIED_FEATURE_SET);
encoding_dim = 5;
condition_dim = 2
input_dim_lstm = encoding_dim;
hidden_dim_lstm = 64;
output_dim_lstm = len(TARGET_FIELDS_FOR_LSTM)
window_size = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 3. 主生成函数 ---
def main():
    print("=============================================");
    print("🚀 STEP 3: 开始执行'良性变形记'，生成最终伪装Bot流量...");
    print("=============================================")

    # 1. 加载所有核心引擎和数据
    print("正在加载所有模型、scaler和原材料...");
    cae_model = ConditionalAE(input_dim_cae, condition_dim, encoding_dim).to(device)
    cae_model.load_state_dict(torch.load(cae_model_path));
    cae_model.eval()
    lstm_model = PredictiveLSTM(input_dim_lstm, hidden_dim_lstm, output_dim_lstm).to(device)
    lstm_model.load_state_dict(torch.load(lstm_model_path));
    lstm_model.eval()
    target_scaler = joblib.load(target_scaler_path)

    # ✅ 核心修改：只加载我们需要数量的“原材料”（良性流量）
    df_benign_processed = pd.read_csv(benign_processed_path, nrows=NUM_TO_GENERATE + window_size)
    df_benign_raw = pd.read_csv(raw_benign_csv_path, nrows=NUM_TO_GENERATE + window_size)
    df_benign_raw.columns = df_benign_raw.columns.str.strip()
    print(f"  -> 已加载 {len(df_benign_raw)} 条良性流量作为生成原材料。")

    # 2. 静态风格迁移
    print("\n[第一阶段] 正在进行静态风格迁移...");
    X_benign_tensor = torch.tensor(df_benign_processed.values, dtype=torch.float32).to(device)
    C_bot_label = torch.zeros(len(X_benign_tensor), condition_dim).to(device);
    C_bot_label[:, 1] = 1
    with torch.no_grad():
        z_fake_bot = cae_model(X_benign_tensor, C_bot_label).cpu().numpy()

    # 3. 动态模式注入
    print("\n[第二阶段] 正在进行动态模式注入...");
    X_seq, pred_indices = [], []
    for i in range(len(z_fake_bot) - window_size):
        X_seq.append(z_fake_bot[i:i + window_size]);
        pred_indices.append(i + window_size - 1)
    with torch.no_grad():
        predictions_scaled = lstm_model(torch.tensor(np.array(X_seq), dtype=torch.float32).to(device)).cpu().numpy()
    predictions_real = target_scaler.inverse_transform(predictions_scaled)

    # 4. 应用约束并生成
    print("\n[第三阶段] 正在应用约束并组装最终伪装Bot...");
    df_camouflage = df_benign_raw.copy()
    df_bot_raw = pd.read_csv(os.path.join(DATA_DIR_FILTERED, 'bot_traffic_target.csv'))
    df_bot_raw.columns = df_bot_raw.columns.str.strip()
    upper_bounds = df_bot_raw[TARGET_FIELDS_FOR_LSTM].quantile(0.95).values
    lower_bounds = df_bot_raw[TARGET_FIELDS_FOR_LSTM].quantile(0.05).values
    original_values_to_modify = df_camouflage.loc[pred_indices, TARGET_FIELDS_FOR_LSTM].values
    modified_values = np.where(
        (predictions_real >= lower_bounds) & (predictions_real <= upper_bounds),
        predictions_real,
        original_values_to_modify
    )
    df_camouflage.loc[pred_indices, TARGET_FIELDS_FOR_LSTM] = modified_values
    final_camouflage_df = df_camouflage.iloc[pred_indices].copy()

    # 5. 保存最终成果
    final_camouflage_df = final_camouflage_df[UNIFIED_FEATURE_SET]
    final_camouflage_df.to_csv(output_csv_path, index=False)

    print("\n=============================================");
    print(f"🎉 恭喜！'良性变形记'完成！");
    print(f"✅ 最终的伪装Bot流量已保存至: {output_csv_path}");
    print(f"   共生成 {len(final_camouflage_df)} 条高保真伪装Bot流量。");
    print("=============================================")


if __name__ == "__main__":
    main()