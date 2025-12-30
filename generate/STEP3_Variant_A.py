# generate/STEP3_Variant_A_no_cluster.py
import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch

# form sklearn.cluster import KMeans # ❌ 移除 KMeans

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

from models.style_transfer_cae import ConditionalAutoencoder
from models.lstm_finetuner import LSTMFinetuner
from models.lstm_predictor import LSTMPredictor
from config import DEFENDER_SET, ATTACKER_KNOWLEDGE_SET, ATTACKER_ACTION_SET, COMPLEX_SET, set_seed

# --- 配置 ---
CLEAN_DATA_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
CAE_MODEL_PATH = os.path.join(project_root, 'models', 'style_transfer_cae.pt')
LSTM_FINETUNER_MODEL_PATH = os.path.join(project_root, 'models', 'lstm_finetuner.pt')
PREDICTOR_MODEL_PATH = os.path.join(project_root, 'models', 'lstm_reconciliation_predictor.pt')

# 输出文件改为 Variant A
OUTPUT_CSV_PATH = os.path.join(project_root, 'data', 'generated', 'variant_A_no_cluster.csv')

FEATURE_DIM_CAE = len(ATTACKER_KNOWLEDGE_SET)
LATENT_DIM_CAE = 5
NUM_CLASSES_CAE = 2
INPUT_DIM_LSTM_FINETUNER = len(ATTACKER_KNOWLEDGE_SET)
OUTPUT_DIM_LSTM_FINETUNER = len(ATTACKER_ACTION_SET)
INPUT_DIM_PREDICTOR = len(ATTACKER_ACTION_SET)
OUTPUT_DIM_PREDICTOR = len(COMPLEX_SET)

NUM_TO_GENERATE = 40000
MIMIC_INTENSITY = 0.98  # 保持强度一致，只改变导师选择策略
WATERMARK_KEY = 97
WATERMARK_FEATURE = 'Flow Duration'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inject_watermark(df, key, feature_name):
    # (保持原有的水印函数不变，为了控制变量，消融实验通常只改变一个因素)
    # ... (代码省略，与最终版一致) ...
    df_w = df.copy()
    values = df_w[feature_name].values.astype(int)
    residuals = values % key
    new_values = values - residuals
    mask_too_small = (new_values <= 0)
    new_values[mask_too_small] += key
    df_w[feature_name] = new_values

    duration_sec = df_w['Flow Duration'] / 1e6
    if 'Total Length of Fwd Packets' in df_w.columns:
        total_bytes = df_w['Total Length of Fwd Packets'] + df_w['Total Length of Bwd Packets']
        df_w['Flow Bytes/s'] = total_bytes / (duration_sec + 1e-9)
    if 'Total Fwd Packets' in df_w.columns:
        total_pkts = df_w['Total Fwd Packets'] + df_w['Total Backward Packets']
        df_w['Flow Packets/s'] = total_pkts / (duration_sec + 1e-9)
    return df_w


def main():
    set_seed(2025)
    print("=" * 60)
    print("🚀 [消融实验 Variant A] 无聚类聚焦 (Random Tutor)...")
    print("=" * 60)

    # 1. 加载模型 (不变)
    scaler = joblib.load(SCALER_PATH)
    predictor = LSTMPredictor(INPUT_DIM_PREDICTOR, OUTPUT_DIM_PREDICTOR).to(device)
    predictor.load_state_dict(torch.load(PREDICTOR_MODEL_PATH, map_location=device))
    predictor.eval()
    cae_model = ConditionalAutoencoder(FEATURE_DIM_CAE, LATENT_DIM_CAE, NUM_CLASSES_CAE).to(device)
    cae_model.load_state_dict(torch.load(CAE_MODEL_PATH, map_location=device))
    cae_model.eval()
    lstm_finetuner = LSTMFinetuner(INPUT_DIM_LSTM_FINETUNER, OUTPUT_DIM_LSTM_FINETUNER).to(device)
    lstm_finetuner.load_state_dict(torch.load(LSTM_FINETUNER_MODEL_PATH, map_location=device))
    lstm_finetuner.eval()

    df_clean_full = pd.read_csv(CLEAN_DATA_PATH)
    df_benign_source = df_clean_full[df_clean_full['label'] == 0].sample(n=NUM_TO_GENERATE, replace=True,
                                                                         random_state=2025)
    df_bot_all = df_clean_full[df_clean_full['label'] == 1]

    # --- ❌ 移除聚类步骤 ---
    print("\n[步骤1.5] 跳过聚类 (Ablation: No Clustering)...")
    print("   -> 直接从真实 Bot 样本中随机抽取导师。")

    # 直接随机抽取作为导师，不经过 KMeans 提纯
    df_bot_tutors = df_bot_all.sample(n=NUM_TO_GENERATE, replace=True, random_state=2025).reset_index(drop=True)

    # --- 2. 风格植入 (不变) ---
    print("\n[步骤2] TIER 1: 执行点对点风格植入...")
    with torch.no_grad():
        source_scaled = scaler.transform(df_benign_source[DEFENDER_SET])
        df_source_scaled = pd.DataFrame(source_scaled, columns=DEFENDER_SET)
        X_benign = torch.tensor(df_source_scaled[ATTACKER_KNOWLEDGE_SET].values, dtype=torch.float32).to(device)
        c_benign = torch.tensor([1.0, 0.0], dtype=torch.float32).expand(len(X_benign), -1).to(device)
        z_benign = cae_model.encode(X_benign, c_benign)

        tutors_scaled = scaler.transform(df_bot_tutors[DEFENDER_SET])
        df_tutors_scaled = pd.DataFrame(tutors_scaled, columns=DEFENDER_SET)
        X_bot = torch.tensor(df_tutors_scaled[ATTACKER_KNOWLEDGE_SET].values, dtype=torch.float32).to(device)
        c_bot_input = torch.tensor([0.0, 1.0], dtype=torch.float32).expand(len(X_bot), -1).to(device)
        z_bot = cae_model.encode(X_bot, c_bot_input)

        z_hybrid = (1 - MIMIC_INTENSITY) * z_benign + MIMIC_INTENSITY * z_bot
        c_bot_target = torch.tensor([0.0, 1.0], dtype=torch.float32).expand(len(z_hybrid), -1).to(device)
        generated_knowledge_features_scaled = cae_model.decode(z_hybrid, c_bot_target)

    # --- 3. LSTM (不变) ---
    print("\n[步骤3] TIER 2: LSTM 战术微调...")
    with torch.no_grad():
        input_for_lstm = generated_knowledge_features_scaled.unsqueeze(1)
        refined_action = lstm_finetuner(input_for_lstm)
        df_knowledge_scaled = pd.DataFrame(generated_knowledge_features_scaled.cpu().numpy(),
                                           columns=ATTACKER_KNOWLEDGE_SET)
        original_action = torch.tensor(df_knowledge_scaled[ATTACKER_ACTION_SET].values, dtype=torch.float32).to(device)
        fused_action = 0.3 * original_action + 0.7 * refined_action
        fused_action = np.clip(fused_action.cpu().numpy(), 0, 1)

    # --- 4. 预测 (不变) ---
    print("\n[步骤4] TIER 3: 衍生特征预测...")
    with torch.no_grad():
        input_predictor = torch.tensor(fused_action, dtype=torch.float32).unsqueeze(1).to(device)
        predicted_complex = predictor(input_predictor).cpu().numpy()
        predicted_complex = np.clip(predicted_complex, 0, 1)

    # --- 5. 逆向缩放 (不变) ---
    print("\n[步骤5] 逆向缩放...")
    df_temp_action = pd.DataFrame(0, index=range(NUM_TO_GENERATE), columns=DEFENDER_SET)
    df_temp_action[ATTACKER_ACTION_SET] = fused_action
    action_unscaled = pd.DataFrame(scaler.inverse_transform(df_temp_action), columns=DEFENDER_SET)[ATTACKER_ACTION_SET]

    df_temp_complex = pd.DataFrame(0, index=range(NUM_TO_GENERATE), columns=DEFENDER_SET)
    df_temp_complex[COMPLEX_SET] = predicted_complex
    complex_unscaled = pd.DataFrame(scaler.inverse_transform(df_temp_complex), columns=DEFENDER_SET)[COMPLEX_SET]

    df_final = pd.concat([action_unscaled, complex_unscaled], axis=1)

    # --- 6. 硬约束 (保留) ---
    print("\n[步骤6] 应用硬约束 (Ablation: Yes)...")
    df_final['Total Fwd Packets'] = df_final['Total Fwd Packets'].clip(lower=1)
    df_final['Total Backward Packets'] = df_final['Total Backward Packets'].clip(lower=0)
    df_final['Average Packet Size'] = df_final['Average Packet Size'].clip(lower=0)
    df_final['Total Length of Fwd Packets'] = df_final['Total Fwd Packets'] * df_final['Average Packet Size']
    df_final['Total Length of Bwd Packets'] = df_final['Total Backward Packets'] * df_final['Average Packet Size']
    total_pkts = df_final['Total Fwd Packets'] + df_final['Total Backward Packets']
    total_len = df_final['Total Length of Fwd Packets'] + df_final['Total Length of Bwd Packets']
    df_final['Packet Length Mean'] = total_len / (total_pkts + 1e-9)
    df_final['Flow Duration'] = df_final['Flow Duration'].clip(lower=1)
    duration_sec = df_final['Flow Duration'] / 1e6
    df_final['Flow Bytes/s'] = total_len / (duration_sec + 1e-9)
    df_final['Flow Packets/s'] = total_pkts / (duration_sec + 1e-9)
    df_final['Down/Up Ratio'] = df_final['Total Backward Packets'] / (df_final['Total Fwd Packets'] + 1e-9)
    cols_root = ['Fwd Packet Length', 'Bwd Packet Length', 'Flow IAT', 'Fwd IAT', 'Bwd IAT']
    for root in cols_root:
        if f'{root} Min' in df_final.columns and f'{root} Max' in df_final.columns:
            df_final[f'{root} Min'] = df_final[f'{root} Min'].clip(lower=0)
            df_final[f'{root} Max'] = np.maximum(df_final[f'{root} Max'], df_final[f'{root} Min'])
            if f'{root} Mean' in df_final.columns:
                df_final[f'{root} Mean'] = np.clip(df_final[f'{root} Mean'], df_final[f'{root} Min'],
                                                   df_final[f'{root} Max'])
    for col in DEFENDER_SET:
        if col not in df_final.columns: df_final[col] = 0
    df_final = df_final[DEFENDER_SET]

    # --- 7. 水印 (保留) ---
    df_final_watermarked = inject_watermark(df_final, WATERMARK_KEY, WATERMARK_FEATURE)

    df_final_watermarked.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ Variant A 生成完毕: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()