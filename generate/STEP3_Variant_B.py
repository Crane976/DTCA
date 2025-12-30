# generate/STEP3_Variant_B_no_constraint.py
import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
from sklearn.cluster import KMeans  # 保留聚类

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

from models.style_transfer_cae import ConditionalAutoencoder
from models.lstm_finetuner import LSTMFinetuner
from models.lstm_predictor import LSTMPredictor
from config import DEFENDER_SET, ATTACKER_KNOWLEDGE_SET, ATTACKER_ACTION_SET, COMPLEX_SET, set_seed

# --- 配置 ---
# (路径保持不变，除了 Output)
CLEAN_DATA_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
CAE_MODEL_PATH = os.path.join(project_root, 'models', 'style_transfer_cae.pt')
LSTM_FINETUNER_MODEL_PATH = os.path.join(project_root, 'models', 'lstm_finetuner.pt')
PREDICTOR_MODEL_PATH = os.path.join(project_root, 'models', 'lstm_reconciliation_predictor.pt')

# 输出文件改为 Variant B
OUTPUT_CSV_PATH = os.path.join(project_root, 'data', 'generated', 'variant_B_no_constraint.csv')

FEATURE_DIM_CAE = len(ATTACKER_KNOWLEDGE_SET)
LATENT_DIM_CAE = 5
NUM_CLASSES_CAE = 2
INPUT_DIM_LSTM_FINETUNER = len(ATTACKER_KNOWLEDGE_SET)
OUTPUT_DIM_LSTM_FINETUNER = len(ATTACKER_ACTION_SET)
INPUT_DIM_PREDICTOR = len(ATTACKER_ACTION_SET)
OUTPUT_DIM_PREDICTOR = len(COMPLEX_SET)

NUM_TO_GENERATE = 40000
MIMIC_INTENSITY = 0.98
NUM_BOT_CLUSTERS = 5
WATERMARK_KEY = 97
WATERMARK_FEATURE = 'Flow Duration'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inject_watermark(df, key, feature_name):
    # 水印逻辑中包含了一些基本的 Bytes/s 计算，这里我们保留它
    # 因为如果不算，水印可能会导致 Duration 变了但 Rate 没变，这本身就是一种微小的硬约束
    # 但核心的 STEP 6 被移除了，所以对比依然有效
    print(f"\n🌊 [步骤7] 注入水印 (Variant B)...")
    df_w = df.copy()
    values = df_w[feature_name].values.astype(int)
    residuals = values % key
    new_values = values - residuals
    mask_too_small = (new_values <= 0)
    new_values[mask_too_small] += key
    df_w[feature_name] = new_values

    # 在 Variant B 中，我们只更新 Duration，**不更新** Bytes/s 和 Pkts/s
    # 这样更能体现"无约束"带来的不自洽性
    print("   -> (注意: Variant B 不会同步更新关联特征，故意保留不自洽性)")

    return df_w


def main():
    set_seed(2025)
    print("=" * 60)
    print("🚀 [消融实验 Variant B] 无硬约束 (No Hard Constraints)...")
    print("=" * 60)

    # 1. 加载 (不变)
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

    # 1.5 聚类 (保留)
    bot_scaled_full = scaler.transform(df_bot_all[DEFENDER_SET])
    kmeans = KMeans(n_clusters=NUM_BOT_CLUSTERS, random_state=2025, n_init=10)
    kmeans.fit(bot_scaled_full)
    centers_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
    df_bot_centers = pd.DataFrame(centers_unscaled, columns=DEFENDER_SET)
    tutor_indices = np.random.randint(0, NUM_BOT_CLUSTERS, size=NUM_TO_GENERATE)
    df_bot_tutors = df_bot_centers.iloc[tutor_indices].reset_index(drop=True)

    # 2. 风格植入 (不变)
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

    # 3. LSTM (不变)
    with torch.no_grad():
        input_for_lstm = generated_knowledge_features_scaled.unsqueeze(1)
        refined_action = lstm_finetuner(input_for_lstm)
        df_knowledge_scaled = pd.DataFrame(generated_knowledge_features_scaled.cpu().numpy(),
                                           columns=ATTACKER_KNOWLEDGE_SET)
        original_action = torch.tensor(df_knowledge_scaled[ATTACKER_ACTION_SET].values, dtype=torch.float32).to(device)
        fused_action = 0.3 * original_action + 0.7 * refined_action
        fused_action = np.clip(fused_action.cpu().numpy(), 0, 1)

    # 4. 预测 (不变)
    with torch.no_grad():
        input_predictor = torch.tensor(fused_action, dtype=torch.float32).unsqueeze(1).to(device)
        predicted_complex = predictor(input_predictor).cpu().numpy()
        predicted_complex = np.clip(predicted_complex, 0, 1)

    # 5. 逆向缩放 (不变)
    df_temp_action = pd.DataFrame(0, index=range(NUM_TO_GENERATE), columns=DEFENDER_SET)
    df_temp_action[ATTACKER_ACTION_SET] = fused_action
    action_unscaled = pd.DataFrame(scaler.inverse_transform(df_temp_action), columns=DEFENDER_SET)[ATTACKER_ACTION_SET]

    df_temp_complex = pd.DataFrame(0, index=range(NUM_TO_GENERATE), columns=DEFENDER_SET)
    df_temp_complex[COMPLEX_SET] = predicted_complex
    complex_unscaled = pd.DataFrame(scaler.inverse_transform(df_temp_complex), columns=DEFENDER_SET)[COMPLEX_SET]

    df_final = pd.concat([action_unscaled, complex_unscaled], axis=1)

    # --- ❌ 移除硬约束 ---
    print("\n[步骤6] 跳过硬约束校准 (Ablation: No Constraints)...")
    # 直接使用神经网络预测的原始值，不做数学修正
    # 仅补全缺失列 (主要是那些 Calculated Set 里的特征，如 Bytes/s，会被补0或保持NaN)
    for col in DEFENDER_SET:
        if col not in df_final.columns:
            # 简单补0，或者如果不补0 Scaler会报错吗？
            # 还是尽量算一下基础的吧，不然模型可能直接报错
            # 为了体现"没有强制约束"，我们只做最基本的补全，不做修正
            df_final[col] = 0

    df_final = df_final[DEFENDER_SET]

    # 7. 水印 (修改版，不更新关联特征)
    df_final_watermarked = inject_watermark(df_final, WATERMARK_KEY, WATERMARK_FEATURE)

    df_final_watermarked.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ Variant B 生成完毕: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()