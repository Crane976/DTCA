# evaluation/final_evaluation_deception_rate.py
import pandas as pd
import numpy as np
import os
import joblib
from config import UNIFIED_FEATURE_SET
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================================
# --- 1. 配置区 ---
# ==========================================================
DATA_DIR = r'D:\DTCA\data'
MODELS_DIR = r'D:\DTCA\models'
FIGURES_DIR = r'D:\DTCA\figures'

# --- 输入路径 ---
hunter_model_path = os.path.join(MODELS_DIR, 'xgboost_hunter.pkl')
scaler_path = os.path.join(MODELS_DIR, 'global_scaler.pkl')
test_set_path = os.path.join(DATA_DIR, 'preprocessed', 'evaluation_test_set.csv')
# 我们的最终“欺骗弹药”
camouflage_bot_path = os.path.join(DATA_DIR, 'generated', 'final_camouflage_bot.csv')

# --- 实验参数 ---
# 我们将注入全部40000条伪装流量
INJECT_ALL_CAMOUFLAGE = True

# --- 中文显示配置 ---
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


# ==========================================================
# --- 2. 主评估函数 ---
# ==========================================================
def main():
    print("=============================================");
    print("🚀 最终评估: 开始计算'欺骗成功率 (DSR)'...");
    print("=============================================")

    # --- 1. 加载核心资产 ---
    print("正在加载'猎手'模型、scaler和所有测试数据...")
    try:
        hunter_model = joblib.load(hunter_model_path)
        scaler = joblib.load(scaler_path)
        df_test = pd.read_csv(test_set_path)
        df_camouflage_bot_raw = pd.read_csv(camouflage_bot_path)
    except FileNotFoundError as e:
        print(f"错误: 找不到核心文件 - {e}"); return

    df_test_benign = df_test[df_test['label'] == 0].copy()
    df_test_bot_real = df_test[df_test['label'] == 1].copy()
    num_real_bot_in_test = len(df_test_bot_real)

    # --- 2. 创建最终的混合测试环境 ---
    print("\n正在创建最终的混合测试环境...")

    # 为流量添加'is_camouflage'标志，用于事后分析
    df_camouflage_bot_raw['is_camouflage'] = 1
    df_test_benign['is_camouflage'] = 0
    df_test_bot_real['is_camouflage'] = 0

    # --- 3. 预处理所有数据 ---
    # 所有数据都必须用训练“猎手”时使用的scaler进行处理
    print("正在预处理混合数据集...")

    # 合并所有流量的原始特征
    df_mix_raw = pd.concat([
        df_test_benign,
        df_test_bot_real,
        df_camouflage_bot_raw
    ], ignore_index=True)

    # 创建真实标签 (对于猎手来说，真实Bot和伪装Bot都应该是Bot)
    y_true_for_hunter = pd.concat([
        pd.Series(np.zeros(len(df_test_benign))),  # 良性
        pd.Series(np.ones(len(df_test_bot_real))),  # 真实Bot
        pd.Series(np.ones(len(df_camouflage_bot_raw)))  # 伪装Bot
    ], ignore_index=True)

    # 提取并归一化特征
    X_mix_features = df_mix_raw[UNIFIED_FEATURE_SET]
    X_mix_scaled = scaler.transform(X_mix_features)

    print(f"  -> 最终混合测试集创建完毕: 共 {len(X_mix_scaled)} 条样本。")
    print(
        f"     其中包含: {len(df_test_benign)} 良性, {num_real_bot_in_test} 真实Bot, {len(df_camouflage_bot_raw)} 伪装Bot。")

    # --- 4. 让“猎手”进行狩猎 ---
    print("\n'猎手'开始在混合环境中进行狩猎...")
    y_pred_mix = hunter_model.predict(X_mix_scaled)

    # --- 5. 分析结果，计算欺骗成功率 ---
    print("\n正在分析狩猎结果，计算DSR...")

    # 筛选出所有被猎手判断为"Bot"(标签1)的告警
    alert_indices = np.where(y_pred_mix == 1)[0]
    total_alerts = len(alert_indices)

    # 在这些告警中，检查有多少是我们注入的伪装Bot
    camouflage_alerts = df_mix_raw.iloc[alert_indices]['is_camouflage'].sum()

    # 计算欺骗成功率
    if total_alerts > 0:
        deception_success_rate = (camouflage_alerts / total_alerts) * 100
    else:
        deception_success_rate = 0

        # 计算对真实Bot的召回率
    # 找到真实Bot在混合集中的原始索引
    real_bot_indices = df_mix_raw[
        (df_mix_raw['is_camouflage'] == 0) & (y_true_for_hunter == 1)
        ].index

    real_bot_preds = y_pred_mix[real_bot_indices]
    real_bot_alerts = np.sum(real_bot_preds == 1)
    real_bot_recall = (real_bot_alerts / num_real_bot_in_test) * 100 if num_real_bot_in_test > 0 else 0

    # --- 6. 打印最终的“战报” ---
    print("\n=============================================")
    print("--- 最终评估结果: '假目标欺骗' ---")
    print("=============================================")
    print(f"战场环境:")
    print(f"  - 真实良性流量: {len(df_test_benign)}")
    print(f"  - 真实Bot流量: {num_real_bot_in_test}")
    print(f"  - 注入的伪装Bot: {len(df_camouflage_bot_raw)}")
    print("---------------------------------------------")
    print(f"战果分析:")
    print(f"  - '猎手'总共发出了 {total_alerts} 个 'Bot' 告警。")
    print(
        f"  - 其中, 捕获到'真实Bot'的数量: {real_bot_alerts} / {num_real_bot_in_test} (召回率: {real_bot_recall:.2f}%)")
    print(f"  - 其中, 捕获到'伪装Bot'的数量: {camouflage_alerts} / {len(df_camouflage_bot_raw)}")
    print("---------------------------------------------")
    print(f"🎯 欺骗成功率 (DSR): {deception_success_rate:.2f}%")
    print(f"  (这意味着'猎手'捕获的所有'Bot'中，有 {deception_success_rate:.2f}% 是我们主动投喂的、带水印的无害诱饵)")
    print("=============================================")


if __name__ == "__main__":
    main()