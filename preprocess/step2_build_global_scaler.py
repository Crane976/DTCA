# preprocess/step2_build_global_scaler.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
from config import UNIFIED_FEATURE_SET # 导入我们最终的统一特征集

# ==========================================================
# --- 1. 配置区 ---
# ==========================================================
# 输入: 我们的基础参照系——纯良性流量
input_path = r'D:\DTCA\data\filtered\benign_traffic.csv'

# 输出
output_dir = r'D:\DTCA\data\preprocessed'
output_csv_path = os.path.join(output_dir, 'benign_traffic_processed.csv')
# ✅ 核心输出：我们唯一的、全局的“度量衡”
scaler_path = os.path.join(r'D:\DTCA\models', 'global_scaler.pkl')

# ==========================================================
# --- 2. 主函数 ---
# ==========================================================
def main():
    print("=============================================")
    print("🚀 STEP 2a: 构建全局Scaler并处理BENIGN流量...")
    print("=============================================")

    print(f"正在加载良性流量数据: {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    df.columns = df.columns.str.strip()

    # --- 1. 数据验证与清理 ---
    missing_features = [f for f in UNIFIED_FEATURE_SET if f not in df.columns]
    if missing_features:
        print(f"错误: 在良性数据中找不到以下特征: {missing_features}"); return

    df_selected = df[UNIFIED_FEATURE_SET].copy()
    print(f"已选择 {len(df_selected.columns)} 个统一特征。")

    df_selected.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_selected.dropna(inplace=True)
    print(f"数据清理后，剩余样本数: {len(df_selected)}")

    # --- 2. 训练全局Scaler并归一化 ---
    print("\n正在训练全局Scaler并对良性数据进行归一化...")
    scaler = MinMaxScaler()
    # ✅ 关键操作: 在良性数据上 .fit_transform()
    features_scaled = scaler.fit_transform(df_selected)
    df_processed = pd.DataFrame(features_scaled, columns=UNIFIED_FEATURE_SET)

    # --- 3. 保存结果 ---
    df_processed.to_csv(output_csv_path, index=False)
    print(f"✅ 已保存处理后的良性流量数据到: {output_csv_path}")

    joblib.dump(scaler, scaler_path)
    print(f"✅ 全局Scaler已保存到: {scaler_path}")

if __name__ == "__main__":
    main()