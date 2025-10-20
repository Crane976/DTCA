# preprocess/step3_apply_global_scaler.py
import pandas as pd
import numpy as np
import os
import joblib
from config import UNIFIED_FEATURE_SET  # 导入我们最终的统一特征集

# ==========================================================
# --- 1. 配置区 ---
# ==========================================================
# 输入:
# 1. 待处理的数据 (Bot流量)
input_path = r'D:\DTCA\data\filtered\bot_traffic_target.csv'
# 2. 我们唯一的“度量衡”
scaler_path = os.path.join(r'D:\DTCA\models', 'global_scaler.pkl')

# 输出
output_dir = r'D:\DTCA\data\preprocessed'
output_csv_path = os.path.join(output_dir, 'bot_traffic_processed.csv')


# ==========================================================
# --- 2. 主函数 ---
# ==========================================================
def main():
    print("=============================================")
    print("🚀 STEP 2b: 应用全局Scaler处理BOT流量...")
    print("=============================================")

    try:
        print(f"正在加载Bot流量数据: {input_path}...")
        df = pd.read_csv(input_path, low_memory=False)
        print(f"正在加载全局Scaler: {scaler_path}...")
        scaler = joblib.load(scaler_path)
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 - {e}")
        print("请确保您已经成功运行了 'step2_build_global_scaler.py'")
        return

    df.columns = df.columns.str.strip()

    # --- 1. 数据验证与清理 ---
    missing_features = [f for f in UNIFIED_FEATURE_SET if f not in df.columns]
    if missing_features:
        print(f"错误: 在Bot数据中找不到以下特征: {missing_features}");
        return

    df_selected = df[UNIFIED_FEATURE_SET].copy()
    df_selected.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_selected.dropna(inplace=True)
    print(f"数据清理后，剩余样本数: {len(df_selected)}")

    # --- 2. 应用全局Scaler ---
    print("\n正在使用全局Scaler对Bot数据进行归一化...")
    # ✅ 关键操作: 只能使用 .transform()
    features_scaled = scaler.transform(df_selected)
    df_processed = pd.DataFrame(features_scaled, columns=UNIFIED_FEATURE_SET)

    # --- 3. 保存结果 ---
    df_processed.to_csv(output_csv_path, index=False)
    print(f"✅ 已保存处理后的Bot流量数据到: {output_csv_path}")


if __name__ == "__main__":
    main()