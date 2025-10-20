# step1_filter_and_split.py
import pandas as pd
import os

# --- 配置 ---
# 原始数据文件路径
raw_data_path = r'D:\DTCA\data\Friday-WorkingHours-Morning.pcap_ISCX.csv'

# 输出文件夹
output_dir = r'D:\DTCA\data\filtered'
os.makedirs(output_dir, exist_ok=True)

# 输出文件名
benign_output_path = os.path.join(output_dir, 'benign_traffic.csv')
bot_output_path = os.path.join(output_dir, 'bot_traffic_target.csv')  # 作为我们要隐藏的目标


# --- 主函数 ---
def main():
    print(f"正在加载原始数据: {raw_data_path}...")
    # 使用 low_memory=False 避免DtypeWarning
    df = pd.read_csv(raw_data_path, low_memory=False)

    # 清理列名中可能存在的首尾空格，这是个好习惯
    df.columns = df.columns.str.strip()

    # 确保标签列存在
    if 'Label' not in df.columns:
        print("错误: 找不到 'Label' 列。请检查CSV文件。")
        return

    print("原始数据集标签分布:")
    print(df['Label'].value_counts())

    # --- 筛选良性流量 ---
    benign_df = df[df['Label'] == 'BENIGN'].copy()
    print(f"\n筛选出 {len(benign_df)} 条良性流量...")
    benign_df.to_csv(benign_output_path, index=False)
    print(f"✅ 已保存良性流量到: {benign_output_path}")

    # --- 筛选Bot流量 (我们的C&C信道) ---
    bot_df = df[df['Label'] == 'Bot'].copy()
    print(f"\n筛选出 {len(bot_df)} 条Bot流量...")
    bot_df.to_csv(bot_output_path, index=False)
    print(f"✅ 已保存Bot流量到: {bot_output_path}")


if __name__ == "__main__":
    main()