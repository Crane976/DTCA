import sys
import platform
import torch
import sklearn
import xgboost


def get_size(bytes, suffix="B"):
    """Scale bytes to its proper format"""
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def check_config():
    print("=" * 60)
    print("🖥️  实验环境配置检测报告 (Experimental Environment Report)")
    print("=" * 60)

    # --- 1. 硬件信息 (Hardware) ---
    print("\n[Hardware Configuration]")
    # CPU
    print(f"CPU Processor: {platform.processor()}")
    # 内存 (RAM) - 这是一个估算，如果你知道确切的物理内存条大小(如16G, 32G)，以物理内存为准
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"System RAM:    {get_size(ram.total)}")
    except ImportError:
        print("System RAM:    (请在任务管理器中查看，例如 16GB 或 32GB)")

    # GPU
    if torch.cuda.is_available():
        print(f"GPU Model:     {torch.cuda.get_device_name(0)}")
        # VRAM
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU VRAM:      {get_size(vram_bytes)}")
        print(f"CUDA Version:  {torch.version.cuda}")
    else:
        print("GPU:           None (Running on CPU)")

    # --- 2. 软件信息 (Software) ---
    print("\n[Software Configuration]")
    print(f"OS Platform:   {platform.system()} {platform.release()}")
    print(f"Python Ver:    {sys.version.split()[0]}")
    print(f"PyTorch Ver:   {torch.__version__}")
    print(f"Scikit-learn:  {sklearn.__version__}")
    print(f"XGBoost Ver:   {xgboost.__version__}")

    # IDE (IDE通常不用代码查，你自己知道是 PyCharm)
    print("IDE Platform:  PyCharm (Professional/Community)")


if __name__ == "__main__":
    check_config()