#!/usr/bin/env python3
"""
CTU-chb 双通道切窗脚本（FHR+UC）
窗口256/步长128，输出双通道信号，适配论文预处理流程
"""
import numpy as np
import wfdb
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
from scipy import interpolate, signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置中文字体（按优先级尝试）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Arial']
except:
    pass
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10


# ===================== 切窗参数 =====================
INPUT_LEN = 256
STRIDE = 128
FS_HZ = 4.0
FHR_MIN, FHR_MAX = 50, 200
UC_MIN, UC_MAX = 0, 100

# 预处理参数（科学设定）
# 中值滤波核大小（去除尖峰噪声，单位：样本点，4Hz采样下1秒=4个点）
MEDIAN_KERNEL = 5                # 约1.25秒窗口，去除瞬时尖峰
# 平滑滤波器参数
SMOOTH_WINDOW = 7                # 平滑窗口大小（约1.75秒）

# 可视化参数
PLOT_SAMPLE_START = 0
PLOT_SAMPLE_END = 1000


# ===================== 路径 =====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

db_root = PROJECT_ROOT / "ctu-chb-intrapartum-cardiotocography-database-1.0.0" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0"

if not db_root.exists():
    raise FileNotFoundError(f"找不到数据集: {db_root}")

MANIFEST_PATH = DATA_DIR / "manifest.csv"
OUTPUT_NPZ = DATA_DIR / "windows_w256_s128_dual.npz"

print(f"项目根目录: {PROJECT_ROOT}")
print(f"数据输出目录: {DATA_DIR}")
print(f"数据集目录: {db_root}")

# ===================== 标签 =====================
if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"找不到标签文件: {MANIFEST_PATH}")
manifest = pd.read_csv(MANIFEST_PATH)
labels_dict = dict(zip(manifest.record_id.astype(str), manifest.label_acidosis_risk))


def median_filter(sig, kernel_size):
    """中值滤波去除尖峰噪声（FHR特有的beat halving/doubling等伪影）"""
    return signal.medfilt(sig, kernel_size=kernel_size)


def moving_average_smooth(sig, window_size):
    """移动平均平滑处理"""
    if window_size <= 1:
        return sig
    kernel = np.ones(window_size) / window_size
    return np.convolve(sig, kernel, mode='same')


def preprocess_fhr(fhr_raw, fhr_min, fhr_max):
    """
    FHR信号预处理流程（科学方法）：
    1. 异常值裁剪到生理范围
    2. 中值滤波去除尖峰（beat halving/doubling等伪影）
    3. 轻度平滑
    4. 样条插值填充缺失区域
    """
    # Step 1: 标记有效点和异常点
    valid_mask = (fhr_raw >= fhr_min) & (fhr_raw <= fhr_max)
    fhr_clipped = np.clip(fhr_raw.copy(), fhr_min, fhr_max)

    # Step 2: 中值滤波（去除尖峰伪影）
    fhr_medfilt = median_filter(fhr_clipped, MEDIAN_KERNEL)

    # Step 3: 轻度平滑
    fhr_smooth = moving_average_smooth(fhr_medfilt, SMOOTH_WINDOW)

    return fhr_smooth, valid_mask


def preprocess_uc(uc_raw, uc_min, uc_max):
    """
    UC信号预处理流程：
    1. 异常值裁剪
    2. 轻度平滑（UC变化较缓慢）
    """
    valid_mask = (uc_raw >= uc_min) & (uc_raw <= uc_max)
    uc_clipped = np.clip(uc_raw.copy(), uc_min, uc_max)

    # UC信号变化较缓慢，使用较小平滑窗口
    uc_smooth = moving_average_smooth(uc_clipped, max(3, SMOOTH_WINDOW // 2))

    return uc_smooth, valid_mask


def plot_filter_comparison(fhr_raw, fhr_processed, uc_raw, uc_processed,
                           save_path, fs=FS_HZ):
    """绘制预处理前后信号对比图"""
    n_samples = min(len(fhr_raw), PLOT_SAMPLE_END)
    t = np.arange(n_samples) / fs

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('胎心信号预处理前后对比（中值滤波+平滑）', fontsize=14, fontweight='bold')

    # FHR 处理前（原始信号）
    axes[0, 0].plot(t, fhr_raw[:n_samples], 'b-', linewidth=0.8)
    axes[0, 0].set_title('FHR（胎心率）- 处理前（原始信号）')
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('FHR (bpm)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 250])

    # FHR 处理后（中值滤波+平滑）
    axes[0, 1].plot(t, fhr_processed[:n_samples], 'g-', linewidth=0.8)
    axes[0, 1].set_title('FHR（胎心率）- 处理后（中值滤波+平滑）')
    axes[0, 1].set_xlabel('时间 (s)')
    axes[0, 1].set_ylabel('FHR (bpm)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 250])

    # UC 处理前（原始信号）
    axes[1, 0].plot(t, uc_raw[:n_samples], 'b-', linewidth=0.8)
    axes[1, 0].set_title('UC（宫缩压力）- 处理前（原始信号）')
    axes[1, 0].set_xlabel('时间 (s)')
    axes[1, 0].set_ylabel('UC (mmHg)')
    axes[1, 0].grid(True, alpha=0.3)

    # UC 处理后（平滑）
    axes[1, 1].plot(t, uc_processed[:n_samples], 'g-', linewidth=0.8)
    axes[1, 1].set_title('UC（宫缩压力）- 处理后（轻度平滑）')
    axes[1, 1].set_xlabel('时间 (s)')
    axes[1, 1].set_ylabel('UC (mmHg)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [可视化] 预处理对比图已保存: {save_path}")


def plot_interpolation_comparison(sig_raw, valid_mask, sig_interpolated,
                                   save_path, signal_name='FHR', fs=FS_HZ):
    """绘制样条插值前后缺失值处理效果图"""
    n_samples = min(len(sig_raw), PLOT_SAMPLE_END)
    t = np.arange(n_samples) / fs

    # 信号中文名称映射
    name_map = {'FHR': 'FHR（胎心率）', 'UC': 'UC（宫缩压力）'}
    display_name = name_map.get(signal_name, signal_name)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    fig.suptitle(f'{display_name} 缺失值处理前后对比（三次样条插值）', fontsize=14, fontweight='bold')

    # 处理前 - 显示有效点和无效区域
    axes[0].plot(t, sig_raw[:n_samples], 'b-', linewidth=0.8, alpha=0.5, label='原始信号')
    invalid_mask = ~valid_mask[:n_samples]
    invalid_x = np.where(invalid_mask)[0]
    if len(invalid_x) > 0:
        axes[0].scatter(invalid_x / fs, sig_raw[:n_samples][invalid_mask],
                       c='red', s=3, alpha=0.3, label='缺失/异常区域（待插值）')
    axes[0].set_title(f'{display_name} - 处理前（标记缺失/异常区域）')
    axes[0].set_xlabel('时间 (s)')
    axes[0].set_ylabel(f'{signal_name}')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # 处理后
    axes[1].plot(t, sig_interpolated[:n_samples], 'g-', linewidth=0.8, label='插值后信号')
    axes[1].set_title(f'{display_name} - 处理后（三次样条插值完成）')
    axes[1].set_xlabel('时间 (s)')
    axes[1].set_ylabel(f'{signal_name}')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [可视化] 插值对比图已保存: {save_path}")

def interpolate_spline(valid_x, valid_y, length):
    """使用三次样条插值补全信号"""
    if len(valid_x) < 4:
        return np.interp(np.arange(length), valid_x, valid_y)
    spline = interpolate.interp1d(valid_x, valid_y, kind='cubic', fill_value='extrapolate')
    return spline(np.arange(length))


def clean_signal(sig, valid_mask, min_max):
    """对信号进行样条插值补全"""
    valid_x = np.where(valid_mask)[0]
    valid_y = sig[valid_mask]
    if len(valid_x) == 0:
        sig_clean = np.full_like(sig, np.nanmean(sig))
    else:
        sig_clean = interpolate_spline(valid_x, valid_y, len(sig))
    return np.clip(sig_clean, min_max[0], min_max[1])


# ===================== 双通道切窗 =====================
records = sorted({f.stem for f in db_root.glob("*.hea")})
X_list, y_list, record_ids = [], [], []

# 可视化输出目录
PLOT_DIR = PROJECT_ROOT / "figures" / "preprocessing"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

for rid in tqdm(records, desc="切窗中"):
    try:
        sig, meta = wfdb.rdsamp(str(db_root / rid))
        fhr = sig[:, 0].astype(np.float32)  # 通道0：胎心率
        uc = sig[:, 1].astype(np.float32)   # 通道1：宫缩

        if rid == records[0]:
            print(f"通道0(FHR)范围: [{fhr.min():.1f}, {fhr.max():.1f}]")
            print(f"通道1(UC)范围: [{uc.min():.1f}, {uc.max():.1f}]")

            # ========== 生成可视化 ==========
            print("\n[可视化] 正在生成预处理效果图...")

            # 1. 预处理前后对比图
            fhr_proc, _ = preprocess_fhr(fhr.copy(), FHR_MIN, FHR_MAX)
            uc_proc, _ = preprocess_uc(uc.copy(), UC_MIN, UC_MAX)
            plot_filter_comparison(fhr, fhr_proc, uc, uc_proc,
                                 PLOT_DIR / "01_preprocessing_comparison.png")

            # 2. 异常值检测与样条插值对比图 (FHR)
            fhr_mask = (fhr >= FHR_MIN) & (fhr <= FHR_MAX)
            fhr_final = clean_signal(fhr.copy(), fhr_mask, (FHR_MIN, FHR_MAX))
            plot_interpolation_comparison(fhr, fhr_mask, fhr_final,
                                         PLOT_DIR / "02_fhr_interpolation_comparison.png",
                                         signal_name='FHR')

            # 3. 异常值检测与样条插值对比图 (UC)
            uc_mask = (uc >= UC_MIN) & (uc <= UC_MAX)
            uc_final = clean_signal(uc.copy(), uc_mask, (UC_MIN, UC_MAX))
            plot_interpolation_comparison(uc, uc_mask, uc_final,
                                         PLOT_DIR / "03_uc_interpolation_comparison.png",
                                         signal_name='UC')

            print("[可视化] 所有预处理效果图已生成！\n")
            # ==================================

        fhr_mask = (fhr >= FHR_MIN) & (fhr <= FHR_MAX)
        uc_mask = (uc >= UC_MIN) & (uc <= UC_MAX)
        if fhr_mask.sum() < INPUT_LEN:
            continue

        # 使用科学的预处理流程
        fhr_clean, _ = preprocess_fhr(fhr, FHR_MIN, FHR_MAX)
        uc_clean, _ = preprocess_uc(uc, UC_MIN, UC_MAX)

        dual_sig = np.stack([fhr_clean, uc_clean], axis=1)

        label = labels_dict[rid]
        for s in range(0, len(dual_sig) - INPUT_LEN + 1, STRIDE):
            win = dual_sig[s:s+INPUT_LEN]
            X_list.append(win)
            y_list.append(label)
            record_ids.append(rid)

    except KeyError:
        print(f"⚠️ {rid} 无对应标签，已跳过")
        continue
    except Exception as e:
        print(f"❌ {rid} 处理出错: {str(e)}")
        continue

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)
ids = np.array(record_ids)

# 归一化到 [0,1]
X[..., 0] = (X[..., 0] - FHR_MIN) / (FHR_MAX - FHR_MIN)
X[..., 1] = (X[..., 1] - UC_MIN) / (UC_MAX - UC_MIN)
X = np.clip(X, 0, 1)

# ===================== 保存数据 =====================
np.savez(OUTPUT_NPZ, X=X, y=y, record_ids=ids)

# 保存归一化参数供后续脚本使用
norm_params = {
    "fhr_min": float(FHR_MIN),
    "fhr_max": float(FHR_MAX),
    "uc_min": float(UC_MIN),
    "uc_max": float(UC_MAX),
    "window_size": INPUT_LEN,
    "stride": STRIDE,
    "sample_rate": FS_HZ
}
with open(DATA_DIR / "norm_params.json", "w", encoding="utf-8") as f:
    json.dump(norm_params, f, indent=2, ensure_ascii=False)

metadata = {
    "window_size": INPUT_LEN,
    "stride": STRIDE,
    "sample_rate": FS_HZ,
    "shape": str(X.shape),
    "total_samples": len(X),
    "normal_samples": int((y == 0).sum()),
    "abnormal_samples": int((y == 1).sum()),
    "channels": ["FHR", "UC"],
    "normalize": "MinMax [0,1]"
}
with open(DATA_DIR / "window_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\n✅ 保存成功！最终数据形状: {X.shape}")
print(f"✅ 总样本：{len(X)} | 正常：{(y==0).sum()} | 异常：{(y==1).sum()}")
print(f"✅ 归一化参数已保存至: {DATA_DIR / 'norm_params.json'}")
print(f"✅ 预处理可视化图已保存至: {PLOT_DIR}")
print(f"   - 01_preprocessing_comparison.png      (中值滤波+平滑前后对比)")
print(f"   - 02_fhr_interpolation_comparison.png (FHR异常值检测与插值对比)")
print(f"   - 03_uc_interpolation_comparison.png   (UC异常值检测与插值对比)")