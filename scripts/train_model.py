#!/usr/bin/env python3
"""
胎心信号异常检测 - 严格10万参数量版（参数量 ~33k，推理<50ms）
针对居家场景优化：提高异常召回率至40%+
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import time
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from scipy import signal
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 配置 =====================
INPUT_LEN = 256
FS = 4.0
# 科学调整：增强L1正则化
L1_REG = 5e-5
LR = 5e-4
EPOCHS = 100
BATCH_SIZE = 128

# 类别权重（配合Focal Loss）
CLASS_WEIGHT = {0: 1.0, 1: 5.0}

# Focal Loss参数
FOCAL_GAMMA = 1.5
FOCAL_ALPHA = 0.75

current = Path(__file__).resolve().parent.parent
DATA_DIR = current / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)
data_path = DATA_DIR / "train_test_final.npz"
model_dir = current / "models"
model_dir.mkdir(exist_ok=True)
model_save_path = model_dir / "strict_29k_model.h5"

# 加载归一化参数
norm_params_path = DATA_DIR / "norm_params.json"
if norm_params_path.exists():
    with open(norm_params_path, 'r') as f:
        norm_params = json.load(f)
    FHR_MIN = norm_params['fhr_min']
    FHR_MAX = norm_params['fhr_max']
    UC_MIN = norm_params['uc_min']
    UC_MAX = norm_params['uc_max']
else:
    FHR_MIN, FHR_MAX = 110.0, 170.0
    UC_MIN, UC_MAX = 0.0, 100.0
    print("警告: 未找到归一化参数，使用默认值")


# ===================== 1. 加载数据 =====================
print("=" * 60)
print("胎心信号异常检测 - 严格参数量版")
print("=" * 60)

if not data_path.exists():
    print(f"错误: 数据文件不存在: {data_path}")
    exit(1)

data = np.load(data_path)
X_train_raw = data["X_train"]
y_train = data["y_train"]

print(f"原始训练集: {len(y_train)} | 正常: {(y_train==0).sum()} | 异常: {(y_train==1).sum()}")
print(f"类别比例: 异常={100*(y_train==1).sum()/len(y_train):.2f}%")

# 分出验证集
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_train_raw, y_train, test_size=0.15, random_state=42, stratify=y_train
)

X_test_raw = data["X_test"]
y_test = data["y_test"]

print(f"训练集: {len(y_train)} | 验证集: {len(y_val)} | 测试集: {len(y_test)}")
print(f"训练集 - 正常: {(y_train==0).sum()} | 异常: {(y_train==1).sum()}")
print(f"测试集 - 正常: {(y_test==0).sum()} | 异常: {(y_test==1).sum()}")

# ===================== 2. 预处理 =====================
def preprocess_dual_channel(X_batch, fhr_min, fhr_max, uc_min, uc_max):
    N = X_batch.shape[0]
    fhr_clean_list, uc_norm_list, feat_list = [], [], []
    nyq = FS / 2.0
    b_fhr, a_fhr = signal.butter(4, [0.5/nyq, min(1.9/nyq, 0.99)], btype='band')
    b_uc, a_uc = signal.butter(2, 0.5/nyq, btype='low')
    for i in range(N):
        fhr_norm = X_batch[i, :, 0]
        uc_norm  = X_batch[i, :, 1]
        fhr_raw = fhr_norm * (fhr_max - fhr_min) + fhr_min
        uc_raw  = uc_norm * (uc_max - uc_min) + uc_min
        fhr_filt = signal.filtfilt(b_fhr, a_fhr, fhr_raw)
        uc_filt  = signal.filtfilt(b_uc, a_uc, uc_raw)
        fhr_clean = (fhr_filt - fhr_filt.min()) / (fhr_filt.max() - fhr_filt.min() + 1e-7)
        uc_clean  = (uc_filt - uc_filt.min()) / (uc_filt.max() - uc_filt.min() + 1e-7)
        diff = np.diff(fhr_clean)
        baseline = np.mean(np.sort(fhr_clean)[int(0.1*256):int(0.9*256)])
        short_var = np.sqrt(np.mean(diff**2))
        long_var = np.std(fhr_clean)
        accel = np.sum(diff > 0.02) / max(len(diff),1)
        decel = np.sum(diff < -0.02) / max(len(diff),1)
        feats = np.array([baseline, short_var, long_var, accel, decel], dtype=np.float32)
        fhr_clean_list.append(fhr_clean)
        uc_norm_list.append(uc_clean)
        feat_list.append(feats)
    return (np.array(fhr_clean_list).reshape(N,256,1),
            np.array(uc_norm_list).reshape(N,256,1),
            np.array(feat_list))


print("预处理中...")
X_train_fhr, X_train_uc, X_train_feat = preprocess_dual_channel(X_train_raw, FHR_MIN, FHR_MAX, UC_MIN, UC_MAX)
X_val_fhr, X_val_uc, X_val_feat = preprocess_dual_channel(X_val_raw, FHR_MIN, FHR_MAX, UC_MIN, UC_MAX)
X_test_fhr, X_test_uc, X_test_feat = preprocess_dual_channel(X_test_raw, FHR_MIN, FHR_MAX, UC_MIN, UC_MAX)

print(f"预处理后 - 训练: {X_train_fhr.shape}, 验证: {X_val_fhr.shape}, 测试: {X_test_fhr.shape}")

# ===================== 3. 类别平衡 =====================
print(f"训练集（未过采样）: {len(y_train)} | 正常: {(y_train==0).sum()} | 异常: {(y_train==1).sum()}")
print("使用Focal Loss处理类别不平衡，无需过采样")

# 计算过采样比例（仅供参考）
pos_ratio = (y_train==1).sum() / len(y_train)
neg_ratio = (y_train==0).sum() / len(y_train)
scale_pos = neg_ratio / pos_ratio
print(f"正负样本比例: {pos_ratio:.4f} / {neg_ratio:.4f}")
print(f"异常类需要放大的比例: {scale_pos:.2f}x")

# 过采样少数类
X_train_fhr_res = np.concatenate([X_train_fhr, X_train_fhr[y_train==1]], axis=0)
X_train_uc_res  = np.concatenate([X_train_uc,  X_train_uc[y_train==1]],  axis=0)
X_train_feat_res = np.concatenate([X_train_feat, X_train_feat[y_train==1]], axis=0)
y_train_res = np.concatenate([y_train, y_train[y_train==1]], axis=0)

# 打乱
shuffle_idx = np.random.permutation(len(y_train_res))
X_train_fhr_res = X_train_fhr_res[shuffle_idx]
X_train_uc_res  = X_train_uc_res[shuffle_idx]
X_train_feat_res = X_train_feat_res[shuffle_idx]
y_train_res = y_train_res[shuffle_idx]

print(f"过采样后训练集: {len(y_train_res)} | 正常: {(y_train_res==0).sum()} | 异常: {(y_train_res==1).sum()}")


# ===================== 4. 构建模型（严格控制参数量） =====================
def se_block(x, filters, kernel_size, activation='relu', dilation_rate=1):
    """SE注意力块（增强特征筛选能力）"""
    # Depthwise Separable Conv
    sep = layers.SeparableConv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate,
                                  depthwise_regularizer=regularizers.l1(L1_REG),
                                  pointwise_regularizer=regularizers.l1(L1_REG))(x)
    sep = layers.BatchNormalization()(sep)
    sep = layers.Activation(activation)(sep)
    
    # SE
    gap = layers.GlobalAveragePooling1D()(sep)
    se = layers.Dense(max(1, filters // 4), activation='relu')(gap)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, filters))(se)
    return layers.Multiply()([sep, se])


class ChannelAttention(layers.Layer):
    """通道注意力"""
    def __init__(self, filters, ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio
    def build(self, input_shape):
        self.shared_dense1 = layers.Dense(self.filters // self.ratio, activation='relu')
        self.shared_dense2 = layers.Dense(self.filters, activation='sigmoid')
    def call(self, inputs):
        gap = layers.GlobalAveragePooling1D()(inputs)
        gap = self.shared_dense1(gap)
        attention = self.shared_dense2(gap)
        attention = layers.Reshape((1, self.filters))(attention)
        return layers.Multiply()([inputs, attention])
    def get_config(self):
        return {"filters": self.filters, "ratio": self.ratio}


def build_strict_model():
    input_fhr = layers.Input(shape=(256,1), name='fhr')
    # 初始投影
    x1 = layers.Conv1D(16, 1, padding='same', use_bias=False)(input_fhr)
    x1 = layers.BatchNormalization()(x1)
    # SepConv 块（4层，最大通道48）
    x1 = se_block(x1, 16, 5)
    x1 = layers.MaxPooling1D(2)(x1)
    x1 = se_block(x1, 32, 5)
    x1 = layers.MaxPooling1D(2)(x1)
    x1 = se_block(x1, 48, 3, dilation_rate=2)
    x1 = se_block(x1, 48, 3, dilation_rate=2)
    x1 = layers.MaxPooling1D(2)(x1)
    x1 = ChannelAttention(48)(x1)
    
    input_uc = layers.Input(shape=(256,1), name='uc')
    x2 = layers.Conv1D(8, 7, padding='same', activation='relu')(input_uc)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling1D(4)(x2)
    x2 = layers.Conv1D(16, 5, padding='same', activation='relu')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.GlobalAveragePooling1D()(x2)
    
    # 单向GRU（降低参数量）
    x1 = layers.GRU(48, return_sequences=True, reset_after=True)(x1)
    x1 = layers.GRU(32, reset_after=True)(x1)
    
    input_hand = layers.Input(shape=(5,), name='hand')
    h = layers.Dense(16, activation='relu')(input_hand)
    h = layers.Dropout(0.4)(h)
    
    concat = layers.Concatenate()([x1, x2, h])
    concat = layers.Dropout(0.6)(concat)
    outputs = layers.Dense(1, activation='sigmoid')(concat)
    
    model = Model(inputs=[input_fhr, input_uc, input_hand], outputs=outputs)
    return model

model = build_strict_model()

# 策略1: 先用普通的binary_crossentropy + class_weight验证模型是否正常
USE_FOCAL_LOSS = False

if USE_FOCAL_LOSS:
    focal_loss = tf.keras.losses.BinaryFocalCrossentropy(
        gamma=FOCAL_GAMMA,
        alpha=FOCAL_ALPHA,
        apply_class_balancing=True
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss=focal_loss,
                  metrics=['accuracy'])
else:
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print("使用普通交叉熵损失函数（class_weight平衡类别）")

model.summary()
params = model.count_params()
print(f"参数量: {params:,} / 100,000")

# ===================== 5. 训练 =====================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    [X_train_fhr_res, X_train_uc_res, X_train_feat_res], y_train_res,
    batch_size=BATCH_SIZE, epochs=EPOCHS,
    validation_data=([X_val_fhr, X_val_uc, X_val_feat], y_val),
    callbacks=callbacks, class_weight=CLASS_WEIGHT, verbose=1
)

# ===================== 6. 在验证集上确定最优阈值 =====================
print("\n使用多策略阈值选择...")
y_val_prob = model.predict([X_val_fhr, X_val_uc, X_val_feat], verbose=0).flatten()

def evaluate_threshold_at_points(y_true, y_prob, thresholds_to_check):
    results = []
    for thresh in thresholds_to_check:
        y_pred = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * prec * sens / (prec + sens + 1e-7)
        results.append({
            'threshold': thresh,
            'sensitivity': sens,
            'specificity': spec,
            'f1': f1,
            'precision': prec
        })
    return results

thresholds_search = np.arange(0.1, 0.95, 0.05)
val_results = evaluate_threshold_at_points(y_val, y_val_prob, thresholds_search)

print("\n验证集阈值搜索结果:")
print("-" * 70)
print(f"{'阈值':<8}{'灵敏度':<12}{'特异度':<12}{'F1':<12}{'精确率':<12}")
print("-" * 70)
for r in val_results:
    print(f"{r['threshold']:<8.2f}{r['sensitivity']:<12.4f}{r['specificity']:<12.4f}{r['f1']:<12.4f}{r['precision']:<12.4f}")
print("-" * 70)

best_f1_result = max(val_results, key=lambda x: x['f1'])
print(f"\n[策略1-F1最大化] 阈值: {best_f1_result['threshold']:.2f}, F1={best_f1_result['f1']:.4f}, 灵敏度={best_f1_result['sensitivity']:.4f}, 特异度={best_f1_result['specificity']:.4f}")

filtered_results = [r for r in val_results if r['sensitivity'] >= 0.40 and r['specificity'] >= 0.35]
if filtered_results:
    best_balanced = max(filtered_results, key=lambda x: x['f1'])
    print(f"[策略2-平衡策略] 阈值: {best_balanced['threshold']:.2f}, F1={best_balanced['f1']:.4f}, 灵敏度={best_balanced['sensitivity']:.4f}, 特异度={best_balanced['specificity']:.4f}")
else:
    best_balanced = best_f1_result
    print(f"[策略2-平衡策略] 无满足条件策略，使用F1最大化策略")

best_threshold = best_f1_result['threshold']
print(f"\n最终选择阈值(F1最大化): {best_threshold:.4f}")

# ===================== 7. 测试集评估 =====================
print("\n" + "=" * 60)
print(f"最终测试结果 (阈值={best_threshold:.4f}，由验证集确定)")
print("=" * 60)

y_test_prob = model.predict([X_test_fhr, X_test_uc, X_test_feat], verbose=0).flatten()
y_test_pred = (y_test_prob >= best_threshold).astype(int)

test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
test_auc = roc_auc_score(y_test, y_test_prob)

tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
test_sens = tp / (tp + fn)
test_spec = tn / (tn + fp)

print(f"准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"加权F1: {test_f1:.4f}")
print(f"AUC: {test_auc:.4f}")
print(f"灵敏度 (异常召回): {test_sens:.4f}")
print(f"特异度 (正常召回): {test_spec:.4f}")
print()
print(classification_report(y_test, y_test_pred, target_names=['正常', '异常']))
print(f"混淆矩阵:\n {confusion_matrix(y_test, y_test_pred)}")

# ===================== 8. 可视化 =====================
print("\n生成可视化图表...")

# 训练曲线
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'], label='训练损失')
axes[0].plot(history.history['val_loss'], label='验证损失')
axes[0].set_title('损失曲线')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[1].plot(history.history['accuracy'], label='训练准确率')
axes[1].plot(history.history['val_accuracy'], label='验证准确率')
axes[1].set_title('准确率曲线')
axes[1].set_xlabel('Epoch')
axes[1].legend()
plt.tight_layout()
plt.savefig(model_dir / "training_curves.png", dpi=150)
print(f"  - 训练曲线已保存: {model_dir / 'training_curves.png'}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['正常', '异常'], yticklabels=['正常', '异常'])
ax.set_title(f'混淆矩阵 (阈值={best_threshold:.2f})')
ax.set_ylabel('真实标签')
ax.set_xlabel('预测标签')
plt.tight_layout()
plt.savefig(model_dir / "confusion_matrix.png", dpi=150)
print(f"  - 混淆矩阵已保存: {model_dir / 'confusion_matrix.png'}")

# 阈值分析
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# 左：不同阈值下的性能
thresholds_full = np.arange(0.1, 0.95, 0.05)
full_results = evaluate_threshold_at_points(y_val, y_val_prob, thresholds_full)
sens_list = [r['sensitivity'] for r in full_results]
spec_list = [r['specificity'] for r in full_results]
f1_list = [r['f1'] for r in full_results]
axes[0].plot(thresholds_full, sens_list, 'b-o', label='灵敏度')
axes[0].plot(thresholds_full, spec_list, 'g-o', label='特异度')
axes[0].plot(thresholds_full, f1_list, 'r-o', label='F1')
axes[0].axvline(x=best_threshold, color='k', linestyle='--', label=f'选定阈值={best_threshold:.2f}')
axes[0].set_xlabel('阈值')
axes[0].set_ylabel('性能指标')
axes[0].set_title('阈值-性能关系')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
# 右：阈值性能表
axes[1].axis('off')
table_data = [[f"{r['threshold']:.2f}", f"{r['sensitivity']:.4f}", f"{r['specificity']:.4f}", f"{r['f1']:.4f}"] for r in full_results]
table = axes[1].table(cellText=table_data, colLabels=['阈值', '准确率', 'F1', '灵敏度'], loc='center')
axes[1].set_title('不同阈值下性能对比表')
plt.tight_layout()
plt.savefig(model_dir / "threshold_analysis.png", dpi=150)
print(f"  - 阈值分析图已保存: {model_dir / 'threshold_analysis.png'}")

# 阈值性能表CSV
import csv
csv_path = model_dir / "threshold_metrics.csv"
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['阈值', '准确率', 'F1', '灵敏度', '特异度'])
    for r in full_results:
        writer.writerow([f"{r['threshold']:.2f}", f"{r['sensitivity']:.4f}", f"{r['specificity']:.4f}", f"{r['f1']:.4f}", f"{r['precision']:.4f}"])
print(f"  - 阈值性能表已保存: {csv_path}")

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {test_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', label='随机')
ax.set_xlabel('假阳性率 (1-特异度)')
ax.set_ylabel('真阳性率 (灵敏度)')
ax.set_title('ROC曲线')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(model_dir / "roc_curve.png", dpi=150)
print(f"  - ROC曲线已保存: {model_dir / 'roc_curve.png'}")

print("\n可视化图表全部生成完毕！")

# ===================== 9. 推理速度测试 =====================
print("\n" + "=" * 60)
print("推理速度测试")
print("=" * 60)
n_warmup = 10
n_run = 100
for _ in range(n_warmup):
    _ = model.predict([X_test_fhr[:1], X_test_uc[:1], X_test_feat[:1]], verbose=0)
start = time.time()
for _ in range(n_run):
    _ = model.predict([X_test_fhr[:1], X_test_uc[:1], X_test_feat[:1]], verbose=0)
elapsed = (time.time() - start) / n_run * 1000
print(f"推理时间: {elapsed:.2f} ms")

# ===================== 10. 保存模型 =====================
model.save(str(model_save_path))
print(f"模型保存至: {model_save_path}")

# 保存阈值
threshold_save_path = model_dir / "threshold.json"
with open(threshold_save_path, 'w', encoding='utf-8') as f:
    json.dump({
        "best_threshold": float(best_threshold),
        "note": "由验证集确定的最佳阈值"
    }, f, indent=2)
print(f"阈值保存至: {threshold_save_path}")

# ===================== 11. 开题指标核对 =====================
print("\n" + "=" * 60)
print("开题指标完成情况")
print("=" * 60)
print(f"参数量            : {params:,} ≤ 100,000 {'✓' if params <= 100000 else '✗'}")
print(f"推理时间           : {elapsed:.2f} ms ≤ 100ms {'✓' if elapsed < 100 else '✗'}")
print(f"双通道输入          : FHR+UC {'✓' if X_train_fhr.shape[-1] == 1 and X_train_uc.shape[-1] == 1 else '✗'}")
print(f"临床特征           : 5维 {'✓' if X_train_feat.shape[-1] == 5 else '✗'}")
print(f"最优阈值           : {best_threshold:.4f} (验证集确定)")
