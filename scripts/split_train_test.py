#!/usr/bin/env python3
"""
CTU-chb 数据集划分脚本
按胎儿分组（record_id）划分训练/验证/测试集，防止数据泄露

训练集: 70%, 验证集: 10%, 测试集: 20%
"""
import numpy as np
import json
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path


# ===================== 路径 =====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = DATA_DIR / "windows_w256_s128_dual.npz"
OUTPUT_PATH = DATA_DIR / "train_test_final.npz"

# ===================== 加载数据 =====================
print("加载切窗数据...")
data = np.load(INPUT_PATH)
X = data["X"]
y = data["y"]
record_ids = data["record_ids"]

print(f"总样本: {len(X)} | 形状: {X.shape}")
print(f"正常: {(y==0).sum()} | 异常: {(y==1).sum()}")

# ===================== 按胎儿分组划分 =====================
# 第一步: 划分训练集(70%)和临时集(30%)
print("\n第一步: 划分训练/临时集 (train_size=0.7)...")
gss1 = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
train_idx, temp_idx = next(gss1.split(X, y, groups=record_ids))

# 第二步: 将临时集划分为验证集(1/3≈14%)和测试集(2/3≈20%)
print("第二步: 划分验证/测试集 (val_size=0.333)...")
temp_record_ids = record_ids[temp_idx]
temp_y = y[temp_idx]
gss2 = GroupShuffleSplit(n_splits=1, train_size=0.333, random_state=42)
val_idx_temp, test_idx_temp = next(gss2.split(np.zeros(len(temp_idx)), temp_y, groups=temp_record_ids))

# 映射回原始索引
val_idx = temp_idx[val_idx_temp]
test_idx = temp_idx[test_idx_temp]

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

# ===================== 统计 =====================
train_records = set(record_ids[train_idx])
val_records = set(record_ids[val_idx])
test_records = set(record_ids[test_idx])

print(f"\n{'='*50}")
print(f"训练集: {X_train.shape} | 正常: {(y_train==0).sum()} | 异常: {(y_train==1).sum()}")
print(f"验证集: {X_val.shape} | 正常: {(y_val==0).sum()} | 异常: {(y_val==1).sum()}")
print(f"测试集: {X_test.shape} | 正常: {(y_test==0).sum()} | 异常: {(y_test==1).sum()}")
print(f"训练胎儿数: {len(train_records)} | 验证胎儿数: {len(val_records)} | 测试胎儿数: {len(test_records)}")

# 检查重叠
train_overlap_val = train_records & val_records
train_overlap_test = train_records & test_records
val_overlap_test = val_records & test_records

if train_overlap_val or train_overlap_test or val_overlap_test:
    print("⚠️ 警告：存在重叠胎儿，请检查！")
else:
    print("✅ 无数据泄露！各集合胎儿无重叠")

# ===================== 保存 =====================
np.savez(OUTPUT_PATH,
         X_train=X_train, X_val=X_val, X_test=X_test,
         y_train=y_train, y_val=y_val, y_test=y_test,
         train_record_ids=record_ids[train_idx],
         val_record_ids=record_ids[val_idx],
         test_record_ids=record_ids[test_idx])

# 保存划分信息
split_info = {
    "train_size": 0.7,
    "val_size": 0.1,
    "test_size": 0.2,
    "train_samples": len(X_train),
    "val_samples": len(X_val),
    "test_samples": len(X_test),
    "train_records": len(train_records),
    "val_records": len(val_records),
    "test_records": len(test_records),
    "random_state": 42
}
with open(DATA_DIR / "split_info.json", "w", encoding="utf-8") as f:
    json.dump(split_info, f, indent=2, ensure_ascii=False)

print(f"\n✅ 保存至: {OUTPUT_PATH}")
print(f"✅ 划分信息已保存至: {DATA_DIR / 'split_info.json'}")