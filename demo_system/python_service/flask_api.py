#!/usr/bin/env python3
"""
胎心信号异常检测 - Python 算法服务
基于 Flask 的轻量化接口，为 J2EE 提供算法调用能力

功能：
- 加载训练好的 CNN-GRU 模型
- 提供数据预处理接口
- 提供胎心信号异常检测接口
- 返回检测结果及置信度
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
import json
import time
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from scipy import signal

# ===================== 配置 =====================
app = Flask(__name__)

# 设置UTF-8编码
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 路径配置 - 修正为项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "strict_29k_model.h5"
NORM_PARAMS_PATH = PROJECT_ROOT / "data" / "processed" / "norm_params.json"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.json"

# 采样率配置
FS = 4.0
INPUT_LEN = 256

# 全局变量
model = None
threshold = 0.5
norm_params = {
    "fhr_min": 50, "fhr_max": 200,
    "uc_min": 0, "uc_max": 100
}

# ===================== 自定义层定义 =====================
class ChannelAttention(layers.Layer):
    """与训练脚本中保持一致的自定义层"""
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
        config = super().get_config()
        config.update({'filters': self.filters, 'ratio': self.ratio})
        return config

# 注册自定义对象（使用filters参数名）
tf.keras.utils.get_custom_objects()['ChannelAttention'] = ChannelAttention

# ===================== 模型加载 =====================
def load_model_and_params():
    """加载模型和归一化参数"""
    global model, threshold, norm_params
    
    logger.info("正在加载模型和参数...")
    
    # 加载归一化参数
    if MODEL_PATH.exists():
        with open(NORM_PARAMS_PATH, "r", encoding="utf-8") as f:
            norm_params = json.load(f)
        logger.info(f"归一化参数: FHR=[{norm_params['fhr_min']},{norm_params['fhr_max']}], UC=[{norm_params['uc_min']},{norm_params['uc_max']}]")
    
    # 加载阈值
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            threshold_info = json.load(f)
            threshold = threshold_info.get("best_threshold", 0.5)
        logger.info(f"检测阈值: {threshold}")
    
    # 加载模型
    if MODEL_PATH.exists():
        model = models.load_model(MODEL_PATH)
        model.summary(print_fn=logger.info)
        logger.info(f"模型加载成功: {MODEL_PATH}")
        
        # 统计参数量
        params = model.count_params()
        logger.info(f"模型参数量: {params:,}")
    else:
        logger.warning(f"模型文件不存在: {MODEL_PATH}")
        logger.warning("将使用模拟模式进行演示")

# ===================== 预处理 =====================
def preprocess_dual_channel(fhr_data, uc_data):
    """
    预处理胎心信号数据
    
    参数:
        fhr_data: 胎心率数据列表 (256个点)
        uc_data: 宫缩数据列表 (256个点)
    
    返回:
        预处理后的 (fhr_clean, uc_clean, hand_features)
    """
    fhr_norm = np.array(fhr_data, dtype=np.float32)
    uc_norm = np.array(uc_data, dtype=np.float32)
    
    # 反归一化
    fhr_raw = fhr_norm * (norm_params["fhr_max"] - norm_params["fhr_min"]) + norm_params["fhr_min"]
    uc_raw = uc_norm * (norm_params["uc_max"] - norm_params["uc_min"]) + norm_params["uc_min"]
    
    # 滤波器设计
    nyq = FS / 2.0
    b_fhr, a_fhr = signal.butter(4, [0.5/nyq, min(1.9/nyq, 0.99)], btype='band')
    b_uc, a_uc = signal.butter(2, 0.5/nyq, btype='low')
    
    # 滤波处理
    fhr_filt = signal.filtfilt(b_fhr, a_fhr, fhr_raw)
    uc_filt = signal.filtfilt(b_uc, a_uc, uc_raw)
    
    # 归一化
    fhr_clean = (fhr_filt - fhr_filt.min()) / (fhr_filt.max() - fhr_filt.min() + 1e-7)
    uc_clean = (uc_filt - uc_filt.min()) / (uc_filt.max() - uc_filt.min() + 1e-7)
    
    # 手工特征提取
    diff = np.diff(fhr_clean)
    baseline = np.mean(np.sort(fhr_clean)[int(0.1*256):int(0.9*256)])
    short_var = np.sqrt(np.mean(diff**2))
    long_var = np.std(fhr_clean)
    accel = np.sum(diff > 0.02) / max(len(diff), 1)
    decel = np.sum(diff < -0.02) / max(len(diff), 1)
    
    hand_features = np.array([baseline, short_var, long_var, accel, decel], dtype=np.float32)
    
    return fhr_clean, uc_clean, hand_features

# ===================== 预测 =====================
def predict_anomaly(fhr_data, uc_data):
    """
    预测胎心信号是否异常
    
    参数:
        fhr_data: 胎心率数据 (256点)
        uc_data: 宫缩数据 (256点)
    
    返回:
        {
            "is_anomaly": bool,
            "confidence": float,  # 置信度 (0-1)
            "probability": float,  # 异常概率
            "threshold": float,  # 使用的阈值
            "risk_level": str,  # 风险等级: normal/suspicious/high
            "features": dict  # 提取的特征
        }
    """
    global model, threshold, norm_params
    
    start_time = time.time()
    
    # 预处理
    fhr_clean, uc_clean, hand_features = preprocess_dual_channel(fhr_data, uc_data)
    
    # 如果模型存在，进行真实预测
    if model is not None:
        # 准备输入
        fhr_input = fhr_clean.reshape(1, 256, 1).astype(np.float32)
        uc_input = uc_clean.reshape(1, 256, 1).astype(np.float32)
        feat_input = hand_features.reshape(1, 5).astype(np.float32)
        
        # 预测
        prob = model.predict([fhr_input, uc_input, feat_input], verbose=0)[0][0]
    else:
        # 模拟模式 - 随机生成结果用于演示
        prob = np.random.random()
    
    # 判定结果
    is_anomaly = prob >= threshold
    
    # 风险等级
    if prob < 0.3:
        risk_level = "normal"
    elif prob < 0.6:
        risk_level = "suspicious"
    else:
        risk_level = "high"
    
    inference_time = (time.time() - start_time) * 1000
    
    return {
        "is_anomaly": bool(is_anomaly),
        "confidence": float(round(prob if is_anomaly else 1-prob, 4)),
        "probability": float(round(prob, 4)),
        "threshold": float(threshold),
        "risk_level": risk_level,
        "inference_time_ms": round(inference_time, 2),
        "features": {
            "baseline": round(float(hand_features[0]), 4),
            "short_variability": round(float(hand_features[1]), 4),
            "long_variability": round(float(hand_features[2]), 4),
            "acceleration_ratio": round(float(hand_features[3]), 4),
            "deceleration_ratio": round(float(hand_features[4]), 4)
        }
    }

# ===================== Flask API =====================
@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold": threshold,
        "params": norm_params
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    胎心信号异常检测接口
    
    请求格式:
    {
        "fhr": [float, ...],  # 256个胎心率值 (归一化0-1或实际值)
        "uc": [float, ...]    # 256个宫缩值 (归一化0-1或实际值)
    }
    
    返回格式:
    {
        "code": 0,
        "message": "success",
        "data": {
            "is_anomaly": bool,
            "confidence": float,
            "probability": float,
            "risk_level": str,
            "inference_time_ms": float
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"code": 1, "message": "请求数据为空", "data": None})
        
        fhr_data = data.get('fhr')
        uc_data = data.get('uc')
        
        if not fhr_data or not uc_data:
            return jsonify({"code": 1, "message": "缺少 fhr 或 uc 参数", "data": None})
        
        if len(fhr_data) != 256 or len(uc_data) != 256:
            return jsonify({
                "code": 1, 
                "message": f"数据长度错误: fhr={len(fhr_data)}, uc={len(uc_data)}, 需要256个点",
                "data": None
            })
        
        # 执行预测
        result = predict_anomaly(fhr_data, uc_data)
        
        return jsonify({
            "code": 0,
            "message": "success",
            "data": result
        })
        
    except Exception as e:
        logger.error(f"预测出错: {str(e)}")
        return jsonify({"code": 1, "message": f"服务器错误: {str(e)}", "data": None})

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    批量预测接口
    
    请求格式:
    {
        "records": [
            {"fhr": [...], "uc": [...]},
            {"fhr": [...], "uc": [...]},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        records = data.get('records', [])
        
        if not records:
            return jsonify({"code": 1, "message": "没有待处理的记录", "data": None})
        
        results = []
        for i, record in enumerate(records):
            fhr_data = record.get('fhr')
            uc_data = record.get('uc')
            
            if not fhr_data or not uc_data:
                results.append({"index": i, "error": "数据不完整"})
                continue
            
            result = predict_anomaly(fhr_data, uc_data)
            result["index"] = i
            results.append(result)
        
        return jsonify({
            "code": 0,
            "message": f"处理了 {len(results)} 条记录",
            "data": results
        })
        
    except Exception as e:
        logger.error(f"批量预测出错: {str(e)}")
        return jsonify({"code": 1, "message": f"服务器错误: {str(e)}", "data": None})

# ===================== 主程序 =====================
if __name__ == '__main__':
    print("=" * 60)
    print("胎心信号异常检测 - Python 算法服务")
    print("=" * 60)
    
    # 加载模型
    load_model_and_params()
    
    print("\n服务启动配置:")
    print(f"  模型路径: {MODEL_PATH}")
    print(f"  阈值: {threshold}")
    print(f"  采样率: {FS} Hz")
    print(f"  输入长度: {INPUT_LEN} 点")
    print("\n可用接口:")
    print("  GET  /health           - 健康检查")
    print("  POST /api/predict      - 单条预测")
    print("  POST /api/batch_predict - 批量预测")
    print("\n" + "=" * 60)
    
    # 启动服务
    app.run(host='127.0.0.1', port=5000, debug=False)
