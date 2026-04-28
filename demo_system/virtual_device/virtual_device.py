#!/usr/bin/env python3
"""
虚拟胎心监护设备模拟器
参考 ISO 11073 医疗器械互操作标准的设备元数据规范设计
（数据字段：设备ID、时间戳、序列号、数据通道等）
为适配现代云端架构，采用 JSON/RESTful 传输协议
信号数据格式依据 PhysioNet CTU-CHB 产科数据库规范
"""

import numpy as np
import json
import time
import logging
import requests
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

# ===================== 配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 胎心信号参数 (符合 PhysioNet CTG 数据库生理范围)
FS = 4.0          # 采样率 4Hz
WINDOW_SIZE = 256  # 每次发送256个点 (64秒数据)

# 胎心率正常范围: 110-160 bpm (临床标准)
# 异常范围: <110 或 >160 bpm
FHR_MIN, FHR_MAX = 50, 200
UC_MIN, UC_MAX = 0, 100

# 服务器配置
DEFAULT_SERVER = "http://127.0.0.1:8080/api/fetal/heartbeat"


@dataclass
class FetalSignalConfig:
    """胎心信号配置"""
    signal_type: str = "normal"  # normal, suspicious, abnormal
    
    # 基础胎心率 (bpm)
    base_fhr: float = 140.0
    
    # 胎心率变化幅度
    fhr_variability: float = 10.0
    
    # 基线变异周期 (秒)
    baseline_period: float = 20.0
    
    # 加速阈值 (相对于基线的上升)
    acceleration_threshold: float = 15.0
    acceleration_probability: float = 0.1
    
    # 减速阈值 (相对于基线的下降)
    deceleration_threshold: float = 15.0
    deceleration_probability: float = 0.05
    
    # 宫缩基础值
    base_uc: float = 20.0
    uc_variability: float = 15.0


# ===================== 信号生成 =====================
class FetalSignalGenerator:
    """胎心信号生成器"""
    
    def __init__(self, config: FetalSignalConfig):
        self.config = config
        self.t = 0.0  # 时间追踪
    
    def generate(self, n_points: int = 256) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        生成胎心率和宫缩数据
        
        参数:
            n_points: 生成的点数
        
        返回:
            (fhr_data, uc_data, signal_type)
        """
        config = self.config
        dt = 1.0 / FS
        
        # 胎心率信号
        fhr_data = np.zeros(n_points)
        
        for i in range(n_points):
            # 基线变异 (正弦波)
            baseline_var = config.fhr_variability * np.sin(2 * np.pi * self.t / config.baseline_period)
            
            # 短期变异 (高频噪声)
            short_var = np.random.normal(0, 2.0)
            
            # 加速事件
            accel = 0.0
            if np.random.random() < config.acceleration_probability:
                # 生成一个加速 (胎动反应)
                accel = config.acceleration_threshold * np.exp(-((i - n_points//4) ** 2) / (2 * (n_points//8) ** 2))
            
            # 减速事件
            decel = 0.0
            if np.random.random() < config.deceleration_probability:
                decel = config.deceleration_threshold * np.exp(-((i - n_points//2) ** 2) / (2 * (n_points//8) ** 2))
            
            # 组装信号
            fhr_data[i] = config.base_fhr + baseline_var + short_var + accel - decel
            
            # 限制范围
            fhr_data[i] = np.clip(fhr_data[i], FHR_MIN, FHR_MAX)
            
            self.t += dt
        
        # 宫缩信号 (较慢变化)
        uc_data = np.zeros(n_points)
        uc_phase = np.random.uniform(0, 2 * np.pi)
        
        for i in range(n_points):
            # 宫缩周期性变化
            uc_base = config.base_uc + config.uc_variability * np.sin(2 * np.pi * self.t / 60.0 + uc_phase)
            uc_noise = np.random.normal(0, 3.0)
            uc_data[i] = max(0, uc_base + uc_noise)
            self.t += dt
        
        return fhr_data, uc_data, config.signal_type


# ===================== 虚拟设备 =====================
class VirtualFetalMonitor:
    """虚拟胎心监护设备"""
    
    def __init__(self, server_url: str = DEFAULT_SERVER):
        self.server_url = server_url
        self.is_running = False
        self.signal_count = 0
        
        # 信号生成配置
        self.configs = {
            "normal": FetalSignalConfig(
                signal_type="normal",
                base_fhr=140.0,
                fhr_variability=10.0,
                base_uc=20.0
            ),
            "suspicious": FetalSignalConfig(
                signal_type="suspicious",
                base_fhr=125.0,
                fhr_variability=15.0,
                acceleration_probability=0.15,
                deceleration_probability=0.1,
                base_uc=40.0
            ),
            "abnormal": FetalSignalConfig(
                signal_type="abnormal",
                base_fhr=100.0,
                fhr_variability=5.0,
                deceleration_probability=0.2,
                deceleration_threshold=30.0,
                base_uc=60.0
            )
        }
        
        # 当前使用的生成器
        self.generator = FetalSignalGenerator(self.configs["normal"])
    
    def set_signal_type(self, signal_type: str):
        """设置信号类型"""
        if signal_type in self.configs:
            self.generator = FetalSignalGenerator(self.configs[signal_type])
            logger.info(f"设备信号类型已切换: {signal_type}")
        else:
            logger.warning(f"未知的信号类型: {signal_type}")
    
    def generate_data(self) -> dict:
        """生成一组胎心数据"""
        fhr_data, uc_data, signal_type = self.generator.generate(WINDOW_SIZE)
        
        self.signal_count += 1
        
        # 归一化到 0-1 范围
        fhr_norm = (fhr_data - FHR_MIN) / (FHR_MAX - FHR_MIN)
        uc_norm = (uc_data - UC_MIN) / (UC_MAX - UC_MIN)
        
        return {
            "device_id": "VIRTUAL-FM-001",
            "timestamp": int(time.time() * 1000),
            "sequence": self.signal_count,
            "signal_type": signal_type,
            "fhr": fhr_norm.tolist(),
            "uc": uc_norm.tolist(),
            "fhr_raw": fhr_data.tolist(),
            "uc_raw": uc_data.tolist()
        }
    
    def send_to_server(self, data: dict) -> Optional[dict]:
        """发送数据到服务器"""
        try:
            response = requests.post(
                self.server_url,
                json=data,
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"服务器返回错误: {response.status_code}")
                return None
        except requests.exceptions.ConnectionError:
            logger.error(f"无法连接到服务器: {self.server_url}")
            return None
        except Exception as e:
            logger.error(f"发送数据出错: {str(e)}")
            return None
    
    def run_demo(self, interval: float = 1.0, signal_type: str = "normal", duration: int = 0):
        """
        运行演示模式
        
        参数:
            interval: 发送间隔 (秒)
            signal_type: 信号类型 (normal/suspicious/abnormal)
            duration: 演示持续时间 (0=无限)
        """
        self.set_signal_type(signal_type)
        self.is_running = True
        
        print("\n" + "=" * 60)
        print("虚拟胎心监护设备 - 演示模式")
        print("=" * 60)
        print(f"服务器地址: {self.server_url}")
        print(f"信号类型: {signal_type}")
        print(f"发送间隔: {interval} 秒")
        print(f"采样率: {FS} Hz, 每次 {WINDOW_SIZE} 点 ({WINDOW_SIZE/FS:.0f}秒)")
        print("=" * 60)
        print("\n开始生成胎心数据... (按 Ctrl+C 停止)\n")
        
        start_time = time.time()
        
        try:
            while self.is_running:
                # 生成数据
                data = self.generate_data()
                
                print(f"[{self.signal_count:03d}] 信号类型: {data['signal_type']}")
                print(f"      FHR均值: {np.mean(data['fhr_raw']):.1f} bpm")
                print(f"      UC均值: {np.mean(data['uc_raw']):.1f}")
                
                # 发送到服务器
                result = self.send_to_server(data)
                
                if result:
                    detection = result.get("data", {})
                    risk = detection.get("risk_level", "unknown")
                    prob = detection.get("probability", 0)
                    infer_time = detection.get("inference_time_ms", 0)
                    
                    risk_display = {
                        "normal": "\033[92m正常\033[0m",
                        "suspicious": "\033[93m可疑\033[0m",
                        "high": "\033[91m异常\033[0m"
                    }.get(risk, risk)
                    
                    print(f"      → 检测结果: {risk_display} (概率: {prob:.2%}, 耗时: {infer_time:.1f}ms)")
                else:
                    print(f"      → 服务器未响应或模型未加载")
                
                print()
                
                # 检查持续时间
                if duration > 0 and (time.time() - start_time) >= duration:
                    break
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n设备已停止")
            self.is_running = False


# ===================== 主程序 =====================
def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description='虚拟胎心监护设备模拟器')
    parser.add_argument('--server', '-s', default=DEFAULT_SERVER,
                       help='J2EE 服务器地址')
    parser.add_argument('--interval', '-i', type=float, default=1.0,
                       help='发送间隔(秒)')
    parser.add_argument('--type', '-t', choices=['normal', 'suspicious', 'abnormal'],
                       default='normal', help='信号类型')
    parser.add_argument('--duration', '-d', type=int, default=0,
                       help='演示持续时间(秒), 0=无限')
    
    args = parser.parse_args()
    
    # 创建虚拟设备
    device = VirtualFetalMonitor(server_url=args.server)
    
    # 运行演示
    device.run_demo(
        interval=args.interval,
        signal_type=args.type,
        duration=args.duration
    )


if __name__ == '__main__':
    main()
