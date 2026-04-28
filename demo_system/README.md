# 胎心信号异常检测系统 - J2EE-Python 跨技术栈演示

## 项目概述

本演示系统展示了基于轻量化 CNN-GRU 模型的胎心信号异常检测完整技术链：

```
虚拟设备 → J2EE接口层(Spring Boot) → Python算法层(Flask+TensorFlow)
```

## 目录结构

```
demo_system/
├── python_service/           # Python 算法服务
│   └── flask_api.py         # Flask RESTful API
├── virtual_device/          # 虚拟设备模拟器
│   └── virtual_device.py    # 胎心信号生成器
├── j2ee_service/            # J2EE Spring Boot 项目
│   ├── pom.xml
│   └── src/main/java/com/fetal/monitor/
│       ├── FetalMonitorApplication.java
│       ├── controller/
│       ├── service/
│       ├── entity/
│       └── config/
├── web_demo/                # 前端演示页面
│   └── index.html
├── start_demo.py           # 一键启动脚本
└── 启动说明.bat
```

## 技术架构

### 1. Python 算法服务 (flask_api.py)

基于 Flask 框架的轻量化 RESTful 接口：

- **端口**: 5000
- **主要功能**:
  - 加载训练好的 CNN-GRU 模型 (`strict_29k_model.h5`)
  - 数据预处理（带通滤波、特征提取）
  - 胎心信号异常检测
  - 返回风险等级、置信度、推理时间

**API 接口**:
- `GET /health` - 健康检查
- `POST /api/predict` - 单条预测
- `POST /api/batch_predict` - 批量预测

### 2. J2EE 接口层 (Spring Boot)

基于 Spring Boot 3.2 构建的中间接口层：

- **端口**: 8080
- **主要功能**:
  - 接收虚拟设备数据（符合 ISO 11073-10407 标准）
  - 调用 Python 算法服务
  - 返回标准化检测结果
  - 记录检测历史和统计

**API 接口**:
- `POST /api/fetal/heartbeat` - 接收设备数据
- `GET /api/fetal/status` - 获取系统状态
- `GET /api/fetal/stats` - 获取检测统计
- `GET /api/fetal/history` - 获取检测历史

### 3. 虚拟设备 (virtual_device.py)

模拟符合生理特征的胎心信号：

- **信号类型**: 正常、可疑、异常
- **数据格式**: FHR (256点) + UC (256点)
- **发送间隔**: 可配置（默认1秒）

**使用方式**:
```bash
python virtual_device.py -t normal      # 正常信号
python virtual_device.py -t suspicious   # 可疑信号
python virtual_device.py -t abnormal     # 异常信号
```

### 4. 前端演示页面 (index.html)

基于 HTML5 + Canvas 的可视化演示：

- 实时显示胎心信号波形
- 显示检测结果和风险等级
- 特征可视化展示
- 检测日志实时滚动

## 快速启动

### 方式一：一键启动（推荐）

```bash
# Windows
双击 "启动说明.bat"

# 或直接运行
python start_demo.py
```

### 方式二：手动启动

**1. 启动 Python 算法服务**
```bash
cd demo_system/python_service
pip install flask tensorflow numpy scipy requests
python flask_api.py
```

**2. 启动 J2EE 服务**（新开终端）
```bash
cd demo_system/j2ee_service
mvn spring-boot:run
```

**3. 打开演示页面**
- 直接双击 `web_demo/index.html`
- 或访问 http://127.0.0.1:8080

### 方式三：命令行演示

**1-2. 启动上述两个服务**

**3. 运行虚拟设备**
```bash
cd demo_system/virtual_device
python virtual_device.py -t normal -i 1
```

## 数据流程

```
1. 虚拟设备生成胎心数据 (FHR + UC)
   ↓
2. J2EE 接收设备数据 (JSON格式)
   ↓
3. J2EE 调用 Python API
   ↓
4. Python 执行预处理:
   - 带通滤波 (0.5-1.9 Hz)
   - 手工特征提取 (5维)
   ↓
5. Python 调用 CNN-GRU 模型推理
   ↓
6. Python 返回检测结果
   ↓
7. J2EE 返回标准化响应
   ↓
8. 前端展示结果
```

## 模型信息

- **模型文件**: `models/strict_29k_model.h5`
- **参数量**: ~29,000 (< 100,000 目标)
- **输入**: 双通道 (FHR 256点 + UC 256点) + 5维手工特征
- **输出**: 二分类 (正常/异常)
- **推理时间**: < 50ms

## API 请求示例

**Python 健康检查**:
```bash
curl http://127.0.0.1:5000/health
```

**Python 预测**:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"fhr": [0.6]*256, "uc": [0.2]*256}'
```

**J2EE 检测**:
```bash
curl -X POST http://127.0.0.1:8080/api/fetal/heartbeat \
  -H "Content-Type: application/json" \
  -d '{
    "deviceId": "TEST-001",
    "fhr": [0.6]*256,
    "uc": [0.2]*256
  }'
```

## 依赖环境

- Python 3.8+
- Java JDK 17+
- Maven 3.6+
- TensorFlow 2.x
- Spring Boot 3.2.0

## 论文展示建议

1. **架构图**: 使用系统流程图展示完整技术链
2. **接口说明**: 展示 REST API 的请求/响应格式
3. **演示截图**: 捕获前端页面的运行效果
4. **日志输出**: 展示命令行虚拟设备的输出
5. **性能数据**: 展示推理时间、检测准确率等指标

## 注意事项

1. 确保端口 5000 和 8080 未被占用
2. 首次启动 J2EE 需要下载 Maven 依赖
3. Python 服务需要能够访问模型文件
4. 前端页面需要浏览器支持 ES6+
