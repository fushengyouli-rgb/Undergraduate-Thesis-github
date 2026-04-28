package com.fetal.monitor.controller;

import com.fetal.monitor.entity.ApiResponse;
import com.fetal.monitor.entity.DetectionResult;
import com.fetal.monitor.entity.FetalSignalRequest;
import com.fetal.monitor.service.FetalDetectionService;
import com.fetal.monitor.service.PythonService;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * 胎心信号检测控制器
 * 
 * 提供RESTful API接口：
 * - POST /api/fetal/heartbeat - 接收设备数据并返回检测结果
 * - GET  /api/fetal/status   - 获取系统状态
 * - GET  /api/fetal/stats    - 获取检测统计
 * - GET  /api/fetal/history  - 获取检测历史
 */
@RestController
@RequestMapping("/api/fetal")
@CrossOrigin(origins = "*")
public class FetalController {

    private static final Logger log = LoggerFactory.getLogger(FetalController.class);

    private final FetalDetectionService fetalDetectionService;
    private final PythonService pythonService;

    public FetalController(FetalDetectionService fetalDetectionService, PythonService pythonService) {
        this.fetalDetectionService = fetalDetectionService;
        this.pythonService = pythonService;
    }

    @PostMapping("/heartbeat")
    public ResponseEntity<ApiResponse<DetectionResult>> receiveHeartbeat(
            @Valid @RequestBody FetalSignalRequest request) {
        
        log.info("接收到设备数据 - 设备ID: {}, 序列号: {}", 
                request.getDeviceId(), request.getSequence());
        
        DetectionResult result = fetalDetectionService.detect(request);
        
        return ResponseEntity.ok(ApiResponse.success(result));
    }

    @GetMapping("/status")
    public ResponseEntity<ApiResponse<Map<String, Object>>> getStatus() {
        Map<String, Object> status = new HashMap<>();
        status.put("j2ee", "running");
        status.put("pythonConnected", pythonService.checkHealth());
        status.put("serverTime", System.currentTimeMillis());
        
        return ResponseEntity.ok(ApiResponse.success(status));
    }

    @GetMapping("/stats")
    public ResponseEntity<ApiResponse<FetalDetectionService.DetectionStats>> getStats() {
        return ResponseEntity.ok(ApiResponse.success(fetalDetectionService.getStats()));
    }

    @GetMapping("/history")
    public ResponseEntity<ApiResponse<Map<Long, DetectionResult>>> getHistory() {
        return ResponseEntity.ok(ApiResponse.success(fetalDetectionService.getHistory()));
    }

    @DeleteMapping("/history")
    public ResponseEntity<ApiResponse<String>> clearHistory() {
        fetalDetectionService.getHistory().clear();
        return ResponseEntity.ok(ApiResponse.success("历史记录已清除", null));
    }
}