package com.fetal.monitor.service;

import com.fetal.monitor.entity.DetectionResult;
import com.fetal.monitor.entity.FetalSignalRequest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * 胎心信号检测服务
 */
@Service
public class FetalDetectionService {

    private static final Logger log = LoggerFactory.getLogger(FetalDetectionService.class);

    private final PythonService pythonService;
    
    private final ConcurrentHashMap<Long, DetectionResult> detectionHistory = new ConcurrentHashMap<>();
    private final AtomicLong recordIdCounter = new AtomicLong(1);

    public FetalDetectionService(PythonService pythonService) {
        this.pythonService = pythonService;
    }

    public DetectionResult detect(FetalSignalRequest request) {
        log.info("开始检测 - 设备: {}, 序列号: {}", 
                request.getDeviceId(), request.getSequence());

        DetectionResult result = pythonService.callDetection(request);

        long recordId = recordIdCounter.getAndIncrement();
        if (result.getInferenceTimeMs() == null) {
            result.setInferenceTimeMs(0.0);
        }
        detectionHistory.put(recordId, result);

        if (detectionHistory.size() > 100) {
            detectionHistory.remove(detectionHistory.keys().nextElement());
        }

        log.info("检测完成 - 风险等级: {}, 概率: {}", 
                result.getRiskLevel(), result.getProbability());

        return result;
    }

    public ConcurrentHashMap<Long, DetectionResult> getHistory() {
        return detectionHistory;
    }

    public DetectionStats getStats() {
        int total = detectionHistory.size();
        if (total == 0) {
            return new DetectionStats(0, 0, 0, 0, 0.0);
        }

        int normal = (int) detectionHistory.values().stream()
                .filter(r -> "normal".equals(r.getRiskLevel()))
                .count();
        int suspicious = (int) detectionHistory.values().stream()
                .filter(r -> "suspicious".equals(r.getRiskLevel()))
                .count();
        int high = (int) detectionHistory.values().stream()
                .filter(r -> "high".equals(r.getRiskLevel()))
                .count();

        double avgInferenceTime = detectionHistory.values().stream()
                .mapToDouble(r -> r.getInferenceTimeMs() != null ? r.getInferenceTimeMs() : 0)
                .average()
                .orElse(0);

        return new DetectionStats(total, normal, suspicious, high, avgInferenceTime);
    }

    public record DetectionStats(
            int total,
            int normal,
            int suspicious,
            int high,
            double avgInferenceTimeMs
    ) {}
}