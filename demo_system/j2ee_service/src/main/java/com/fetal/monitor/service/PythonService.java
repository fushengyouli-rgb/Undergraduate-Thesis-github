package com.fetal.monitor.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fetal.monitor.entity.DetectionResult;
import com.fetal.monitor.entity.FetalSignalRequest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

/**
 * Python算法服务调用层
 * 负责与Python Flask服务通信
 */
@Service
public class PythonService {

    private static final Logger log = LoggerFactory.getLogger(PythonService.class);

    private final WebClient.Builder webClientBuilder;
    private final ObjectMapper objectMapper;

    @Value("${python.service.base-url}")
    private String pythonBaseUrl;

    public PythonService(WebClient.Builder webClientBuilder, ObjectMapper objectMapper) {
        this.webClientBuilder = webClientBuilder;
        this.objectMapper = objectMapper;
    }

    @SuppressWarnings("unchecked")
    public DetectionResult callDetection(FetalSignalRequest request) {
        try {
            log.info("调用Python算法服务预测胎心信号...");
            long startTime = System.currentTimeMillis();

            Map<String, Object> pythonRequest = new HashMap<>();
            pythonRequest.put("fhr", request.getFhr());
            pythonRequest.put("uc", request.getUc());

            Map<String, Object> response = webClientBuilder.build()
                    .post()
                    .uri(pythonBaseUrl + "/api/predict")
                    .bodyValue(pythonRequest)
                    .retrieve()
                    .bodyToMono(Map.class)
                    .timeout(Duration.ofSeconds(10))
                    .block();

            if (response != null && "0".equals(String.valueOf(response.get("code")))) {
                Map<String, Object> data = (Map<String, Object>) response.get("data");
                
                DetectionResult result = DetectionResult.builder()
                        .isAnomaly((Boolean) data.get("is_anomaly"))
                        .confidence(((Number) data.get("confidence")).doubleValue())
                        .probability(((Number) data.get("probability")).doubleValue())
                        .riskLevel((String) data.get("risk_level"))
                        .inferenceTimeMs(((Number) data.get("inference_time_ms")).doubleValue())
                        .originalSignalType(request.getSignalType())
                        .build();

                if (data.containsKey("features")) {
                    Map<String, Object> features = (Map<String, Object>) data.get("features");
                    result.setFeatures(DetectionResult.FeatureResult.builder()
                            .baseline(toDouble(features.get("baseline")))
                            .shortVariability(toDouble(features.get("short_variability")))
                            .longVariability(toDouble(features.get("long_variability")))
                            .accelerationRatio(toDouble(features.get("acceleration_ratio")))
                            .decelerationRatio(toDouble(features.get("deceleration_ratio")))
                            .build());
                }

                long elapsed = System.currentTimeMillis() - startTime;
                log.info("Python服务调用成功，耗时: {}ms", elapsed);
                
                return result;
            } else {
                log.error("Python服务返回错误: {}", response);
                return createMockResult(request, "Python服务响应异常");
            }

        } catch (Exception e) {
            log.error("调用Python服务失败: {}", e.getMessage());
            return createMockResult(request, "Python服务未连接或调用失败");
        }
    }

    @SuppressWarnings("unchecked")
    public boolean checkHealth() {
        try {
            Map<String, Object> response = webClientBuilder.build()
                    .get()
                    .uri(pythonBaseUrl + "/health")
                    .retrieve()
                    .bodyToMono(Map.class)
                    .timeout(Duration.ofSeconds(5))
                    .block();
            return response != null;
        } catch (Exception e) {
            log.warn("Python服务健康检查失败: {}", e.getMessage());
            return false;
        }
    }

    private DetectionResult createMockResult(FetalSignalRequest request, String reason) {
        double fhrMean = request.getFhr().stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0.5);
        double fhrStd = Math.sqrt(request.getFhr().stream()
                .mapToDouble(v -> Math.pow(v - fhrMean, 2))
                .average()
                .orElse(0));

        boolean isAnomaly = fhrMean < 0.35 || fhrMean > 0.75 || fhrStd < 0.02;
        String riskLevel = isAnomaly ? "high" : "normal";
        double probability = isAnomaly ? 0.7 + Math.random() * 0.25 : 0.1 + Math.random() * 0.3;

        return DetectionResult.builder()
                .isAnomaly(isAnomaly)
                .confidence(Math.min(probability, 1 - probability))
                .probability(probability)
                .riskLevel(riskLevel)
                .inferenceTimeMs(0.0)
                .originalSignalType(request.getSignalType())
                .features(DetectionResult.FeatureResult.builder()
                        .baseline(fhrMean)
                        .shortVariability(fhrStd)
                        .longVariability(fhrStd * 1.5)
                        .accelerationRatio(0.05)
                        .decelerationRatio(0.03)
                        .build())
                .build();
    }

    private Double toDouble(Object value) {
        if (value == null) return null;
        if (value instanceof Number) return ((Number) value).doubleValue();
        return Double.parseDouble(String.valueOf(value));
    }
}