package com.fetal.monitor.entity;

/**
 * 胎心信号检测结果
 */
public class DetectionResult {

    private Boolean isAnomaly;
    private Double confidence;
    private Double probability;
    private String riskLevel;
    private Double inferenceTimeMs;
    private FeatureResult features;
    private String originalSignalType;

    public DetectionResult() {}

    public DetectionResult(Boolean isAnomaly, Double confidence, Double probability, String riskLevel,
                           Double inferenceTimeMs, FeatureResult features, String originalSignalType) {
        this.isAnomaly = isAnomaly;
        this.confidence = confidence;
        this.probability = probability;
        this.riskLevel = riskLevel;
        this.inferenceTimeMs = inferenceTimeMs;
        this.features = features;
        this.originalSignalType = originalSignalType;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private Boolean isAnomaly;
        private Double confidence;
        private Double probability;
        private String riskLevel;
        private Double inferenceTimeMs;
        private FeatureResult features;
        private String originalSignalType;

        public Builder isAnomaly(Boolean isAnomaly) { this.isAnomaly = isAnomaly; return this; }
        public Builder confidence(Double confidence) { this.confidence = confidence; return this; }
        public Builder probability(Double probability) { this.probability = probability; return this; }
        public Builder riskLevel(String riskLevel) { this.riskLevel = riskLevel; return this; }
        public Builder inferenceTimeMs(Double inferenceTimeMs) { this.inferenceTimeMs = inferenceTimeMs; return this; }
        public Builder features(FeatureResult features) { this.features = features; return this; }
        public Builder originalSignalType(String originalSignalType) { this.originalSignalType = originalSignalType; return this; }
        public DetectionResult build() {
            return new DetectionResult(isAnomaly, confidence, probability, riskLevel, inferenceTimeMs, features, originalSignalType);
        }
    }

    public Boolean getIsAnomaly() { return isAnomaly; }
    public void setIsAnomaly(Boolean isAnomaly) { this.isAnomaly = isAnomaly; }
    public Double getConfidence() { return confidence; }
    public void setConfidence(Double confidence) { this.confidence = confidence; }
    public Double getProbability() { return probability; }
    public void setProbability(Double probability) { this.probability = probability; }
    public String getRiskLevel() { return riskLevel; }
    public void setRiskLevel(String riskLevel) { this.riskLevel = riskLevel; }
    public Double getInferenceTimeMs() { return inferenceTimeMs; }
    public void setInferenceTimeMs(Double inferenceTimeMs) { this.inferenceTimeMs = inferenceTimeMs; }
    public FeatureResult getFeatures() { return features; }
    public void setFeatures(FeatureResult features) { this.features = features; }
    public String getOriginalSignalType() { return originalSignalType; }
    public void setOriginalSignalType(String originalSignalType) { this.originalSignalType = originalSignalType; }

    public static class FeatureResult {
        private Double baseline;
        private Double shortVariability;
        private Double longVariability;
        private Double accelerationRatio;
        private Double decelerationRatio;

        public FeatureResult() {}

        public FeatureResult(Double baseline, Double shortVariability, Double longVariability,
                            Double accelerationRatio, Double decelerationRatio) {
            this.baseline = baseline;
            this.shortVariability = shortVariability;
            this.longVariability = longVariability;
            this.accelerationRatio = accelerationRatio;
            this.decelerationRatio = decelerationRatio;
        }

        public static Builder builder() {
            return new Builder();
        }

        public static class Builder {
            private Double baseline;
            private Double shortVariability;
            private Double longVariability;
            private Double accelerationRatio;
            private Double decelerationRatio;

            public Builder baseline(Double baseline) { this.baseline = baseline; return this; }
            public Builder shortVariability(Double shortVariability) { this.shortVariability = shortVariability; return this; }
            public Builder longVariability(Double longVariability) { this.longVariability = longVariability; return this; }
            public Builder accelerationRatio(Double accelerationRatio) { this.accelerationRatio = accelerationRatio; return this; }
            public Builder decelerationRatio(Double decelerationRatio) { this.decelerationRatio = decelerationRatio; return this; }
            public FeatureResult build() {
                return new FeatureResult(baseline, shortVariability, longVariability, accelerationRatio, decelerationRatio);
            }
        }

        public Double getBaseline() { return baseline; }
        public void setBaseline(Double baseline) { this.baseline = baseline; }
        public Double getShortVariability() { return shortVariability; }
        public void setShortVariability(Double shortVariability) { this.shortVariability = shortVariability; }
        public Double getLongVariability() { return longVariability; }
        public void setLongVariability(Double longVariability) { this.longVariability = longVariability; }
        public Double getAccelerationRatio() { return accelerationRatio; }
        public void setAccelerationRatio(Double accelerationRatio) { this.accelerationRatio = accelerationRatio; }
        public Double getDecelerationRatio() { return decelerationRatio; }
        public void setDecelerationRatio(Double decelerationRatio) { this.decelerationRatio = decelerationRatio; }
    }
}