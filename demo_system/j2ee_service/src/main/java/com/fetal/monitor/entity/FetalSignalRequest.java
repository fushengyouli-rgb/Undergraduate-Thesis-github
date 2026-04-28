package com.fetal.monitor.entity;

import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

import java.util.List;

/**
 * 胎心信号数据请求实体
 * 符合ISO 11073-10407胎儿监护设备标准
 */
public class FetalSignalRequest {

    private String deviceId;
    private Long timestamp;
    private Integer sequence;
    private String signalType;

    @NotNull(message = "胎心率数据不能为空")
    @Size(min = 256, max = 256, message = "胎心率数据必须为256个点")
    private List<Double> fhr;

    @NotNull(message = "宫缩数据不能为空")
    @Size(min = 256, max = 256, message = "宫缩数据必须为256个点")
    private List<Double> uc;

    private List<Double> fhrRaw;
    private List<Double> ucRaw;

    public FetalSignalRequest() {}

    public FetalSignalRequest(String deviceId, Long timestamp, Integer sequence, String signalType,
                              List<Double> fhr, List<Double> uc, List<Double> fhrRaw, List<Double> ucRaw) {
        this.deviceId = deviceId;
        this.timestamp = timestamp;
        this.sequence = sequence;
        this.signalType = signalType;
        this.fhr = fhr;
        this.uc = uc;
        this.fhrRaw = fhrRaw;
        this.ucRaw = ucRaw;
    }

    public String getDeviceId() { return deviceId; }
    public void setDeviceId(String deviceId) { this.deviceId = deviceId; }
    public Long getTimestamp() { return timestamp; }
    public void setTimestamp(Long timestamp) { this.timestamp = timestamp; }
    public Integer getSequence() { return sequence; }
    public void setSequence(Integer sequence) { this.sequence = sequence; }
    public String getSignalType() { return signalType; }
    public void setSignalType(String signalType) { this.signalType = signalType; }
    public List<Double> getFhr() { return fhr; }
    public void setFhr(List<Double> fhr) { this.fhr = fhr; }
    public List<Double> getUc() { return uc; }
    public void setUc(List<Double> uc) { this.uc = uc; }
    public List<Double> getFhrRaw() { return fhrRaw; }
    public void setFhrRaw(List<Double> fhrRaw) { this.fhrRaw = fhrRaw; }
    public List<Double> getUcRaw() { return ucRaw; }
    public void setUcRaw(List<Double> ucRaw) { this.ucRaw = ucRaw; }
}