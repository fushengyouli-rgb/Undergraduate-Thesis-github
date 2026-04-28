package com.fetal.monitor;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * 胎心信号异常检测系统 - J2EE接口层
 * 
 * 基于Spring Boot构建的轻量化RESTful接口，
 * 负责接收虚拟设备数据、调用Python算法服务、返回检测结果
 */
@SpringBootApplication
public class FetalMonitorApplication {

    public static void main(String[] args) {
        SpringApplication.run(FetalMonitorApplication.class, args);
    }
}
