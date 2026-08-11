# 74283 AI Digital Circuit Fault Diagnosis with IoT Display

> An AI-assisted digital circuit fault diagnosis system that combines machine learning, digital electronics, Streamlit, Wi-Fi, ESP32, and an I²C LCD.

## Overview

This project is a proof-of-concept machine-learning system for diagnosing known anomalies in a **74283 4-bit binary adder** from observed circuit input/output behavior.

The system converts circuit behavior into a **14-feature observation** and uses a **Support Vector Machine (SVM) with an RBF kernel** to classify the anomaly pattern.

The resulting diagnosis is sent from the Streamlit application to an **ESP32 over Wi-Fi** and displayed on a physical **16×2 I²C LCD**.

---

## System Architecture

```text
             74283 Circuit Behavior
                       |
                       v
              14-Feature Observation
                       |
                       v
                 SVM Classifier
                  (RBF Kernel)
                       |
                       v
                Fault Diagnosis
                       |
                       v
              Streamlit Application
                       |
                  Wi-Fi / HTTP
                       |
                       v
                 ESP32-WROOM-32
                       |
                      I²C
                       |
                       v
                  16×2 LCD
