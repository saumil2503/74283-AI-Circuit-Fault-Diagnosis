# 74283 AI Digital Circuit Fault Diagnosis with IoT Display

## Overview

This project is a proof-of-concept machine-learning system for diagnosing known anomalies in a 74283 4-bit binary adder from observed circuit input/output behavior.

The system converts circuit behavior into a 14-feature machine-learning observation and uses an SVM classifier with an RBF kernel to identify the anomaly class that most closely matches the observed behavior.

The ML diagnosis is integrated with an ESP32 over Wi-Fi and displayed on a physical I²C LCD.

---

## System Architecture

```text
              74283 Circuit Data
                      |
                      v
              Python ML Pipeline
                      |
                      v
                 SVM Classifier
                      |
                      v
              Streamlit Dashboard
                      |
                    Wi-Fi
                      |
                      v
                ESP32-WROOM-32
                      |
                    I²C
                      |
                      v
                  LCD Display
