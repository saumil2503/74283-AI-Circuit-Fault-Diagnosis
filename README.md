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
```---

## Key Features

- 74283 4-bit binary adder fault diagnosis
- 14-feature machine-learning representation
- SVM classification with RBF kernel
- Streamlit-based diagnosis interface
- Expected vs observed output comparison
- Wi-Fi communication between computer and ESP32
- ESP32 hardware integration
- I²C LCD fault display
- ML model evaluation
- Physical hardware demonstration

---

## Machine Learning

### Model

**Support Vector Machine (SVM)**

### Kernel

**RBF (Radial Basis Function)**

### Input

The system converts the observed circuit behaviour into a **14-feature machine-learning observation**.

### Evaluation

The final SVM model achieved a reported **70.24% held-out test accuracy**.

The repository contains the trained model and evaluation results.

---

## Fault Classes

The available datasets contain four known anomaly classes:

| Class | Fault Location | Gate Modification |
|---|---|---|
| z2 | Internal node z2 | NAND2 → AND2 |
| z17 | Internal node z17 | AND5 → NAND5 |
| z18 | Internal node z18 | AND2 → NAND2 |
| o1 | Output node o1 | NOR5 → OR5 |

The classifier identifies the anomaly class whose learned behaviour most closely matches the observed circuit behaviour.

---

## 74283 Signal Mapping

The application uses the following dataset-to-74283 signal mapping:

| Dataset Signal | 74283 Signal |
|---|---|
| i1 | A3 |
| i2 | B3 |
| i3 | A2 |
| i4 | B2 |
| i5 | A1 |
| i6 | B1 |
| i7 | Cin |
| i8 | A0 |
| i9 | B0 |
| o1 | S3 |
| o2 | Cout |
| o3 | S2 |
| o4 | S1 |
| o5 | S0 |

---

## Fault Diagnosis Workflow

```text
Circuit Inputs
      |
      v
Expected 74283 Output
      |
      v
Observed Circuit Output
      |
      v
Expected vs Observed Comparison
      |
      v
14-Feature Observation
      |
      v
SVM Classification
      |
      v
Fault Diagnosis
      |
      v
Streamlit Dashboard
      |
      v
Wi-Fi / HTTP
      |
      v
ESP32
      |
      v
I²C LCD
```

---

# Demo & Results

## Streamlit Application

The Streamlit application provides the main interface for applying circuit inputs, calculating the expected 74283 output, entering the observed output, and running the ML-based diagnosis.

![Streamlit Application](screenshots/streamlit%20ui.jpg)

---

## Fault Diagnosis Result

When the observed circuit output differs from the expected 74283 output, the observed feature pattern is classified using the trained SVM model.

Example:

```text
Closest Known Anomaly Class : z2
Likely Fault Location       : Internal node z2
Likely Gate Modification    : NAND2 → AND2
```

![Fault Diagnosis Result](screenshots/test%20result%20anomaly%20streamlit.jpg)

---

## 14-Feature Observation & Signal Mapping

The application displays the 14-feature machine-learning observation together with the dataset signal mapping used for the diagnosis.

![Signal Mapping](screenshots/signal%20mapping%20and%20data%20set%20for%20anomaly%20streamlit.jpg)

---

## ESP32 Hardware

The diagnosis system communicates with an ESP32 over the local Wi-Fi network.

![ESP32 Hardware](screenshots/esp32%20connection.png)

---

## Physical LCD Output

The ESP32 receives the diagnosis and displays the result on the physical I²C LCD.

![LCD Output](screenshots/anomaly%20output%20lcd.png)

---

## ESP32 Integration

The ESP32 acts as the wireless communication and physical display controller.

The Streamlit application sends the diagnosis to the ESP32 using HTTP over the local Wi-Fi network.

The ESP32 then displays the received diagnosis on the I²C LCD.

### I²C Connections

| ESP32 | LCD |
|---|---|
| GPIO 21 | SDA |
| GPIO 22 | SCL |
| GND | GND |
| 5V / VIN | VCC |

---

## Hardware

- ESP32-WROOM-32
- 16×2 I²C LCD
- Jumper wires
- USB cable
- Computer
- Wi-Fi network

---

## Software & Tools

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Arduino IDE
- ESP32 Arduino Core

---

## Repository Structure

```text
74283-AI-Circuit-Fault-Diagnosis/
│
├── dataset/
│   └── 74283 anomaly datasets
│
├── models/
│   └── svm_anomaly_classifier.pkl
│
├── outputs/
│   ├── held_out_test_set.csv
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── svm_classification_report.txt
│   ├── svm_confusion_matrix.png
│   └── test_data.csv
│
├── screenshots/
│   ├── anomaly output lcd.png
│   ├── esp32 connection.png
│   ├── signal mapping and data set for anomaly streamlit.jpg
│   ├── streamlit ui.jpg
│   └── test result anomaly streamlit.jpg
│
├── esp32/
│   └── esp32_lcd_receiver.ino
│
├── app.py
├── find_mapping.py
├── predict.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## FPGA Requirement

**No FPGA is required for the current implementation.**

The ESP32 is used for Wi-Fi communication and physical LCD display.

---

## Limitations

This system classifies observed behaviour among the known anomaly classes represented in the available datasets.

A previously unseen fault may still be classified as one of the known classes. Therefore, the ML prediction should be interpreted as the **closest learned anomaly pattern**, rather than definitive physical fault verification.

---

## Future Improvements

- Automatic physical 74283 signal acquisition
- Additional fault classes
- Hardware-in-the-loop testing
- More realistic noisy datasets
- Real-time fault monitoring
- Automatic test-vector generation
- RTL verification integration

---

## Project Status

### Completed

- [x] Dataset preparation
- [x] 14-feature ML representation
- [x] 74283 signal mapping
- [x] SVM classifier
- [x] Held-out testing
- [x] Streamlit dashboard
- [x] ESP32 Wi-Fi communication
- [x] I²C LCD integration
- [x] Physical LCD demonstration
- [x] GitHub repository

### Future

- [ ] Automatic physical 74283 signal acquisition
- [ ] Additional fault classes
- [ ] Hardware-in-the-loop verification
- [ ] RTL verification integration

---

## Resume Description

**AI-Based 74283 Digital Circuit Fault Diagnosis with IoT Display** — Developed an SVM-based anomaly classifier for a 74283 4-bit adder using 14 circuit I/O features and integrated the Streamlit diagnosis application with an ESP32 over Wi-Fi to display fault information on a physical I²C LCD.













