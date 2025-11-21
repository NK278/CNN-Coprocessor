# ✨ 1D CNN Coprocessor — FPGA Hardware Accelerator

![Vivado](https://img.shields.io/badge/Vivado-Design%20Suite-AA0000?logo=xilinx&logoColor=white)
![Vitis HLS](https://img.shields.io/badge/Vitis%20HLS-Hardware%20C%2B%2B-00599C?logo=c%2B%2B&logoColor=white)
![Zynq-7000 SoC](https://img.shields.io/badge/Zynq-7000%20SoC-FF6F00?logo=xilinx&logoColor=white)
![Target FPGA](https://img.shields.io/badge/FPGA-Accelerator-4B275F)
![Language](https://img.shields.io/badge/Language-C%2B%2B-00599C)
![Domain](https://img.shields.io/badge/Domain-Edge%20AI%20%2F%20Epilepsy%20Detection-2E7D32)


High-performance **1D Convolutional Neural Network (CNN)** accelerator implemented in **Vitis HLS** and deployed on a **Zynq-7000 SoC**.
The design converts a trained CNN into a fully synthesizable hardware IP, supporting **real-time epilepsy detection**.

---

## 🚀 Key Features

* ⚡ **End-to-end hardware CNN pipeline:**
  *Conv → ReLU → MaxPool → Flatten → Dense → Softmax*

* 🎯 **II = 1 pipelined execution** across all stages
  *(Conv, ReLU, Pool, Flatten, FC)*

* 🧩 **32-way parallel MAC engine** using `UNROLL + ARRAY_PARTITION`

* 💾 All intermediate activations stored in **on-chip BRAM** (no external DRAM)

* 🔌 **AXI-Stream** interfacing with **AXI-DMA** for high-speed input/output

* 🧪 Full **CSIM**, **RTL Synthesis**, and **Processor-in-Loop (PIL)** verification

* 📐 Achieves **9.098 ns** clock period (Target: 10 ns)

---

## 📁 Project Structure

```
├── cnn_sdk/              # PS-side ARM application (DMA config + inference)
├── model/                # Trained CNN (weights, parameters, preprocessing)
├── report/               # Full project report (architecture, analysis, tables)
├── sources/              # Vitis HLS hardware sources (.cpp / .h)
├── testbench/            # HLS C-simulation testbench
│
├── Project Report_ Hardware Impleme...   # Original PDF report uploaded
├── solution6.log         # HLS solution build log
├── vivado_hls.log        # Vivado HLS synthesis log
└── README.md             # Project documentation

```

---

## 🧠 Hardware Architecture Overview

### 🔹 CNN Top Block

<img src="https://github.com/user-attachments/assets/43f79ba7-5a6f-476f-bb6a-61838faa5cf4" width="100%">

### 🔹 Vivado Block Design

<img src="https://github.com/user-attachments/assets/6740c748-b49e-4449-b929-f4d64fa86335" width="75%">

---

## ⚙️ HLS Optimizations

### ⏱️ 1. Loop Pipelining (II = 1)

Applied to:

* Convolution loops
* ReLU
* MaxPool
* Flatten
* Dense layers

➡️ Achieves **1 output per clock cycle**.

---

### 🔀 2. Parallel MAC Engine

Used in Fully Connected Layer (FC1):

```cpp
#pragma HLS ARRAY_PARTITION complete
#pragma HLS UNROLL
```

➡️ Computes **32 outputs in parallel**.

---

### 🧱 3. On-chip Memory Optimization

* All intermediate feature maps stored in **BRAM**
* Minimizes latency
* Removes DRAM bottlenecks
* Ensures deterministic timing

---

### 🔌 4. AXI-Streaming + DMA Pipeline

Two-way streaming:

#### 1) PS DDR → DMA MM2S → CNN IP

#### 2) CNN IP → DMA S2MM → DDR → PS

Designed for:

* Continuous low-latency flow
* Zero CPU intervention
* High throughput inference

---

## 📊 Resource Utilization (Vivado HLS)

| Component   | Utilization | Available | Usage |
| ----------- | ----------- | --------- | ----- |
| **BRAM18K** | 94          | 280       | 33%   |
| **DSP48**   | 92          | 220       | 41%   |
| **LUT**     | 21,426      | 53,200    | 40%   |
| **FF**      | 16,626      | 106,400   | 15%   |

🟢 **Achieved Clock:** **9.098 ns** (Target: 10 ns)

---

## 🧪 Verification Workflow

### 1️⃣ C Simulation (CSIM)

* Validates CNN correctness
* Uses `input_sample2[]`
* Ensures functional correctness of:
  *Conv → ReLU → Pool → Flatten → FC → Softmax*

---

### 2️⃣ RTL Synthesis

Outputs include:

* Resource Report
* Latency Breakdown
* Pipeline interval tables
* Timing estimation

---

### 3️⃣ Processor-in-the-Loop (ARM PS Application)

Located in `/cnn_sdk/`.

PS workflow:

1. **DDR → DMA (MM2S)**
2. **CNN IP Execution (PL)**
3. **PL → DMA (S2MM) → DDR**
4. **PS reads predictions**

Outputs:

* `prob(class=0)`
* `prob(class=1)`
* `predicted label`

---

## ⭐ Show Support

If this project helped you, consider **⭐ starring the repository**!

