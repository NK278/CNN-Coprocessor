✨ 1D CNN Coprocessor — FPGA Hardware Accelerator

High-performance 1D Convolutional Neural Network (CNN) accelerator implemented using Vitis HLS and deployed on a Zynq-7000 SoC.
The design converts a trained CNN into a fully synthesizable hardware IP, supporting real-time epilepsy detection.
🚀 Key Features

⚡ End-to-end hardware CNN pipeline: Conv → ReLU → MaxPool → Flatten → Dense → Softmax

🎯 II = 1 pipelined execution across Conv, ReLU, Pool, Flatten, FC

🧩 32-way parallel MAC engine for FC1 via UNROLL + ARRAY_PARTITION

💾 All intermediate maps stored in on-chip BRAM (no external DRAM access)

🔌 AXI-Stream interfacing with AXI-DMA for high-speed data movement

🧪 Complete C Simulation (CSIM), Synthesis, and Processor-in-Loop verification

📐 Satisfies timing at 9.098 ns (Target: 10 ns)

🗂️ Full technical report included in /report/ (contains architecture, analysis, resource tables, screenshots)

├── cnn_sdk/              # PS-side ARM application (DMA config + inference)
├── report/               # Full project report 
├── sources/              # Vitis HLS hardware source files (.cpp/.h)
├── testbench/            # HLS C-simulation testbench
│
├── solution6.log         # HLS solution build log
├── vivado_hls.log        # Vivado HLS synthesis log
└── README.md             # Project documentation 

🧠 Hardware Architecture Overview
🔹 CNN Top Block

 AXI-Stream In                     AXI-Stream Out
        │                                 ↑
        ▼                                 │
 ┌─────────────────┐           ┌──────────────────────────┐
 │   cnn_top_axi    │─────────▶│ probs[0], probs[1], pred │
 └─────────────────┘           └──────────────────────────┘
        │
        ▼
 [Conv1] → [ReLU1] → [Pool1] → [Conv2] → [ReLU2] → [Pool2]
        ▼
     [Flatten]
        ▼
 [FC1] → [FC2] → [FC3] → [FC4] → [Softmax] → [Argmax]


⚙️ HLS Optimizations Used
⏱️ 1. Loop Pipelining (II = 1)

Used across:

Convolution loops

ReLU activation

Pooling

Flatten

Dense layers
Achieves 1 output per clock cycle.

🔀 2. Parallel MAC Engine

#pragma HLS ARRAY_PARTITION complete +
#pragma HLS UNROLL
→ FC1 computes 32 outputs in parallel.

🧱 3. On-chip Memory Optimization

All intermediate activations placed in BRAM.

🔌 4. AXI-Streaming + DMA

Continuous, low-latency streaming between:

 1) PS DDR → DMA → PL CNN IP

 2) PL → DMA → DDR → PS

📊 Resource Utilization (from Vivado HLS)

| Component   | Utilization | Available | %   |
| ----------- | ----------- | --------- | --- |
| **BRAM18K** | 94          | 280       | 33% |
| **DSP48**   | 92          | 220       | 41% |
| **LUT**     | 21,426      | 53,200    | 40% |
| **FF**      | 16,626      | 106,400   | 15% |

Achieved Clock: 9.098 ns (Target: 10 ns) ✔️

🧪 Verification Workflow
1️⃣ C Simulation (CSIM)

Validates end-to-end CNN correctness using input_sample2[].

2️⃣ RTL Synthesis

Generates:

resource report

latency breakdown

pipeline interval tables

timing estimation

3️⃣ Processor-in-the-Loop (PS Application)

Located in /cnn_sdk/.
The PS performs:
DDR → DMA MM2S → CNN IP → DMA S2MM → DDR
Outputs:

 1)prob(class=0)

 2)prob(class=1)

 3)predicted label
 
⭐ Show Support

If this repo helped you, don’t forget to ⭐ star the repository!

