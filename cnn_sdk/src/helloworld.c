/******************************************************************************
*
* Copyright (C) 2009 - 2014 Xilinx, Inc.  All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* Use of the Software is limited solely to applications:
* (a) running on a Xilinx device, or
* (b) that interact with a Xilinx device through a bus or interconnect.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* XILINX  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
* OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
* Except as contained in this notice, the name of the Xilinx shall not be used
* in advertising or otherwise to promote the sale, use or other dealings in
* this Software without prior written authorization from Xilinx.
*
******************************************************************************/

/*
 * helloworld.c: simple test application
 *
 * This application configures UART 16550 to baud rate 9600.
 * PS7 UART (Zynq) is not initialized by this application, since
 * bootrom/bsp configures it to baud rate 115200
 *
 * ------------------------------------------------
 * | UART TYPE   BAUD RATE                        |
 * ------------------------------------------------
 *   uartns550   9600
 *   uartlite    Configurable only in HW design
 *   ps7_uart    115200 (configured by bootrom/bsp)
 */




#include <stdio.h>
#include <stdlib.h>
#include "platform.h"
#include"header.h"
#include "xparameters.h"
#include "xaxidma.h"
#include "xil_printf.h"
#include "xtime_l.h"

//#define N_TEST      1000        // <-- set to number of test samples
//#define INPUT_LEN   178
#define NUM_CLASSES 2

// test data provided somewhere else (DDR / BRAM)
//extern float X_test[N_TEST][INPUT_LEN];     // shape: [N_TEST][178]
//extern unsigned char y_test[N_TEST];        // true label (0 or 1)

// AXI DMA device ID
#define DMA_DEVICE_ID  XPAR_AXI_DMA_0_DEVICE_ID

// global DMA instance
XAxiDma AxiDma;

// simple argmax helper in case you want to recompute in SW
static int argmax2(const float *p)
{
    return (p[0] >= p[1]) ? 0 : 1;
}

int main(void)
{
    int status;

    // ---------------- DMA init ----------------
    XAxiDma_Config *CfgPtr = XAxiDma_LookupConfig(DMA_DEVICE_ID);
    if (!CfgPtr) {
        printf("No config found for DMA %d\r\n", DMA_DEVICE_ID);
        return XST_FAILURE;
    }

    status = XAxiDma_CfgInitialize(&AxiDma, CfgPtr);
    if (status != XST_SUCCESS) {
        printf("DMA initialization failed\r\n");
        return XST_FAILURE;
    }

    if (XAxiDma_HasSg(&AxiDma)) {
        printf("Device configured as SG mode, expected simple mode\r\n");
        return XST_FAILURE;
    }

    printf("DMA init done\r\n");

    // ---------------- Buffers ----------------
    // TX buffer: one input sample (178 floats)
    float tx_buf[INPUT_LEN];
    // RX buffer: 3 floats from CNN IP -> [p0, p1, pred_class_as_float]
    float rx_buf[3];

    // ---------------- Inference loop ----------------
    unsigned int correct = 0;
    XTime t_start, t_end;
    double total_time_us = 0.0;

    for (int idx = 0; idx < N_TEST; ++idx) {

        // copy current sample to TX buffer
        for (int i = 0; i < INPUT_LEN; ++i) {
            tx_buf[i] = input_sample[idx][i];
        }

        // time only first sample (or you can time all – change condition)
        if (idx == 0) {
            XTime_SetTime(0);
            XTime_GetTime(&t_start);
        }

        // Start RX first (device->DMA)
        status = XAxiDma_SimpleTransfer(
                     &AxiDma,
                     (UINTPTR)rx_buf,
                     3 * sizeof(float),
                     XAXIDMA_DEVICE_TO_DMA);
        if (status != XST_SUCCESS) {
            printf("DMA device-to-DMA config failed\r\n");
            return XST_FAILURE;
        }

        // Start TX (DMA->device)
        status = XAxiDma_SimpleTransfer(
                     &AxiDma,
                     (UINTPTR)tx_buf,
                     INPUT_LEN * sizeof(float),
                     XAXIDMA_DMA_TO_DEVICE);
        if (status != XST_SUCCESS) {
            printf("DMA DMA-to-device config failed\r\n");
            return XST_FAILURE;
        }

        // Wait for TX done (MM2S)
        do {
            status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR,
                                     0x04);               // MM2S_DMASR
            status &= 0x00000002;                          // IOC_Irq bit
        } while (status != 0x00000002);

        // Wait for RX done (S2MM)
        do {
            status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR,
                                     0x34);               // S2MM_DMASR
            status &= 0x00000002;
        } while (status != 0x00000002);

        if (idx == 0) {
            XTime_GetTime(&t_end);
            total_time_us = (double)(t_end - t_start) /
                            (COUNTS_PER_SECOND / 1000000.0);
        }

        // ---------------- Read prediction ----------------
        float p0   = rx_buf[0];
        float p1   = rx_buf[1];
        int   pred = (int)(rx_buf[2] + 0.5f);  // cnn_top_axi sends pred as float
        // (optional) sanity: recompute argmax on SW
        // int pred_sw = argmax2(&rx_buf[0]);

        if (pred == (int)y_test[idx]) {
            correct++;
        }
    }

    float accuracy = ((float)correct) / ((float)N_TEST);

    printf("============================================\r\n");
    printf(" CNN hardware inference results\r\n");
    printf(" Test samples       : %d\r\n", N_TEST);
    printf(" Correct predictions: %u\r\n", correct);
    printf(" Accuracy           : %0.4f (%.2f%%)\r\n",
           accuracy, accuracy * 100.0f);
    printf(" Time per inference : %0.3f microseconds (measured on 1st sample)\r\n",
           total_time_us);
    printf("============================================\r\n");

    return 0;
}
