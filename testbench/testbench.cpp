
// ============================== Testbench ===================================
// This is only to show end-to-end flow using your header input.
// For synthesis, set cnn_top_axi as the top function and ignore main().
//#ifdef HLS_TESTBENCH
#define HLS_TESTBENCH
#include "input_sample.h"

#include "conv_weights.h"

void cnn_top_axi(hls::stream<axis_t> &s_in, hls::stream<axis_t> &s_out);

int main() {

    hls::stream<axis_t> in, out;

    // Push the header input into the AXI stream
    for (int i=0; i<INPUT_LEN; ++i) {
        axis_t t;
        t.data = input_sample2[i];
        t.last = (i==INPUT_LEN-1);
        in.write(t);
    }
//    inputPort.write(localWrite);
    // Run kernel
    cnn_top_axi(in, out);

    // Read back softmax + prediction
    axis_t r0 = out.read();
    axis_t r1 = out.read();
    axis_t r2 = out.read();

    float p0 = r0.data;
    float p1 = r1.data;
    int   y  = static_cast<int>(r2.data);

    std::printf("Softmax: [%.6f, %.6f]\n", p0, p1);
    std::printf("Predicted class: %d\n", y);
    return 0;
}

//#endif
// ===========================================================================

