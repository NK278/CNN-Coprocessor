#include"conv_weights.h"




static void conv1d_same_1in(
    const float x[L0],                 // [178]
    float y[L1][COUT1]                 // [178][8]
){
    const int pad = K1/2; // 2
    loop2:for (int n = 0; n < L1; ++n) {
//#pragma HLS PIPELINE
        for (int f = 0; f < COUT1; ++f) {
            float acc = conv1_b[f];
            for (int k = 0; k < K1; ++k) {
                int xi = n + k - pad;
                float xv = (xi >= 0 && xi < L0) ? x[xi] : 0.0f;
                acc += xv * conv1_w[f][k][0]; // Cin=1
            }
            y[n][f] = acc;
        }
    }
}

// ---- ReLU for (L1=178, COUT1=8) ----
static void relu_L1_C1(const float x[L1][COUT1], float y[L1][COUT1]) {
   loop3:for (int n = 0; n < L1; ++n)
#pragma HLS PIPELINE
        for (int c = 0; c < COUT1; ++c)
            y[n][c] = (x[n][c] > 0.0f) ? x[n][c] : 0.0f;
}

// ---- MaxPool stride=2 for (L1=178, COUT1=8) -> (LP1=89, COUT1) ----
static void maxpool_L1_C1(const float x[L1][COUT1], float y[LP1][COUT1]) {
   loop4:for (int i = 0; i < LP1; ++i) {
#pragma HLS PIPELINE
        int a = 2*i;
        int b = 2*i + 1;
        for (int c = 0; c < COUT1; ++c) {
            float v0 = x[a][c];
            float v1 = (b < L1) ? x[b][c] : x[a][c];
            y[i][c] = (v0 > v1) ? v0 : v1;
        }
    }
}

// ---- Conv2: SAME padding, fixed dims (L2=89, Cin=8, Cout=12, K2=3) ----
static void conv2_same_fixed(
    const float x[L2][CIN2],     // (89,8)
    float y[L2][COUT2]           // (89,12)
){
    const int pad = K2/2; // 1
   loop5:for (int n = 0; n < L2; ++n) {

        for (int f = 0; f < COUT2; ++f) {
            float acc = conv2_b[f];
            for (int k = 0; k < K2; ++k) {
                int xi = n + k - pad;
                if (xi >= 0 && xi < L2) {
                    for (int c = 0; c < CIN2; ++c) {
                        acc += x[xi][c] * conv2_w[f][k][c]; // conv2_w[12][3][8]
                    }
                }
            }
            y[n][f] = acc;
        }
    }
}

// ---- ReLU for (L2=89, COUT2=12) ----
static void relu_L2_C2(const float x[L2][COUT2], float y[L2][COUT2]) {
   loop6:for (int n = 0; n < L2; ++n)
#pragma HLS PIPELINE
        for (int c = 0; c < COUT2; ++c)
            y[n][c] = (x[n][c] > 0.0f) ? x[n][c] : 0.0f;
}

// ---- MaxPool stride=2 for (L2=89, COUT2=12) -> (LP2=44, COUT2) ----
static void maxpool_L2_C2(const float x[L2][COUT2], float y[LP2][COUT2]) {
   loop7:for (int i = 0; i < LP2; ++i) {
        int a = 2*i;
        int b = 2*i + 1;
        for (int c = 0; c < COUT2; ++c) {
#pragma HLS PIPELINE
            float v0 = x[a][c];
            float v1 = (b < L2) ? x[b][c] : x[a][c];
            y[i][c] = (v0 > v1) ? v0 : v1;
        }
    }
}

// ---- Flatten: (LP2=44, C2=12) -> FLAT_LEN=528 ----
static void flatten_LP2_C2(const float x[LP2][COUT2], float y[FLAT_LEN]) {
    int t = 0;
    loop8:for (int n = 0; n < LP2; ++n)
#pragma HLS PIPELINE
        for (int c = 0; c < COUT2; ++c)
            y[t++] = x[n][c];
}

// ---- FC1: 528 -> 32 ----
static void dense_FC1(const float x[FLAT_LEN], float y[FC1_OUT]) {
    loop9:for (int o = 0; o < FC1_OUT; ++o) {
//#pragma HLS PIPELINE
        float acc = fc1_b[o];
        for (int i = 0; i < FLAT_LEN; ++i) acc += fc1_w[o][i] * x[i];
        y[o] = acc;
    }
}

// ---- FC2: 32 -> 16 ----
static void dense_FC2(const float x[FC1_OUT], float y[FC2_OUT]) {
   loop10: for (int o = 0; o < FC2_OUT; ++o) {
#pragma HLS PIPELINE
        float acc = fc2_b[o];
        for (int i = 0; i < FC1_OUT; ++i) acc += fc2_w[o][i] * x[i];
        y[o] = acc;
    }
}

// ---- FC3: 16 -> 8 ----
static void dense_FC3(const float x[FC2_OUT], float y[FC3_OUT]) {
    loop11:for (int o = 0; o < FC3_OUT; ++o) {
#pragma HLS PIPELINE
        float acc = fc3_b[o];
        for (int i = 0; i < FC2_OUT; ++i) acc += fc3_w[o][i] * x[i];
        y[o] = acc;
    }
}

// ---- FC4 (softmax head): 8 -> 2 ----
static void dense_FC4(const float x[FC3_OUT], float y[FC4_OUT]) {
    loop12:for (int o = 0; o < FC4_OUT; ++o) {
#pragma HLS PIPELINE
        float acc = fc4_b[o];
        for (int i = 0; i < FC3_OUT; ++i) acc += fc4_w[o][i] * x[i];
        y[o] = acc;
    }
}

static void relu_vec(int N, const float x[], float y[]) {
    loop13:for (int i = 0; i < N; ++i) y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
}

static void softmax_vec(int N, const float x[], float p[]) {
    // numerically stable
    float m = x[0];
   loop14: for (int i = 1; i < N; ++i) m = (x[i] > m) ? x[i] : m;
    float s = 0.0f;
    for (int i = 0; i < N; ++i) { p[i] = std::exp(x[i] - m); s += p[i]; }
    float inv = 1.0f / s;
    for (int i = 0; i < N; ++i) p[i] *= inv;
}

static int argmax_vec(int N, const float x[]) {
    int idx = 0; float best = x[0];
    for (int i = 1; i < N; ++i) if (x[i] > best) { best = x[i]; idx = i; }
    return idx;
}
// ===========================================================================

// ============================== TOP KERNEL =================================
// AXI-Stream in: 178 floats (INPUT_LEN). AXI-Stream out: 2 softmax floats
// followed by one more word that carries the argmax (as float).
void cnn_top_axi(hls::stream<axis_t> &s_in, hls::stream<axis_t> &s_out) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=s_out
#pragma HLS INTERFACE axis register both port=s_in
    float x0[L0];                       // (178, 1)
    float c1[L1][COUT1];                // (178, 8)
    float r1[L1][COUT1];                // (178, 8)
    float p1[LP1][COUT1];               // (89, 8)

    float c2[L2][COUT2];                // (89, 12)
    float r2[L2][COUT2];                // (89, 12)
    float p2[LP2][COUT2];               // (44, 12)

    float flat[FLAT_LEN];               // 528
    float f1[FC1_OUT];                  // 32
    float a1[FC1_OUT];                  // after relu
    float f2[FC2_OUT];                  // 16
    float a2[FC2_OUT];
    float f3[FC3_OUT];                  // 8
    float a3[FC3_OUT];
    float logits[FC4_OUT];              // 2
    float probs[FC4_OUT];               // 2

    // ---- Read INPUT_LEN samples from AXI stream ----
   loop1: for (int i = 0; i < L0; ++i) {
        axis_t t = s_in.read();
        x0[i] = t.data;
    }

    // ---- conv1 (Cin=1, Cout=8, K=5) + ReLU + MaxPool(2) ----
    conv1d_same_1in(x0, c1);
    relu_L1_C1(c1, r1);
    maxpool_L1_C1(r1, p1);                  // -> (89,8)

    // ---- conv2 (Cin=8, Cout=12, K=3) + ReLU + MaxPool(2) ----
    conv2_same_fixed(p1, c2);
    relu_L2_C2(c2, r2);
    maxpool_L2_C2(r2, p2);                  // -> (44,12)

    // ---- Flatten (44*12=528) ----
    flatten_LP2_C2(p2, flat);

    // ---- FC1 528->32 + ReLU ----
    dense_FC1(flat,  f1);
    relu_vec(FC1_OUT, f1, a1);

    // ---- FC2 32->16 + ReLU ----
    dense_FC2(a1,    f2);
    relu_vec(FC2_OUT, f2, a2);

    // ---- FC3 16->8 + ReLU ----
    dense_FC3(a2, f3);
    relu_vec(FC3_OUT, f3, a3);

    // ---- FC4 8->2 + Softmax ----
    dense_FC4(a3,    logits);
    softmax_vec(FC4_OUT, logits, probs);
    int pred = argmax_vec(FC4_OUT, probs);

    // ---- Stream out: probs[0], probs[1], then pred as a float ----
    axis_t o0; o0.data = probs[0]; o0.last = 0; s_out.write(o0);
    axis_t o1; o1.data = probs[1]; o1.last = 0; s_out.write(o1);
    axis_t o2; o2.data = static_cast<float>(pred); o2.last = 1; s_out.write(o2);
}


