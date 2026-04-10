// HIL solver binary for Koopman MPC closed-loop testing.
//
// Protocol (little-endian, same structure as DCNN HIL for tool reuse):
//   Python → C++:
//     0x01 CONFIG: 6×f32 params (Q, R, beta_0, u_min, u_max, delta_u_max)
//     0x02 SOLVE:  30×f32 z_k + f32 u_prev = 124 bytes
//     0xFF DONE:   exit
//   C++ → Python:
//     CONFIG ACK: single byte 0x01
//     SOLVE resp: 7×f32 u_optimal + u8 qp_iters + u32 time_us = 33 bytes

#include "stable_neuron_solver/koopman_controller.hpp"
#include "koopman_weights.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace stable_neuron;

static KoopmanController g_controller;

static bool read_exact(void* buf, size_t len) {
    size_t total = 0;
    auto* p = static_cast<uint8_t*>(buf);
    while (total < len) {
        size_t n = fread(p + total, 1, len - total, stdin);
        if (n == 0) return false;
        total += n;
    }
    return true;
}

static void write_exact(const void* buf, size_t len) {
    fwrite(buf, 1, len, stdout);
    fflush(stdout);
}

int main() {
    setvbuf(stdin, nullptr, _IONBF, 0);
    setvbuf(stdout, nullptr, _IONBF, 0);

    while (true) {
        uint8_t cmd;
        if (!read_exact(&cmd, 1)) break;

        if (cmd == 0xFF) break;

        if (cmd == 0x01) {  // CONFIG
            float params[6];
            if (!read_exact(params, sizeof(params))) break;

            KoopmanParams kp;
            kp.Q = static_cast<KScalar>(params[0]);
            kp.R = static_cast<KScalar>(params[1]);
            kp.beta_0 = static_cast<KScalar>(params[2]);
            kp.u_min = static_cast<KScalar>(params[3]);
            kp.u_max = static_cast<KScalar>(params[4]);
            kp.delta_u_max = static_cast<KScalar>(params[5]);
            g_controller.configure(kp);

            uint8_t ack = 0x01;
            write_exact(&ack, 1);

        } else if (cmd == 0x02) {  // SOLVE
            float input_f32[KN_STATE + 1];  // 30 z_k + 1 u_prev
            if (!read_exact(input_f32, sizeof(input_f32))) break;

            KScalar z_k[KN_STATE];
            for (int i = 0; i < KN_STATE; i++)
                z_k[i] = static_cast<KScalar>(input_f32[i]);
            KScalar u_prev = static_cast<KScalar>(input_f32[KN_STATE]);

            auto t0 = std::chrono::high_resolution_clock::now();
            KoopmanResult result = g_controller.solve<false>(z_k, u_prev, g_koopman_weights);
            auto t1 = std::chrono::high_resolution_clock::now();
            uint32_t time_us = static_cast<uint32_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

            uint8_t status = result.is_feasible ? 0x01 : 0x02;
            float u_opt_f32[KN];
            for (int i = 0; i < KN; i++)
                u_opt_f32[i] = static_cast<float>(result.u_optimal[i]);
            uint8_t qp_iters = static_cast<uint8_t>(result.qp_iters);

            write_exact(&status, 1);
            write_exact(u_opt_f32, sizeof(u_opt_f32));
            write_exact(&qp_iters, 1);
            write_exact(&time_us, sizeof(time_us));
        }
    }

    return 0;
}
