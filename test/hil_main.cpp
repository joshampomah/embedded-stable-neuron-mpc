// HIL (hardware-in-the-loop) solver binary for closed-loop testing.
//
// Reads binary commands from stdin, writes binary responses to stdout.
// Uses the same SCPController + ModelWeights as the test binary.
//
// Protocol (little-endian):
//   Python → C++:
//     0x01 CONFIG: 9×f32 params + i32 maxiters + 10×f32 W_bounds = 81 bytes
//     0x02 SOLVE:  30×f32 z_k + f32 u_prev = 125 bytes
//     0xFF DONE:   exit
//   C++ → Python:
//     0x01/0x02 OK/FAIL: 5×f32 u_optimal + u8 n_iters + u8 ipm_iters + u32 time_us = 27 bytes
//     (CONFIG ACK: single byte 0x01)
//
// Build: cmake --build build-desktop && ./build-desktop/hil_solver

#include "stable_neuron_solver/scp_controller.hpp"
#include "network_weights.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using namespace stable_neuron;

static SCPController g_controller;

static bool read_exact(void* buf, size_t len) {
    size_t total = 0;
    auto* p = static_cast<uint8_t*>(buf);
    while (total < len) {
        size_t n = fread(p + total, 1, len - total, stdin);
        if (n == 0) return false;  // EOF or error
        total += n;
    }
    return true;
}

static void write_exact(const void* buf, size_t len) {
    fwrite(buf, 1, len, stdout);
    fflush(stdout);
}

int main() {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    setvbuf(stdin, nullptr, _IONBF, 0);
    setvbuf(stdout, nullptr, _IONBF, 0);

    while (true) {
        uint8_t cmd;
        if (!read_exact(&cmd, 1)) break;  // EOF

        if (cmd == 0xFF) break;  // DONE

        if (cmd == 0x01) {  // CONFIG
            float params[9];
            int32_t maxiters;
            float w_bounds_f32[N * 2];
            if (!read_exact(params, sizeof(params))) break;
            if (!read_exact(&maxiters, sizeof(maxiters))) break;
            if (!read_exact(w_bounds_f32, sizeof(w_bounds_f32))) break;

            SCPParams scp_params;
            scp_params.Q = static_cast<Scalar>(params[0]);
            scp_params.R = static_cast<Scalar>(params[1]);
            scp_params.R_delta = static_cast<Scalar>(params[2]);
            scp_params.tube_weight = static_cast<Scalar>(params[3]);
            scp_params.beta_0 = static_cast<Scalar>(params[4]);
            scp_params.u_min = static_cast<Scalar>(params[5]);
            scp_params.u_max = static_cast<Scalar>(params[6]);
            scp_params.delta_u_max = static_cast<Scalar>(params[7]);
            scp_params.delta_J_min = static_cast<Scalar>(params[8]);
            scp_params.delta_u_tol = Scalar(0.1) * scp_params.delta_u_max;
            scp_params.maxiters = maxiters;

            Scalar w_bounds_s[N * 2];
            for (int i = 0; i < N * 2; i++)
                w_bounds_s[i] = static_cast<Scalar>(w_bounds_f32[i]);
            g_controller.configure(scp_params, w_bounds_s);

            uint8_t ack = 0x01;
            write_exact(&ack, 1);

        } else if (cmd == 0x02) {  // SOLVE
            float input_f32[N_STATE + 1];  // 30 z_k + 1 u_prev
            if (!read_exact(input_f32, sizeof(input_f32))) break;

            Scalar z_k[N_STATE];
            for (int i = 0; i < N_STATE; i++)
                z_k[i] = static_cast<Scalar>(input_f32[i]);
            Scalar u_prev = static_cast<Scalar>(input_f32[N_STATE]);

            auto t0 = std::chrono::high_resolution_clock::now();
            SCPResult result = g_controller.solve<false>(z_k, u_prev, g_model_weights);
            auto t1 = std::chrono::high_resolution_clock::now();
            uint32_t time_us = static_cast<uint32_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

            // Pack response: status(1) + u_opt(20) + n_iters(1) + ipm_iters(1) + time_us(4) = 27 bytes
            uint8_t status = result.converged ? 0x01 : 0x02;
            float u_opt_f32[N];
            for (int i = 0; i < N; i++)
                u_opt_f32[i] = static_cast<float>(result.u_optimal[i]);
            uint8_t n_iters = static_cast<uint8_t>(result.n_iterations);
            uint8_t ipm_iters = static_cast<uint8_t>(result.last_piqp_iters);

            write_exact(&status, 1);
            write_exact(u_opt_f32, sizeof(u_opt_f32));
            write_exact(&n_iters, 1);
            write_exact(&ipm_iters, 1);
            write_exact(&time_us, sizeof(time_us));
        }
        // Unknown commands are silently ignored
    }

    return 0;
}
