# embedded-stable-neuron-mpc

Embedded C++/Python stability-reduced QP solver for closed-loop deep brain stimulation (DBS).

Implements two MPC controllers targeting the STM32L476RG (Cortex-M4F, 80 MHz, 128 KB SRAM):

| Controller | Model | QP type | Typical solve time |
|---|---|---|---|
| **DCNN-MPC** | Input-Convex Neural Network (5-step horizon) | stability-reduced SCP | ~4 ms |
| **Koopman-MPC** | Lifted-linear dynamics (7-step horizon) | Single QP | ~0.22 ms |

## Repository structure

```
include/stable_neuron_solver/   C++ solver headers
src/                        C++ solver sources
firmware/                   STM32 bare-metal firmware
test/                       HIL test binaries (stdin/stdout protocol)
generated/                  Model config headers + demo weight headers
python/embedded_mpc/        Python tooling (profiling, benchmarking)
tests/                      Python pytest suite
cmake/                      ARM cross-compilation toolchain file
doc/                        Accompanying project report (PDF)
```

## Quick start: Python tooling

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Optional PIQP backend:

```bash
pip install -e ".[dev,piqp]"
```

## Quick start: C++ (desktop)

```bash
cmake -B build -DUSE_CUSTOM_IPM=ON
cmake --build build -j4
./build/hil_solver   # stdin/stdout HIL protocol
```

## Quick start: ARM firmware

Requires `arm-none-eabi-gcc` toolchain:

```bash
cmake -B build-arm \
  -DCMAKE_TOOLCHAIN_FILE=cmake/arm-none-eabi.cmake \
  -DEMBEDDED_TARGET=ON \
  -DUSE_CUSTOM_IPM=ON
cmake --build build-arm -j4
# Flash with OpenOCD: see cmake/arm-none-eabi.cmake for target config
```

## Key design decisions

- **Stable neuron elimination**: Interval arithmetic over rate-constrained control bounds classifies ~96% of ICNN neurons as stable-active or stable-inactive, reducing the QP from ~670 variables to ~25-50.
- **Custom Mehrotra IPM**: A hand-coded interior point method (no dynamic allocation, no exceptions) runs the stability-reduced QP on bare metal in 4 ms.
- **Koopman linearization**: 46 LASSO-selected analytical features lift the 30-dim state to a space where dynamics are (approximately) linear, enabling a single QP per timestep.
- **float32 throughout**: All embedded computations use `float` (FPU-accelerated VMLA.F32 on Cortex-M4F).

## Architecture

For a code-level tour — project context, both controller pipelines, an
interior-point methods primer from log-barrier through PMM, and a
feature-by-feature mapping of PIQP to the custom IPM — see
[`ARCHITECTURE.md`](ARCHITECTURE.md). That document is the right
starting point for a code review; the `doc/` PDF covers experiments,
results, and full derivations.

## Important: demo weights

The weight files in `generated/` contain **randomly generated values** and will
not produce meaningful control outputs. See `DISCLAIMER.md`.

## Report

The accompanying 4YP report, *Closed-loop Deep Brain Stimulation Using
Machine Learning for Treatment of Parkinson's Disease*, is in
[`doc/ampomah_4yp_closed_loop_dbs.pdf`](doc/ampomah_4yp_closed_loop_dbs.pdf).
It derives the controllers, the stable-neuron elimination, and the
float32 Mehrotra+PMM solver that this repo implements; section
references in the code (e.g. `§modeling:koopman`, `§implementation:ipm`)
point into it.

## License

MIT — see `LICENSE`.
