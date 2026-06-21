# embedded-stable-neuron-mpc

Embedded C++ and Python implementation of the project MPC solvers for
closed-loop deep brain stimulation (DBS).

This repository is the systems/firmware part of the public code release. It
contains the C++ solver code, STM32 bare-metal firmware, hardware-in-the-loop
test programs, Python reference tooling, demo-safe generated weights, and the
final project report PDF.

This code is a research prototype. It is not a medical device and must not be
used for clinical decision-making or patient treatment. See
[DISCLAIMER.md](DISCLAIMER.md).

## Repository Set

The project is split by responsibility:

| Repository | Purpose |
|---|---|
| [closed-loop-dbs-bench](https://github.com/joshampomah/closed-loop-dbs-bench) | Shared benchmark, synthetic DBS plant, metrics, plotting utilities, bang-bang/PI/linear baselines |
| [dcnn-tube-mpc-dbs](https://github.com/joshampomah/dcnn-tube-mpc-dbs) | DC neural network tube MPC method: predictor, SCP controller, uncertainty bounds, synthetic training/demo code |
| [koopman-mpc-dbs](https://github.com/joshampomah/koopman-mpc-dbs) | Koopman MPC method: lifted-linear predictor, dense QP builder, OLS training/demo code |
| [embedded-stable-neuron-mpc](https://github.com/joshampomah/embedded-stable-neuron-mpc) | C++/STM32 implementation of the stable-neuron and Koopman QP solvers, plus the final report PDF |

Use the method repos to understand the Python controller designs. Use this repo
to inspect how those controllers were reduced to an embedded C++/STM32
implementation.

## What Is In This Repo

- `include/stable_neuron_solver/`: public C++ headers for the solver library.
- `src/`: C++ implementation of ICNN forward passes, Jacobians, stable-neuron
  classification, QP assembly, the custom IPM, and the Koopman controller.
- `firmware/`: STM32L476RG bare-metal firmware entry points and board support.
- `test/`: desktop/HIL-style C++ test binaries using stdin/stdout protocols.
- `generated/`: model config headers and demo-safe generated weight headers.
- `python/embedded_mpc/`: Python reference/profiling solver utilities.
- `tests/`: pytest coverage for the Python tooling.
- `cmake/`: ARM cross-compilation toolchain file.
- `ARCHITECTURE.md`: code-level tour of the solver design and implementation.
- `doc/ampomah_4yp_closed_loop_dbs.pdf`: final A4 4YP report PDF.

## What Is Not In This Repo

- No patient recordings.
- No patient-trained weights.
- No private report source or experiment archive.
- No full Python benchmark suite; that lives in `closed-loop-dbs-bench`.

The files in `generated/` and the firmware reference test cases contain
synthetic randomly generated demo values. They are included so the code can
compile and exercise the solver structure, not to reproduce the project's
patient-data results.

## Implemented Controllers

| Controller | Model | Embedded optimization problem |
|---|---|---|
| DCNN-MPC | Input-convex neural network, 5-step horizon | Stability-reduced SCP QP |
| Koopman-MPC | Lifted-linear model, 7-step horizon | Single dense QP |

The target board used in the project was the STM32L476RG Cortex-M4F
(80 MHz, 128 KB SRAM).

## Installation: Python Tooling

Requires Python 3.10-3.12.

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Optional PIQP support:

```bash
pip install -e ".[dev,piqp]"
```

## Build: Desktop C++

```bash
cmake -B build -DUSE_CUSTOM_IPM=ON
cmake --build build -j4
./build/hil_solver
```

`hil_solver` uses a simple stdin/stdout protocol intended for hardware-in-the-
loop style testing.

## Build: STM32 Firmware

Requires an `arm-none-eabi-gcc` toolchain.

```bash
cmake -B build-arm \
  -DCMAKE_TOOLCHAIN_FILE=cmake/arm-none-eabi.cmake \
  -DEMBEDDED_TARGET=ON \
  -DUSE_CUSTOM_IPM=ON
cmake --build build-arm -j4
```

Flashing/debugging depends on the local OpenOCD/ST-Link setup; see
`cmake/arm-none-eabi.cmake` for the expected target configuration.

## Key Design Points

- Stable-neuron elimination uses interval arithmetic over rate-constrained
  control bounds to classify ICNN neurons as stable-active or stable-inactive.
  This reduces the QP size before solving.
- The custom solver is a float32 Mehrotra-style interior-point method with no
  dynamic allocation and no exceptions in the embedded path.
- The Koopman controller uses a 46-feature analytical lift so the MPC step is a
  single QP.
- The C++ implementation is intentionally explicit and fixed-size where
  possible, reflecting the STM32 memory/performance constraints.

## How This Relates To The Python Repos

- `dcnn-tube-mpc-dbs` contains the Python DCNN/SCP controller logic. This repo
  contains the C++ embedded version of the same controller family using
  demo-safe generated weights.
- `koopman-mpc-dbs` contains the Python Koopman model and dense QP controller.
  This repo contains the C++ embedded implementation of the Koopman path.
- `closed-loop-dbs-bench` is the common benchmark harness. This repo is for
  firmware and solver implementation details, not the main simulation study.

## Report And Architecture Notes

Start with [ARCHITECTURE.md](ARCHITECTURE.md) for a code-level tour of the
embedded implementation.

The final 4YP report is included at
[`doc/ampomah_4yp_closed_loop_dbs.pdf`](doc/ampomah_4yp_closed_loop_dbs.pdf).
It contains the derivations, experiment discussion, and full project context.

## Tests

```bash
pytest tests/ -v
```

The GitHub Actions workflow runs the Python tests. Hardware-in-the-loop tests
require the physical STM32 setup and are not part of CI.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
