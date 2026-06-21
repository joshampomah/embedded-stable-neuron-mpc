# Contributing

This repository archives the code for a completed research project. We are not
actively accepting external contributions, but bug reports and questions are
welcome via GitHub Issues.

## Code style

- Python: follow PEP 8, use type hints where practical.
- C++: follow the existing style (C++17, Eigen, no exceptions/RTTI in firmware).

## Testing

All Python changes must pass `pytest tests/ -v --tb=short`.

C++ changes require a CMake desktop build (`cmake -B build && cmake --build build`).
Cross-compilation for ARM requires the `arm-none-eabi-gcc` toolchain and the
`cmake/arm-none-eabi.cmake` toolchain file.
