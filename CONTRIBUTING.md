# Contributing

Contributions are welcome. Please open an issue before submitting a pull
request for significant changes.

## Development setup

```bash
git clone https://github.com/your-handle/embedded-stable-neuron-mpc.git
cd embedded-stable-neuron-mpc
pip install -e ".[dev]"
pytest tests/ -v
```

## Code style

- Python: follow PEP 8, use type hints where practical.
- C++: follow the existing style (C++17, Eigen, no exceptions/RTTI in firmware).

## Testing

All Python changes must pass `pytest tests/ -v --tb=short`.

C++ changes require a CMake desktop build (`cmake -B build && cmake --build build`).
Cross-compilation for ARM requires the `arm-none-eabi-gcc` toolchain and the
`cmake/arm-none-eabi.cmake` toolchain file.
