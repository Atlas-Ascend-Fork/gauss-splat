# Running Tests

On Linux with an NPU, configure CANN, PyTorch, and torch_npu and install this project. From the repository root, enter the test directory:

```bash
cd tests
```

Run a Python test file, replacing the filename for other cases:

```bash
python test_spherical_harmonics.py
```

The test in `host` uses pytest and can be run separately:

```bash
python -m pytest host/test_nested_image_downsampling.py
```

Build and run all C++ operator tests:

```bash
bash kernel_tests/build_kernel_tests.sh && bash kernel_tests/run_kernel_tests.sh
```

To see CANN DEBUG logs in the terminal, run this before the tests:

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

These variables apply to the current shell and its child processes. C++ build logs and run results are in `kernel_tests/build_out/`.
