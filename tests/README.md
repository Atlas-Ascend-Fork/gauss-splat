# 测试执行说明

在已配置 CANN、PyTorch、torch_npu 并安装本项目的 Linux/NPU 环境中，从仓库根目录进入测试目录：

```bash
cd tests
```

运行 Python 用例，例如（其他用例替换文件名即可）：

```bash
python test_spherical_harmonics.py
```

`host` 目录中的用例使用 pytest，可单独运行：

```bash
python -m pytest host/test_nested_image_downsampling.py
```

构建并运行全部 C++ 算子用例：

```bash
bash kernel_tests/build_kernel_tests.sh && bash kernel_tests/run_kernel_tests.sh
```

需要在终端查看 CANN DEBUG 日志时，在运行用例前执行：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

这两个变量只对当前 shell 及其启动的进程生效。C++ 测试的构建日志和运行结果位于 `kernel_tests/build_out/`。
