# solve

提交团队: aodebiao 的队伍
提交者: aodebiao


## 算子实现介绍

线性修正方程求解算子 solve

## 奇异矩阵与 CI 回归验证

LU 分解或回代发现零主元（包括负零）时，kernel 中止执行，不执行除零，也不将主元替换成任意小常数后返回错误解。非零的小主元保持原求解路径。

无需 NPU 的宿主机回归测试会将实际 `LuKernel` 编译到测试程序中，仅替换设备搬运、同步和异常接口；覆盖正常求解、行主元交换、多右端项、非零小主元以及分解/回代的零主元。

```bash
# 在本算子目录执行；需要支持 C++17 的宿主机编译器。
SOLVE_TEST_CXX="c++" python3 -m unittest discover -s tests/host -p test_singular_pivots.py -v
```

宿主机测试不验证 NPU 同步时序、DMA 或设备编译兼容性，设备编译与精度测试仍需 CANN/NPU 环境。
