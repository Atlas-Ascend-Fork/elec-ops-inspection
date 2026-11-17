# FusedDisconnectorKeypointSoftArgmax

隔离开关关键点坐标和热图分数融合解码算子，包含 Host 注册、Tiling、Ascend C Kernel 和构建配置。

```text
.
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   └── fused_disconnector_keypoint_soft_argmax.cpp
└── op_kernel/
    ├── CMakeLists.txt
    ├── fused_disconnector_keypoint_soft_argmax.cpp
    ├── fused_disconnector_keypoint_soft_argmax_tiling.h
    └── tiling_key_fused_disconnector_keypoint_soft_argmax.h
```
