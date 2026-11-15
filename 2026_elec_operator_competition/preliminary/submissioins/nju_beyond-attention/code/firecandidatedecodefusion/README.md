# FireCandidateDecodeFusion

```text
.
├── CMakeLists.txt                         # 工程构建配置
├── op_host/
│   ├── CMakeLists.txt                     # Host 构建配置
│   └── fire_candidate_decode_fusion.cpp   # 算子注册与 Tiling
└── op_kernel/
    ├── CMakeLists.txt                     # Kernel 构建配置
    ├── fire_candidate_decode_fusion.cpp   # 算子入口
    ├── fire_candidate_decode_fusion_tiling.h
    ├── tiling_key_fire_candidate_decode_fusion.h
    └── fire_*.h                          # 存储、计算与分派实现
```
