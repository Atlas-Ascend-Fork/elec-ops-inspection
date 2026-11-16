# swiglu

提交团队: 马斯克盘
提交者: TACONailoong


## 算子实现介绍

fp16 的 11 bit 尾数相乘只有 22 bit，乘积在 fp32 中可以精确表示。因此：

把 x、W 先 Cast 到 fp32，再在 fp32 中做乘法，乘积无舍入误差；
点积在 fp32 中分 K 块累加，误差量级约 2^-24 * sqrt(d)（d = 8192 时仍 ≪ 1e-3）；
swish 的 exp / div 全部在 fp32 中完成，最后按 CAST_RINT（就近舍入）转回 fp16。
反例（已排除）：若在 fp16 中做乘积，随机向量点积存在相消 （|dot| ~ sqrt(d)*sigma 而 sum|terms| ~ 0.8*d*sigma），每项 ~5e-4 的相对舍入会放大到 d = 8192 时约 3.6e-2 的相对误差，必然超过 1e-3 容差。
