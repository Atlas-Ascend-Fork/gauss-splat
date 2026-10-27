/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// flash_gaussian_build_mask_package/op_kernel/arch35/flash_gaussian_build_mask_kernel.h
// =============================================================================
//
// ROLE: normal 主模板（tilingKey=0）设备侧 kernel，FlashGaussianBuildMask on
//   arch35（Ascend 950 / A5，DAV_3510，AIV-only，regbase 纯 Vector）。
//   实现 docs/flash_gaussian_build_mask/design/Kernel.md §9 与
//   docs/flash_gaussian_build_mask/design/branches/DESIGN-BRANCH-0.md §3–§5：
//   三级过滤（Y 预过滤 → 包围盒粗过滤 → 中心/椭圆精过滤）+ 两级 GatherMask
//   有序压缩 + 末级输出压缩，扁平 (b,c)×tile 任务空间。
//
// 3510 落地差异与运行期实证（相对 A2/A3 参考实现 op_kernel/flash_gaussian_build_mask.cpp，
// 数学链与 DESIGN-BRANCH-0.md §5.3 逐步一致，仅承载机制按 3510 运行期实证调整）：
//   1. 扁平任务空间 / tile 坐标逐任务 GM 标量读 / tile_sum 2048 分块写回 /
//      FetchEventID 事件 / 中心闭区间 GE / 尾批 Ceil8（同文件首版说明，均保留）；
//   2. GM↔UB 搬运统一 DataCopyPad（1B 粒度）+ 自然边界裁剪：3510 整张量 DataCopy
//      要求 GM 侧 32B 对齐且 Ceil8 尾批越过张量末端触发 GM overflow 检查（MTE 507035）；
//   3. 谓词与压缩链以 asc_vf_call 启动的 __simd_vf__ 函数承载（本文件
//      YFilterMaskVF/BoundMaskVF/FineMaskVF/CompressVF/ArangeVF/RadiusHeadVF）：
//      3510 SDK 的整张量 fp32→uint8 比较族（CompareScalar/Compare/And/Or 计数形）
//      与 tensor-pattern GatherMask 入口在本工程实测触发向量核异常（vec error、
//      mte error=0，分级二分定位；A2/dav_c220 的整张量实现无此问题），生产 3510
//      内核（nms_with_mask arch35 等）亦仅使用 Reg 级谓词 + 常量 pattern GatherMask。
//      本实现的谓词数学（含 §5.3「A5 边界校正」GE 闭区间）与设计逐式一致，
//      符合 Kernel.md §9.5「统一 asc_vf_call 启动」约束；
//   4. 计数参数化 VF 使尾批 pad 处理内联化：谓词/压缩 VF 以 validLen 为 count，
//      UpdateMask 掩码天然排除 [validLen, curGaussNum) pad 区——A2 的
//      Duplicate+SetValue+CompareScalar pad 门控不再需要（掩码语义等价）；
//   5. 输出尾部不做逐 tile 全段 0 填充：参考实现（op_kernel/flash_gaussian_build_mask.cpp
//      FlagFilter）仅写出 Ceil8 有效前缀，tile 行 [tileSum, N) 尾段为未定义值；消费侧
//      （gaussian_sort）按 tileSum 读取前缀，故逐 tile 补写 [segBase+tileSum, segBase+N)
//      既非必要、又会把 O(tile 数 × N) 的 MTE3 写回流量引到 HBM（实测 MTE3 占比 0.69），
//      与参考实现保持一致不再补写。
//
// UB 内存规划（§4，TBuf 单缓冲 DB=1、无 TQue）：19 路 fp32 lane（input 11 +
// computing 6 + mask 1 + gatherIndex 1）+ tile_sum 计数 lane（2048×int32），
// baseBufferSize = gaussNumPerLoop × 4 B（≤ 设计 81 B/高斯预算）。
// =============================================================================

#pragma once

#include "kernel_operator.h"                          // Ascend C kernel framework (AscendC:: namespace)
#include "flash_gaussian_build_mask_tiling_struct.h"  // FlashGaussianBuildMaskTilingData / GATHER_NUM_SLOTS

using namespace AscendC;

// ---------------------------------------------------------------------------
// 常量（DESIGN-BRANCH-0.md §4 / §5；数值口径与参考实现一致）
// ---------------------------------------------------------------------------
constexpr uint32_t INPUT_BUFFER_NUM = 11;     // fp32 输入 lane 数（§4 划分表）
constexpr uint32_t COMPUTE_BUFFER_NUM = 6;    // fp32 计算 lane 数
constexpr uint32_t TILE_SUM_BUF_ELEMS = 2048; // tile_sum 计数 lane 固定容量（8 KB ≤ 31 KB 余量，§5.4）

// 输入 lane 下标（批内基址 = laneIdx × curGaussNum；槽位复用见 §5.3「槽内派生量覆盖复用」）
constexpr uint32_t IL_MX = 0;      // means2d x → 全程存活
constexpr uint32_t IL_MY = 1;      // means2d y → 全程存活
constexpr uint32_t IL_OP = 2;      // opacity → w2 折叠覆盖（跨级存活，第三级椭圆 c 项消费）
constexpr uint32_t IL_C00 = 3;     // cov00 → w → c00（槽位分相复用：先 covars2d 对角元，GetRect 后被 conics 覆盖）
constexpr uint32_t IL_C01 = 4;     // cov11 → h → c01（同上）
constexpr uint32_t IL_C11 = 5;     // conics c11
constexpr uint32_t IL_RMIN_W = 6;  // rect rmin_w（GetRect 产出）
constexpr uint32_t IL_RMIN_H = 7;  // rect rmin_h
constexpr uint32_t IL_RMAX_W = 8;  // rect rmax_w
constexpr uint32_t IL_RMAX_H = 9;  // rect rmax_h
constexpr uint32_t IL_DEPTHS = 10; // depths（直通压缩）

// 计算 lane 下标（GetRect：VF 开方输出/矩形钳制暂存；压缩集暂存；深度压缩 lane）
constexpr uint32_t CL_T0 = 0;     // 索引压缩暂存（GaussIndexGather 中转）
constexpr uint32_t CL_T1 = 1;
constexpr uint32_t CL_SQX = 2;    // √(cov00·w2)（RadiusHeadVF x 轴输出）
constexpr uint32_t CL_SQY = 3;    // √(cov11·w2)（RadiusHeadVF y 轴输出）
constexpr uint32_t CL_W2TMP = 4;  // w2 暂存（RadiusHeadVF 折叠输出中转）
constexpr uint32_t CL_DEPTH = 4;  // 深度压缩 lane（与 CL_W2TMP 分时复用：gather 阶段 w2 已消费）

// 压缩写出的 Computing lane 下标（级内压缩集暂存，参考实现同款分时复用）
constexpr uint32_t GL_M0 = 0;
constexpr uint32_t GL_M1 = 1;
constexpr uint32_t GL_OP = 2;
constexpr uint32_t GL_C00 = 3;
constexpr uint32_t GL_C01 = 4;
constexpr uint32_t GL_C11 = 5;
constexpr uint32_t GL_DEPTH = 4;  // 与 GL_C01 分时复用（depth 压缩在 conics CopyOut 之后）

// gather_mask GM 中转区槽位（20 槽：0–11 = Y 过滤后 12 路，12–19 = bound 过滤后 8 路）
constexpr uint32_t SLOT_MX = 0;
constexpr uint32_t SLOT_MY = 1;
constexpr uint32_t SLOT_OP = 2;
constexpr uint32_t SLOT_C00 = 3;
constexpr uint32_t SLOT_C01 = 4;
constexpr uint32_t SLOT_C11 = 5;
constexpr uint32_t SLOT_RMIN_W = 6;
constexpr uint32_t SLOT_RMIN_H = 7;
constexpr uint32_t SLOT_RMAX_W = 8;
constexpr uint32_t SLOT_RMAX_H = 9;
constexpr uint32_t SLOT_DEPTHS = 10;
constexpr uint32_t SLOT_GIDX = 11;
constexpr uint32_t SLOT2_MX = 12;
constexpr uint32_t SLOT2_MY = 13;
constexpr uint32_t SLOT2_OP = 14;
constexpr uint32_t SLOT2_C00 = 15;
constexpr uint32_t SLOT2_C01 = 16;
constexpr uint32_t SLOT2_C11 = 17;
constexpr uint32_t SLOT2_DEPTHS = 18;
constexpr uint32_t SLOT2_GIDX = 19;

constexpr int32_t FLOAT_SIZE = 4;  // fp32 字节

constexpr float ZERO_FLOAT_VALUE = 0.0f;
constexpr float ONE_FLOAT_VALUE = 1.0f;
constexpr float TWO_FLOAT_VALUE = 2.0f;
constexpr float LN2 = 0.69314718055f;  // 与 golden / ST oracle 同一常量（Validation.md 步骤 1）
constexpr float LN2_COEFF = 8.0f;      // power = ln2·(8 + log2(op))

#define Ceil8(num) (static_cast<uint32_t>(((num) + 7) & ~uint64_t(7)))

// ---------------------------------------------------------------------------
// FlashGaussianBuildMaskRadiusHeadVF — S1→S2 头段 VF 链（API.md §1 VF 链 1）
//
//   链长: Log2 → Muls → Adds → Muls → Mul → Sqrt = 6 ≤ 7；非 Cast；
//   x/y 轴两次独立 asc_vf_call（cov00Src / cov11Src 对称换参）。
//   输出: w2 = 2·(LN2·log2(op) + 8·LN2) 折叠 lane（跨段存活，S6/S7 椭圆 c 项消费）与
//         √(cov·w2) lane；⌊·⌋+1 尾步（Floor 无 VF-safe 形，API.md §1 断点注记）在
//         VF 外以整张量 API 收尾。
//   UpdateMask 引用传递：count 内部自动递减 VL_T（float:64），调用后不可手动递减。
// ---------------------------------------------------------------------------
template <typename T>
__simd_vf__ inline void FlashGaussianBuildMaskRadiusHeadVF(
    __ubuf__ T* w2Dst,     // 输出: 2·power 折叠 lane
    __ubuf__ T* sqDst,     // 输出: √(cov·w2) lane
    __ubuf__ T* opSrc,     // 输入: opacity lane
    __ubuf__ T* covSrc,    // 输入: covars2d 对角元 lane（x 轴 cov00 / y 轴 cov11）
    uint32_t count,        // 元素数 = curGaussNum（8 对齐 lane 长）
    uint16_t repeatTimes)  // 寄存器迭代数 = CeilDiv(count, VL_T)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    AscendC::Reg::RegTensor<T> vOp, vCov, vT, vW2, vSq;
    AscendC::Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(vOp, opSrc + i * repeatElm);
        AscendC::Reg::LoadAlign(vCov, covSrc + i * repeatElm);
        AscendC::Reg::Log2(vT, vOp, mask);                    // 1. log2(op)
        AscendC::Reg::Muls(vT, vT, LN2, mask);                // 2. LN2·log2(op)
        AscendC::Reg::Adds(vT, vT, LN2 * LN2_COEFF, mask);    // 3. + 8·LN2 → power
        AscendC::Reg::Muls(vW2, vT, TWO_FLOAT_VALUE, mask);   // 4. ×2 → w2（折叠）
        AscendC::Reg::Mul(vT, vCov, vW2, mask);               // 5. cov·w2
        AscendC::Reg::Sqrt(vSq, vT, mask);                    // 6. √(cov·w2)
        AscendC::Reg::StoreAlign(w2Dst + i * repeatElm, vW2, mask);
        AscendC::Reg::StoreAlign(sqDst + i * repeatElm, vSq, mask);
    }
}

// ---------------------------------------------------------------------------
// FlashGaussianBuildMaskArangeVF — 原始索引向量生成（CreateVecIndex 等价 VF 落地）
//   dst[i] = firstValue + i（fp32 整数编码，N ≤ 2^24−1 精确无损）。
//   与 SDK CreateVecIndexCalcVci2 同逻辑（Arange → 分寄存器 UpdateMask/StoreAlign/
//   Adds），经 asc_vf_call 启动（3510 的 Level-2 CreateVecIndex 直调 __simd_vf__
//   内部函数在本工程实测触发向量核异常）。
// ---------------------------------------------------------------------------
template <typename T>
__simd_vf__ inline void FlashGaussianBuildMaskArangeVF(__ubuf__ T* dst, int32_t firstValue, uint32_t count,
                                                        uint16_t repeatTimes)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    AscendC::Reg::RegTensor<T> vreg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::Arange(vreg, static_cast<T>(firstValue));
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::StoreAlign(dst + i * repeatElm, vreg, mask);
        AscendC::Reg::Adds(vreg, vreg, static_cast<T>(repeatElm), mask);
    }
}

// ---------------------------------------------------------------------------
// FlashGaussianBuildMaskYFilterMaskVF — 第一级 Y 预过滤谓词（§5.3 FilterByGridY）
//   mask = (rmin_h < maxTy) ∧ (rmax_h > minTy)（开区间，相切不保留）→ 0/1 fp32 物化
//   （Kernel.md §9.1 ①：谓词真→1.0 命中、假→0.0 排除）
// ---------------------------------------------------------------------------
template <typename T>
__simd_vf__ inline void FlashGaussianBuildMaskYFilterMaskVF(__ubuf__ T* maskDst, __ubuf__ T* rminHSrc,
                                                            __ubuf__ T* rmaxHSrc, T minTy, T maxTy, uint32_t count,
                                                            uint16_t repeatTimes)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    AscendC::Reg::RegTensor<T> vRminH, vRmaxH, vOut, vOnes, vZeros;
    AscendC::Reg::MaskReg preg, cmpLo, cmpHi, pred;
    AscendC::Reg::MaskReg pregAll = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(vOnes, ONE_FLOAT_VALUE, pregAll);
    AscendC::Reg::Duplicate(vZeros, ZERO_FLOAT_VALUE, pregAll);
    uint32_t sreg = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        preg = AscendC::Reg::UpdateMask<T>(sreg);
        AscendC::Reg::LoadAlign(vRminH, rminHSrc + i * repeatElm);
        AscendC::Reg::LoadAlign(vRmaxH, rmaxHSrc + i * repeatElm);
        AscendC::Reg::CompareScalar<T, CMPMODE::LT>(cmpLo, vRminH, maxTy, preg);  // rmin_h < maxTy
        AscendC::Reg::CompareScalar<T, CMPMODE::GT>(cmpHi, vRmaxH, minTy, preg);  // rmax_h > minTy
        AscendC::Reg::MaskAnd(pred, cmpLo, cmpHi, preg);
        AscendC::Reg::Select<T>(vOut, vOnes, vZeros, pred);
        AscendC::Reg::StoreAlign(maskDst + i * repeatElm, vOut, preg);
    }
}

// ---------------------------------------------------------------------------
// FlashGaussianBuildMaskBoundMaskVF — 第二级包围盒粗过滤谓词（§5.3 S4 ← 公式 2)）
//   all_in = (min(rmax_w, tx+ts) > max(rmin_w, tx)) ∧ (min(rmax_h, ty+ts) > max(rmin_h, ty))
//   → 0/1 fp32 物化
// ---------------------------------------------------------------------------
template <typename T>
__simd_vf__ inline void FlashGaussianBuildMaskBoundMaskVF(__ubuf__ T* maskDst, __ubuf__ T* rminWSrc,
                                                          __ubuf__ T* rminHSrc, __ubuf__ T* rmaxWSrc,
                                                          __ubuf__ T* rmaxHSrc, T tx, T ty, T ts, uint32_t count,
                                                          uint16_t repeatTimes)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    AscendC::Reg::RegTensor<T> vRminW, vRminH, vRmaxW, vRmaxH;
    AscendC::Reg::RegTensor<T> vS0, vS1, vS2, vS3, vOut, vOnes, vZeros, vTx, vTy, vTxTs, vTyTs;
    AscendC::Reg::MaskReg preg, cGt, cLt, selW, selH, cmpX, cmpY, allIn;
    AscendC::Reg::MaskReg pregAll = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(vOnes, ONE_FLOAT_VALUE, pregAll);
    AscendC::Reg::Duplicate(vZeros, ZERO_FLOAT_VALUE, pregAll);
    AscendC::Reg::Duplicate(vTx, tx, pregAll);
    AscendC::Reg::Duplicate(vTy, ty, pregAll);
    AscendC::Reg::Duplicate(vTxTs, tx + ts, pregAll);
    AscendC::Reg::Duplicate(vTyTs, ty + ts, pregAll);
    uint32_t sreg = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        preg = AscendC::Reg::UpdateMask<T>(sreg);
        AscendC::Reg::LoadAlign(vRminW, rminWSrc + i * repeatElm);
        AscendC::Reg::LoadAlign(vRminH, rminHSrc + i * repeatElm);
        AscendC::Reg::LoadAlign(vRmaxW, rmaxWSrc + i * repeatElm);
        AscendC::Reg::LoadAlign(vRmaxH, rmaxHSrc + i * repeatElm);
        // s0 = max(rmin_w, tx)；s1 = min(rmax_w, tx+ts)；s2/s3 y 轴同构
        AscendC::Reg::CompareScalar<T, CMPMODE::GT>(cGt, vRminW, tx, preg);
        AscendC::Reg::Select<T>(vS0, vRminW, vTx, cGt);
        AscendC::Reg::CompareScalar<T, CMPMODE::LT>(cLt, vRmaxW, tx + ts, preg);
        AscendC::Reg::Select<T>(vS1, vRmaxW, vTxTs, cLt);
        AscendC::Reg::CompareScalar<T, CMPMODE::GT>(cGt, vRminH, ty, preg);
        AscendC::Reg::Select<T>(vS2, vRminH, vTy, cGt);
        AscendC::Reg::CompareScalar<T, CMPMODE::LT>(cLt, vRmaxH, ty + ts, preg);
        AscendC::Reg::Select<T>(vS3, vRmaxH, vTyTs, cLt);
        // all_in = (s1 > s0) ∧ (s3 > s2)（两轴开区间重叠）
        AscendC::Reg::Compare<T, CMPMODE::GT>(cmpX, vS1, vS0, preg);
        AscendC::Reg::Compare<T, CMPMODE::GT>(cmpY, vS3, vS2, preg);
        AscendC::Reg::MaskAnd(allIn, cmpX, cmpY, preg);
        AscendC::Reg::Select<T>(vOut, vOnes, vZeros, allIn);
        AscendC::Reg::StoreAlign(maskDst + i * repeatElm, vOut, preg);
    }
}

// ---------------------------------------------------------------------------
// FlashGaussianBuildMaskFineMaskVF — 第三级精过滤谓词（§5.3 S5–S8 ← 公式 3)/4)）
//   center = (tx ≤ mx ≤ tx+ts−1) ∧ (ty ≤ my ≤ ty+ts−1)（闭区间，「A5 边界校正」GE）
//   isect  = flagX ∨ flagY（椭圆判别式 Δ/t₁/t₂ 符号判定，x 竖边 a=c11 / y 横边 a=c00）
//   mask   = center ∨ isect（all_in 已在第二级结构性施加）→ 0/1 fp32 物化
// ---------------------------------------------------------------------------
template <typename T>
__simd_vf__ inline void FlashGaussianBuildMaskFineMaskVF(__ubuf__ T* maskDst, __ubuf__ T* mxSrc, __ubuf__ T* mySrc,
                                                         __ubuf__ T* w2Src, __ubuf__ T* c00Src, __ubuf__ T* c01Src,
                                                         __ubuf__ T* c11Src, T tx, T ty, T ts, uint32_t count,
                                                         uint16_t repeatTimes)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    AscendC::Reg::RegTensor<T> vMx, vMy, vW2, vC00, vC01, vC11;
    AscendC::Reg::RegTensor<T> vDx, vNear, vFar, vB, vC, vDelta, vT1, vT2, vSq, vOut, vOnes, vZeros, vTmp;
    AscendC::Reg::MaskReg preg, cgeTx, cleTxHi, cgeTy, cleTyHi, center;
    AscendC::Reg::MaskReg farX, cDelta, cT1Le, cT1Sq, cT2Ge, cT2Sq, orT1, orT2, flagX;
    AscendC::Reg::MaskReg farY, flagY, isect, finalM;
    AscendC::Reg::MaskReg pregAll = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(vOnes, ONE_FLOAT_VALUE, pregAll);
    AscendC::Reg::Duplicate(vZeros, ZERO_FLOAT_VALUE, pregAll);
    const T txHi = tx + ts - ONE_FLOAT_VALUE;
    const T tyHi = ty + ts - ONE_FLOAT_VALUE;
    // 远边选择阈值：golden「mxc·2 < tx + (tx + tile_size − 1.0)」——括号内先加后减，
    // 外层再加 tx；等价半阈值 0.5·s（÷2 精确，2·mx 溢出拐角与 golden 同型）。
    const T halfSpanX = float(0.5f) * (tx + (tx + ts - ONE_FLOAT_VALUE));
    const T halfSpanY = float(0.5f) * (ty + (ty + ts - ONE_FLOAT_VALUE));
    uint32_t sreg = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        preg = AscendC::Reg::UpdateMask<T>(sreg);
        AscendC::Reg::LoadAlign(vMx, mxSrc + i * repeatElm);
        AscendC::Reg::LoadAlign(vMy, mySrc + i * repeatElm);
        AscendC::Reg::LoadAlign(vW2, w2Src + i * repeatElm);
        AscendC::Reg::LoadAlign(vC00, c00Src + i * repeatElm);
        AscendC::Reg::LoadAlign(vC01, c01Src + i * repeatElm);
        AscendC::Reg::LoadAlign(vC11, c11Src + i * repeatElm);

        // ── S5 中心包含（闭区间，A5 边界校正：下界 GE）──
        AscendC::Reg::CompareScalar<T, CMPMODE::GE>(cgeTx, vMx, tx, preg);
        AscendC::Reg::CompareScalar<T, CMPMODE::LE>(cleTxHi, vMx, txHi, preg);
        AscendC::Reg::CompareScalar<T, CMPMODE::GE>(cgeTy, vMy, ty, preg);
        AscendC::Reg::CompareScalar<T, CMPMODE::LE>(cleTyHi, vMy, tyHi, preg);
        AscendC::Reg::MaskAnd(center, cgeTx, cleTxHi, preg);
        AscendC::Reg::MaskAnd(cgeTy, cgeTy, cleTyHi, preg);  // 复用 cgeTy 作 y 轴合成
        AscendC::Reg::MaskAnd(center, center, cgeTy, preg);

        // ── S6 x 方向竖边（a = c11）──
        AscendC::Reg::CompareScalar<T, CMPMODE::LT>(farX, vMx, halfSpanX, preg);  // 2·mx < 2·tx+ts−1
        AscendC::Reg::Adds(vNear, vMx, float(-1.0f) * tx, preg);                  // mx − tx
        AscendC::Reg::Adds(vFar, vMx, float(-1.0f) * txHi, preg);                 // mx − (tx+ts−1)
        AscendC::Reg::Select<T>(vDx, vNear, vFar, farX);                          // 取远离中心的竖边
        AscendC::Reg::Muls(vB, vC01, TWO_FLOAT_VALUE * float(-1.0f), preg);
        AscendC::Reg::Mul(vB, vB, vDx, preg);                                     // b = −2·c01·dx
        AscendC::Reg::Mul(vC, vC00, vDx, preg);
        AscendC::Reg::Mul(vC, vC, vDx, preg);
        AscendC::Reg::Sub(vC, vC, vW2, preg);                                     // c = c00·dx² − w2
        AscendC::Reg::Mul(vDelta, vB, vB, preg);                                  // b²
        // Δ = b² − (4·c11)·c —— 标量因子先乘（golden「4.0*c11」结合序）：c11 > FLT_MAX/4
        // 时 4·c11 → ±inf、(4·c11)·c → ∓inf、Δ = ±inf/NaN；后乘结合 (c11·c)·4 全程有限，
        // 极值 conics（±FLT_MAX 级）下 t²≤Δ 谓词翻转 → 必须与 golden 结合序逐位对齐
        //（×2/×4 为 2 的幂，无溢出时两种结合逐位一致，常规数据行为不变）。
        AscendC::Reg::Muls(vSq, vC11, TWO_FLOAT_VALUE * TWO_FLOAT_VALUE, preg);   // 4·c11
        AscendC::Reg::Mul(vSq, vSq, vC, preg);                                    // (4·c11)·c
        AscendC::Reg::Sub(vDelta, vDelta, vSq, preg);                             // Δ = b² − (4·c11)·c
        // t1 = (ty−my)·(2·c11) + b、t2 = (ty+ts−1−my)·(2·c11) + b —— 同款结合序
        //（2·c11 先乘，c11 > FLT_MAX/2 → ±inf）；b − (my−ty′)·(2·c11) 与
        // (ty′−my)·(2·c11) + b 逐位一致（fl(a−b) = −fl(b−a)、b−(−x) = b+x 精确）。
        AscendC::Reg::Muls(vT1, vC11, TWO_FLOAT_VALUE, preg);                     // 2·c11
        AscendC::Reg::Adds(vNear, vMy, float(-1.0f) * ty, preg);                  // my − ty（vNear 已消费）
        AscendC::Reg::Mul(vT1, vNear, vT1, preg);                                 // (my−ty)·(2·c11)
        AscendC::Reg::Sub(vT1, vB, vT1, preg);                                    // t1
        AscendC::Reg::Muls(vT2, vC11, TWO_FLOAT_VALUE, preg);                     // 2·c11
        AscendC::Reg::Adds(vNear, vMy, float(-1.0f) * tyHi, preg);                // my − tyHi
        AscendC::Reg::Mul(vT2, vNear, vT2, preg);                                 // (my−tyHi)·(2·c11)
        AscendC::Reg::Sub(vT2, vB, vT2, preg);                                    // t2
        AscendC::Reg::CompareScalar<T, CMPMODE::GE>(cDelta, vDelta, ZERO_FLOAT_VALUE, preg);  // Δ ≥ 0
        AscendC::Reg::CompareScalar<T, CMPMODE::LE>(cT1Le, vT1, ZERO_FLOAT_VALUE, preg);      // t1 ≤ 0
        AscendC::Reg::Mul(vSq, vT1, vT1, preg);
        AscendC::Reg::Compare<T, CMPMODE::LE>(cT1Sq, vSq, vDelta, preg);                     // t1² ≤ Δ
        AscendC::Reg::MaskOr(orT1, cT1Le, cT1Sq, preg);
        AscendC::Reg::CompareScalar<T, CMPMODE::GE>(cT2Ge, vT2, ZERO_FLOAT_VALUE, preg);     // t2 ≥ 0
        AscendC::Reg::Mul(vSq, vT2, vT2, preg);
        AscendC::Reg::Compare<T, CMPMODE::LE>(cT2Sq, vSq, vDelta, preg);                     // t2² ≤ Δ
        AscendC::Reg::MaskOr(orT2, cT2Ge, cT2Sq, preg);
        AscendC::Reg::MaskAnd(flagX, cDelta, orT1, preg);
        AscendC::Reg::MaskAnd(flagX, flagX, orT2, preg);

        // ── S7 y 方向横边（a = c00，dy 取远横边，c = c11·dy² − w2，t1/t2 用 x 边界）──
        AscendC::Reg::CompareScalar<T, CMPMODE::LT>(farY, vMy, halfSpanY, preg);
        AscendC::Reg::Adds(vNear, vMy, float(-1.0f) * ty, preg);
        AscendC::Reg::Adds(vFar, vMy, float(-1.0f) * tyHi, preg);
        AscendC::Reg::Select<T>(vDx, vNear, vFar, farY);                          // dy（复用 vDx）
        AscendC::Reg::Muls(vB, vC01, TWO_FLOAT_VALUE * float(-1.0f), preg);
        AscendC::Reg::Mul(vB, vB, vDx, preg);                                     // b = −2·c01·dy
        AscendC::Reg::Mul(vC, vC11, vDx, preg);
        AscendC::Reg::Mul(vC, vC, vDx, preg);
        AscendC::Reg::Sub(vC, vC, vW2, preg);                                     // c = c11·dy² − w2
        AscendC::Reg::Mul(vDelta, vB, vB, preg);                                  // b²
        AscendC::Reg::Muls(vSq, vC00, TWO_FLOAT_VALUE * TWO_FLOAT_VALUE, preg);   // 4·c00（先乘标量，同 S6）
        AscendC::Reg::Mul(vSq, vSq, vC, preg);                                    // (4·c00)·c
        AscendC::Reg::Sub(vDelta, vDelta, vSq, preg);                             // Δ = b² − (4·c00)·c
        AscendC::Reg::Muls(vT1, vC00, TWO_FLOAT_VALUE, preg);                     // 2·c00（先乘标量，同 S6）
        AscendC::Reg::Adds(vNear, vMx, float(-1.0f) * tx, preg);                  // mx − tx（vNear 已消费）
        AscendC::Reg::Mul(vT1, vNear, vT1, preg);                                 // (mx−tx)·(2·c00)
        AscendC::Reg::Sub(vT1, vB, vT1, preg);                                    // t1 = (tx−mx)·(2·c00) + b
        AscendC::Reg::Muls(vT2, vC00, TWO_FLOAT_VALUE, preg);                     // 2·c00
        AscendC::Reg::Adds(vNear, vMx, float(-1.0f) * txHi, preg);                // mx − txHi
        AscendC::Reg::Mul(vT2, vNear, vT2, preg);                                 // (mx−txHi)·(2·c00)
        AscendC::Reg::Sub(vT2, vB, vT2, preg);                                    // t2 = (tx+ts−1−mx)·(2·c00) + b
        AscendC::Reg::CompareScalar<T, CMPMODE::GE>(cDelta, vDelta, ZERO_FLOAT_VALUE, preg);
        AscendC::Reg::CompareScalar<T, CMPMODE::LE>(cT1Le, vT1, ZERO_FLOAT_VALUE, preg);
        AscendC::Reg::Mul(vSq, vT1, vT1, preg);
        AscendC::Reg::Compare<T, CMPMODE::LE>(cT1Sq, vSq, vDelta, preg);
        AscendC::Reg::MaskOr(orT1, cT1Le, cT1Sq, preg);
        AscendC::Reg::CompareScalar<T, CMPMODE::GE>(cT2Ge, vT2, ZERO_FLOAT_VALUE, preg);
        AscendC::Reg::Mul(vSq, vT2, vT2, preg);
        AscendC::Reg::Compare<T, CMPMODE::LE>(cT2Sq, vSq, vDelta, preg);
        AscendC::Reg::MaskOr(orT2, cT2Ge, cT2Sq, preg);
        AscendC::Reg::MaskAnd(flagY, cDelta, orT1, preg);
        AscendC::Reg::MaskAnd(flagY, flagY, orT2, preg);

        // ── S8 mask = center ∨ isect（all_in 第二级结构性施加；cnt 截断第一级循环上界）──
        AscendC::Reg::MaskOr(isect, flagX, flagY, preg);
        AscendC::Reg::MaskOr(finalM, center, isect, preg);
        AscendC::Reg::Select<T>(vOut, vOnes, vZeros, finalM);
        AscendC::Reg::StoreAlign(maskDst + i * repeatElm, vOut, preg);
    }
}

// ---------------------------------------------------------------------------
// FlashGaussianBuildMaskCompressVF — GatherMask 有序压缩的 VF 落地（Kernel.md §9.4）
//   按 0/1 掩码 lane（谓词物化产物）把 src 的命中元素按原序压缩写至 dst 前缀；
//   命中计数经 SPR.AR 累加（调用侧 GetSpr 读取，SDK GatherMask 同款）。
//   与 SDK GatherMaskReduceElePerVec 同机制（vsqz 压缩 + vstas 非对齐写出），
//   掩码源改为 0/1 fp32 lane（Reg::CompareScalar > 0 重建谓词，nms_with_mask
//   arch35 生产内核同款 Reg 级谓词模式）；count 以 validLen 为界，pad 区不进入
//   压缩/计数（UpdateMask 掩码天然排除，§9.1 ② 等价机制）。
// ---------------------------------------------------------------------------
template <typename T>
__simd_vf__ inline void FlashGaussianBuildMaskCompressVF(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* maskLane,
                                                         uint32_t count, uint16_t repeatTimes)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    AscendC::Reg::RegTensor<T> srcReg;
    AscendC::Reg::RegTensor<T> maskRegT;
    AscendC::Reg::RegTensor<T> dstReg;
    AscendC::Reg::MaskReg preg;
    AscendC::Reg::MaskReg predMask;
    AscendC::Reg::UnalignReg ureg;
    AscendC::Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();  // 命中计数累加器清零
    uint32_t sreg = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        preg = AscendC::Reg::UpdateMask<T>(sreg);
        AscendC::Reg::LoadAlign(srcReg, src + i * repeatElm);
        AscendC::Reg::LoadAlign(maskRegT, maskLane + i * repeatElm);
        AscendC::Reg::CompareScalar<T, CMPMODE::GT>(predMask, maskRegT, ZERO_FLOAT_VALUE, preg);  // 谓词重建
        AscendC::Reg::GatherMask<T, AscendC::Reg::GatherMaskMode::STORE_REG>(dstReg, srcReg, predMask);
        AscendC::Reg::StoreUnAlign(dst, dstReg, ureg);
    }
    AscendC::Reg::StoreUnAlignPost(dst, ureg);
}

// ===========================================================================
// class FlashGaussianBuildMaskKernel<T> — normal 主模板（tilingKey=0）
//
// 模板参数 T = DTYPE_MEANS2D（fp32 唯一组合；dtype 不进 TilingKey，编译期实例化）。
// cnt / tile_sum 为 int32 计数量，非 T。
// ===========================================================================
template <typename T>
class FlashGaussianBuildMaskKernel {
public:
    __aicore__ inline FlashGaussianBuildMaskKernel() {}

    // -----------------------------------------------------------------------
    // Init — PreInit（blockDim 校验 + 长短核任务段映射）→ GlobalBufferInit
    //        （8 输入 + 3 输出 GM 绑定）→ LocalBufferInit（§4 lane 表）→ 事件 ID 预取
    // -----------------------------------------------------------------------
    __aicore__ inline void Init(GM_ADDR means2d, GM_ADDR opacity, GM_ADDR conics, GM_ADDR covars2d,
                                GM_ADDR depths, GM_ADDR cnt, GM_ADDR tile_grid, GM_ADDR gathermask,
                                GM_ADDR tile_sum, GM_ADDR tile_depths, GM_ADDR gauss_index,
                                const FlashGaussianBuildMaskTilingData* tilingData, TPipe* pipe)
    {
        // ── PreInit：blockDim 非零校验（设备侧空实现防御；host SetBlockDim 恒 ≥ 1）+ 长短核任务段映射 ──
        ASSERT((tilingData->blockDim != 0) && "Block Dim can not be Zero!");
        td_ = tilingData;
        blockIndex = GetBlockIdx();
        if (blockIndex < td_->numLcore) {                       // 长核段
            taskNum_ = td_->taskNumPerLcore;
            taskStart_ = static_cast<uint32_t>(blockIndex) * td_->taskNumPerLcore;
        } else {                                                // 短核段
            taskNum_ = td_->taskNumPerScore;
            taskStart_ = td_->numLcore * td_->taskNumPerLcore
                       + (static_cast<uint32_t>(blockIndex) - td_->numLcore) * td_->taskNumPerScore;
        }
        gatherBlockStartAddr = blockIndex * static_cast<uint64_t>(td_->gaussNum) * GATHER_NUM_SLOTS;
        curBcIdx_ = UINT32_MAX;               // 促使首个任务触发视图切换（第一级 Y 预过滤）
        curTaskIdx_ = 0;
        curValidLen_ = 0;
        gatherStartIndex = 0;

        batchSize = td_->batchSize;
        cameraNum = td_->cameraNum;
        numTile = td_->numTile;
        gaussNum = td_->gaussNum;
        gaussNumPerLoop = td_->gaussNumPerLoop;
        imageWidth = td_->imageWidth;         // Float 属性 fp32 直传（§6）
        imageHeight = td_->imageHeight;
        tileSize = td_->tileSize;             // Int 属性转 fp32（§6）

        // ── GM 绑定（Interface.md「算子原型」顺序）──
        uint64_t baseGaussSize = static_cast<uint64_t>(batchSize) * cameraNum * gaussNum;  // B·C·N
        means2dGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(means2d), baseGaussSize * 2);
        opacityGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(opacity), baseGaussSize);
        conicsGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(conics), baseGaussSize * 3);
        covars2dGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(covars2d), baseGaussSize * 3);
        depthsGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(depths), baseGaussSize);
        cntGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cnt), static_cast<uint64_t>(batchSize) * cameraNum);
        tilegridGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(tile_grid), static_cast<uint64_t>(numTile) * 2);
        gathermaskGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gathermask));  // [BLOCK_NUM,20,N] 中转区
        gaussindexGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gauss_index), baseGaussSize * numTile);
        tiledepthsGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(tile_depths), baseGaussSize * numTile);
        tilesumGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tile_sum),
                                  static_cast<uint64_t>(batchSize) * cameraNum * numTile);
        // GM 自然边界（OOB 裁剪用：Ceil8 对齐尾批越过张量末端触发 GM overflow 检查）
        means2dEnd_ = baseGaussSize * 2;
        opacityEnd_ = baseGaussSize;
        conicsEnd_ = baseGaussSize * 3;
        covars2dEnd_ = baseGaussSize * 3;
        depthsEnd_ = baseGaussSize;

        // ── TBuf 分配（§4 划分表；baseBufferSize = gaussNumPerLoop × 4B）──
        pipe_ = pipe;
        uint64_t baseBufferSize = static_cast<uint64_t>(gaussNumPerLoop) * FLOAT_SIZE;
        pipe_->InitBuffer(InputTensorBuffer, static_cast<uint32_t>(baseBufferSize) * INPUT_BUFFER_NUM);
        pipe_->InitBuffer(MaskTensorBuffer, static_cast<uint32_t>(baseBufferSize));
        pipe_->InitBuffer(ComputingTensorBuffer, static_cast<uint32_t>(baseBufferSize) * COMPUTE_BUFFER_NUM);
        pipe_->InitBuffer(GatherIndexTensorBuffer, static_cast<uint32_t>(baseBufferSize));
        pipe_->InitBuffer(TileSumTensorBuffer, TILE_SUM_BUF_ELEMS * sizeof(int32_t));  // 固定容量计数 lane

        InputTensor = InputTensorBuffer.Get<T>();
        MaskTensor = MaskTensorBuffer.Get<T>();
        ComputingTensor = ComputingTensorBuffer.Get<T>();
        GatherIndexTensor = GatherIndexTensorBuffer.Get<T>();
        TileSumTensor = TileSumTensorBuffer.Get<int32_t>();
        Duplicate(GatherIndexTensor, ZERO_FLOAT_VALUE, gaussNumPerLoop);  // 索引 lane 预清零

        // ── 事件 ID 预取（A5 规范：FetchEventID 动态获取，禁止硬编码 EVENT_ID）──
        evMte2V_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        evVmte2_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        evVmte3_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        evMte3V_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    }

    // -----------------------------------------------------------------------
    // Process — 任务段循环（§5.3 编排总览）：
    //   段内逐任务 view-major 解码 → bcIdx 切换触发第一级 Y 预过滤（逐 (核,视图) 一次）
    //   → 逐 tile 二/三级过滤 + 输出压缩 → tile_sum 计数 lane 记录，lane 满 / 段末
    //   分块 DataCopyPad 写回（块基址 = 块首任务平铺索引 = tile_sum 平铺索引）。
    //   输出列表仅写 Ceil8 有效前缀，[tileSum, N) 尾段不写（与参考实现一致）。
    // -----------------------------------------------------------------------
    __aicore__ inline void Process()
    {
        uint32_t tileSumLaneCnt = 0;             // 当前 staging 块内已记录任务数
        uint32_t tileSumLaneBase = taskStart_;   // 块首任务平铺索引
        const uint32_t taskEnd = taskStart_ + taskNum_;
        for (uint32_t task = taskStart_; task < taskEnd; task++) {
            const uint32_t bcIdx = task / numTile;  // view-major 解码
            const uint32_t tIdx = task % numTile;
            if (bcIdx != curBcIdx_) {
                // 视图切换：cnt 标量读（g < cnt 截断的循环上界；越界 cnt 按 golden 语义
                // 钳到 [0, N]）+ 本核在该视图 tile 子段的 y 覆盖（段内全部 tile 扫描）
                int32_t cntVal = cntGM.GetValue(bcIdx);
                cntGaussNum = (cntVal > 0) ? static_cast<uint32_t>(cntVal) : 0u;
                if (cntGaussNum > gaussNum) {
                    cntGaussNum = gaussNum;
                }
                const uint32_t viewEnd = (taskEnd < (bcIdx + 1) * numTile) ? taskEnd : (bcIdx + 1) * numTile;
                const uint32_t firstTIdx = task - bcIdx * numTile;
                const uint32_t lastTIdx = viewEnd - 1 - bcIdx * numTile;
                // 段 y 覆盖 = [min(ty), max(ty + tileSize))——须扫描段内全部 tile 取真
                // 上下界：spec/golden 对 tile_grid 取值无有序性约束（黑盒 L1 级随机
                // tile_grid 合法），历史实现取首/末 tile 的 ty 仅在行优先有序网格下
                // 等价，无序输入下误剔除非边界高斯（keepY 非命中必要条件 → 精度失败）。
                // keepY 为性能削减（DESIGN-BRANCH-0 §5.3「不改变掩码语义」）：真上下界
                // 下 rect_h ∩ [ty_t, ty_t+ts) ≠ ∅ ⇒ rmin_h < maxTy ∧ rmax_h > minTy
                // 恒成立（必要条件保持，任意 tile_grid 取值安全）；NaN ty 比较恒 False
                // 被跳过，不污染界（NaN tile 由二级 all_in 按 golden 语义排除）。
                float minTy = tilegridGM.GetValue(firstTIdx * 2);        // [t,0] = ty
                float maxTy = minTy + tileSize;
                for (uint32_t t = firstTIdx + 1; t <= lastTIdx; ++t) {
                    const float tyVal = tilegridGM.GetValue(t * 2);
                    if (tyVal < minTy) {
                        minTy = tyVal;
                    }
                    const float tyHi = tyVal + tileSize;
                    if (tyHi > maxTy) {
                        maxTy = tyHi;
                    }
                }
                curBcIdx_ = bcIdx;  // 先行更新视图号（CopyInForBound 的 channel-first 基址消费点）
                AxisYFilter(minTy, maxTy);
            }
            // tile 坐标逐任务 GM 标量读（§5.2，2 次/tile；[t,0]=ty、[t,1]=tx）
            float tileGridY = tilegridGM.GetValue(tIdx * 2);
            float tileGridX = tilegridGM.GetValue(tIdx * 2 + 1);
            curTaskIdx_ = task - taskStart_;
            ComputingForSingleTile(tileGridX, tileGridY);

            // tile_sum[t] = 本 tile 三级命中计数（尾值读取后归零，供下一 tile 二级前缀从 0 累加）
            TileSumTensor.SetValue(tileSumLaneCnt, static_cast<int32_t>(gatherStartIndex));
            gatherStartIndex = 0;
            tileSumLaneCnt++;
            if (tileSumLaneCnt == TILE_SUM_BUF_ELEMS || task + 1 == taskEnd) {
                CopyOutTileSum(tileSumLaneBase, tileSumLaneCnt);  // lane 满 / 段末分块写回
                tileSumLaneBase = task + 1;
                tileSumLaneCnt = 0;
            }
        }
    }

private:
    // =======================================================================
    // 分批长度现算（§2 f5：批内 Ceil8 对齐）
    // =======================================================================
    __aicore__ inline void ComputingCurGaussNum(uint32_t totalGaussNum)
    {
        uint32_t alignedGaussNum = Ceil8(totalGaussNum);  // 8 元素 = 32B 对齐拷贝
        tailNum = alignedGaussNum - totalGaussNum;        // 尾批 pad 数（VF count 掩码排除）
        taskLoop = static_cast<int32_t>((totalGaussNum + gaussNumPerLoop - 1) / gaussNumPerLoop);
    }

    __aicore__ inline void GetComputeLength(uint32_t taskLoopIndex)
    {
        if (taskLoopIndex == static_cast<uint32_t>(taskLoop) - 1) {
            taskNumPerCurLoop = (cntGaussNum + tailNum) - taskLoopIndex * gaussNumPerLoop;
        } else {
            taskNumPerCurLoop = gaussNumPerLoop;
        }
        // 本批有效长度（VF count；[curValidLen_, curGaussNum) pad 区由 UpdateMask 掩码排除）
        uint32_t valid = cntGaussNum - taskLoopIndex * gaussNumPerLoop;
        curValidLen_ = (valid < taskNumPerCurLoop) ? valid : taskNumPerCurLoop;
    }

    // =======================================================================
    // 第一级 · Y 预过滤（逐 (核,视图) 一次，bcIdx 切换时由 Process 触发）
    // =======================================================================
    __aicore__ inline void AxisYFilter(float minTileGridY, float maxTileGridY)
    {
        ComputingCurGaussNum(cntGaussNum);  // cnt 截断：仅扫描 [0, cnt)
        for (uint32_t taskLoopIndex = 0; taskLoopIndex < static_cast<uint32_t>(taskLoop); taskLoopIndex++) {
            GetComputeLength(taskLoopIndex);
            CopyInForBound(taskLoopIndex);
                        GetRect(taskLoopIndex);
                        CopyInForFlashGS(taskLoopIndex);
            Copyout2yFilterWS(taskLoopIndex, minTileGridY, maxTileGridY);
        }
        yFilterGaussNum = static_cast<uint32_t>(gatherStartIndex);  // 紧凑集总长 → 第二级批界
        gatherStartIndex = 0;
    }

    __aicore__ inline void Copyout2yFilterWS(uint32_t taskLoopIndex, float minTileGridY, float maxTileGridY)
    {
        // Y 覆盖剔除谓词 → 0/1 掩码物化（开区间：rectmin_h < maxTy+ts ∧ rectmax_h > minTy）
        YFilterMask(minTileGridY, maxTileGridY, curValidLen_);
        
        uint64_t rsvdGSCnt = 0;
        uint64_t realGSCnt = GaussIndexGather(taskLoopIndex, 0);
                uint64_t copyLength = Ceil8(realGSCnt);
        // copyout: means2d, opacity
        Means2dGather(rsvdGSCnt);
        OpacityGather(rsvdGSCnt);

        SetFlag<HardEvent::MTE2_V>(evMte2V_);
        WaitFlag<HardEvent::MTE2_V>(evMte2V_);
        ConicsGather(rsvdGSCnt);

        SetFlag<HardEvent::V_MTE3>(evVmte3_);
        WaitFlag<HardEvent::V_MTE3>(evVmte3_);

        Means2dCopyOut(copyLength, 0);
        OpacityCopyOut(copyLength, 0);
        ConicsCopyOut(copyLength, 0);
        GaussIndexCopyOut(copyLength, 0);

        SetFlag<HardEvent::MTE3_V>(evMte3V_);
        WaitFlag<HardEvent::MTE3_V>(evMte3V_);

        // copyout: rect, depths
        RectGather(rsvdGSCnt);
        DepthGather(rsvdGSCnt);

        SetFlag<HardEvent::V_MTE3>(evVmte3_);
        WaitFlag<HardEvent::V_MTE3>(evVmte3_);

        RectCopyOut(copyLength);
        DepthCopyOut(copyLength, 0);

        gatherStartIndex += realGSCnt;
    }

    // =======================================================================
    // 逐 tile 二/三级过滤 + 输出压缩（§5.3 编排）
    // =======================================================================
    __aicore__ inline void ComputingForSingleTile(float tileGridX, float tileGridY)
    {
        // 第二级 · bound 批循环（批界 = Y 紧凑集总长 yFilterGaussNum）
        cntGaussNum = yFilterGaussNum;
                ComputingCurGaussNum(cntGaussNum);
        for (uint32_t taskLoopIndex = 0; taskLoopIndex < static_cast<uint32_t>(taskLoop); taskLoopIndex++) {
            GetComputeLength(taskLoopIndex);
            CopyInFromYFilter(taskLoopIndex);
            BoundFilter(taskLoopIndex, tileGridX, tileGridY);
        }
        // 第三级 · 精过滤批循环（批界 = bound 紧凑集总长）
        cntGaussNum = static_cast<uint32_t>(gatherStartIndex);
                ComputingCurGaussNum(cntGaussNum);
        gatherStartIndex = 0;
        for (uint32_t taskLoopIndex = 0; taskLoopIndex < static_cast<uint32_t>(taskLoop); taskLoopIndex++) {
            GetComputeLength(taskLoopIndex);
            CopyInFromBoundFilter(taskLoopIndex);
            FlagFilter(tileGridX, tileGridY, taskLoopIndex);
        }
        // 尾值 gatherStartIndex = 本 tile 命中总数，交 Process 记入 tile_sum 后归零
    }

    // =======================================================================
    // CopyIn —— 第一级：op 输入 GM → UB（1-D flat，channel-first 基址标量预计算 +
    // DataCopyPad 1B 粒度 + 张量末端裁剪；cov01 不进计算链，不搬入）
    // =======================================================================
    __aicore__ inline void CopyInForBound(uint32_t taskLoopIndex)
    {
        uint64_t viewBase = static_cast<uint64_t>(curBcIdx_) * gaussNum;  // 单视图基址
        uint64_t taskCopyinIndex = taskLoopIndex * gaussNumPerLoop;
        SetFlag<HardEvent::V_MTE2>(evVmte2_);
        WaitFlag<HardEvent::V_MTE2>(evVmte2_);

        CopyGm2UbPad(opacityGM, viewBase + taskCopyinIndex, opacityEnd_, InputTensor[taskNumPerCurLoop * IL_OP], taskNumPerCurLoop);
        CopyGm2UbPad(covars2dGM, viewBase * 3 + taskCopyinIndex, covars2dEnd_, InputTensor[taskNumPerCurLoop * IL_C00], taskNumPerCurLoop);
        CopyGm2UbPad(covars2dGM, viewBase * 3 + taskCopyinIndex + gaussNum * 2, covars2dEnd_, InputTensor[taskNumPerCurLoop * IL_C01], taskNumPerCurLoop);
        SetFlag<HardEvent::MTE2_V>(evMte2V_);
        WaitFlag<HardEvent::MTE2_V>(evMte2V_);
        SetFlag<HardEvent::V_MTE2>(evVmte2_);
        WaitFlag<HardEvent::V_MTE2>(evVmte2_);
        CopyGm2UbPad(means2dGM, viewBase * 2 + taskCopyinIndex, means2dEnd_, InputTensor[taskNumPerCurLoop * IL_MX], taskNumPerCurLoop);
        CopyGm2UbPad(means2dGM, viewBase * 2 + taskCopyinIndex + gaussNum, means2dEnd_, InputTensor[taskNumPerCurLoop * IL_MY], taskNumPerCurLoop);
    }

    __aicore__ inline void CopyInForFlashGS(uint32_t taskLoopIndex)
    {
        uint64_t viewBase = static_cast<uint64_t>(curBcIdx_) * gaussNum;
        uint64_t taskCopyinIndex = taskLoopIndex * gaussNumPerLoop;
        SetFlag<HardEvent::V_MTE2>(evVmte2_);
        WaitFlag<HardEvent::V_MTE2>(evVmte2_);

        CopyGm2UbPad(conicsGM, viewBase * 3 + taskCopyinIndex, conicsEnd_, InputTensor[taskNumPerCurLoop * IL_C00], taskNumPerCurLoop);
        CopyGm2UbPad(conicsGM, viewBase * 3 + taskCopyinIndex + gaussNum, conicsEnd_, InputTensor[taskNumPerCurLoop * IL_C01], taskNumPerCurLoop);
        CopyGm2UbPad(conicsGM, viewBase * 3 + taskCopyinIndex + gaussNum * 2, conicsEnd_, InputTensor[taskNumPerCurLoop * IL_C11], taskNumPerCurLoop);
        CopyGm2UbPad(depthsGM, viewBase + taskCopyinIndex, depthsEnd_, InputTensor[taskNumPerCurLoop * IL_DEPTHS], taskNumPerCurLoop);
    }

    // =======================================================================
    // 第一级 · GetRect（S1–S3 ← Overview.md 公式 1)）
    // =======================================================================
    __aicore__ inline void GetRect(uint32_t taskLoopIndex)
    {
        if (taskLoopIndex > 0) {
            SetFlag<HardEvent::MTE3_V>(evMte3V_);
            WaitFlag<HardEvent::MTE3_V>(evMte3V_);
        }

        // S1→S2 头段：RadiusHeadVF（asc_vf_call，API.md §1 VF 链 1）
        //   w2 = 2·(LN2·log2(op) + 8·LN2) → CL_W2TMP；√(cov·w2) → CL_SQX / CL_SQY。
        //   x/y 轴两次独立调用（cov00 / cov11 对称换参）；两次均读原始 opacity lane，
        //   w2 先落 CL_W2TMP，随后 Adds(+0) 折叠覆盖写回 op lane（跨级存活）。
        constexpr uint32_t vlF32 = GetVecLen() / sizeof(float);
        uint16_t repeatTimes = static_cast<uint16_t>((taskNumPerCurLoop + vlF32 - 1) / vlF32);
        asc_vf_call<FlashGaussianBuildMaskRadiusHeadVF<T>>(
            (__ubuf__ T*)ComputingTensor[taskNumPerCurLoop * CL_W2TMP].GetPhyAddr(),
            (__ubuf__ T*)ComputingTensor[taskNumPerCurLoop * CL_SQX].GetPhyAddr(),
            (__ubuf__ T*)InputTensor[taskNumPerCurLoop * IL_OP].GetPhyAddr(),
            (__ubuf__ T*)InputTensor[taskNumPerCurLoop * IL_C00].GetPhyAddr(),
            taskNumPerCurLoop, repeatTimes);
        asc_vf_call<FlashGaussianBuildMaskRadiusHeadVF<T>>(
            (__ubuf__ T*)ComputingTensor[taskNumPerCurLoop * CL_W2TMP].GetPhyAddr(),
            (__ubuf__ T*)ComputingTensor[taskNumPerCurLoop * CL_SQY].GetPhyAddr(),
            (__ubuf__ T*)InputTensor[taskNumPerCurLoop * IL_OP].GetPhyAddr(),
            (__ubuf__ T*)InputTensor[taskNumPerCurLoop * IL_C01].GetPhyAddr(),
            taskNumPerCurLoop, repeatTimes);

        // S2 尾步：w = ⌊√(cov00·w2)+1⌋、h = ⌊√(cov11·w2)+1⌋（Floor 无 VF-safe 形，
        // 整张量收尾，API.md §1 断点注记）；w/h 覆盖写回 covars2d 槽位
        Adds(ComputingTensor[taskNumPerCurLoop * CL_SQX], ComputingTensor[taskNumPerCurLoop * CL_SQX],
             ONE_FLOAT_VALUE, taskNumPerCurLoop);
        Adds(ComputingTensor[taskNumPerCurLoop * CL_SQY], ComputingTensor[taskNumPerCurLoop * CL_SQY],
             ONE_FLOAT_VALUE, taskNumPerCurLoop);
        Floor(InputTensor[taskNumPerCurLoop * IL_C00], ComputingTensor[taskNumPerCurLoop * CL_SQX],
              taskNumPerCurLoop);
        Floor(InputTensor[taskNumPerCurLoop * IL_C01], ComputingTensor[taskNumPerCurLoop * CL_SQY],
              taskNumPerCurLoop);

        // w2 折叠覆盖写回 op lane（跨级存活：第三级椭圆 c 项直接消费 c00·dx² − w2）
        Adds(InputTensor[taskNumPerCurLoop * IL_OP], ComputingTensor[taskNumPerCurLoop * CL_W2TMP],
             ZERO_FLOAT_VALUE, taskNumPerCurLoop);

        SetFlag<HardEvent::MTE2_V>(evMte2V_);
        WaitFlag<HardEvent::MTE2_V>(evMte2V_);

        // S3: rect = clip(m ∓ w/h, 0, W−1 / H−1) —— ClampMin(0) 先于 ClampMax(W−1)：
        // 契约域外输入（opacity ≤ 0 / 协方差对角元为负 → w/h = NaN）在该次序下
        // rmin == rmax 或 NaN，两轴开区间重叠判定恒 False → 掩码全 False（golden 语义）
        Sub(ComputingTensor[taskNumPerCurLoop * CL_T1], InputTensor[taskNumPerCurLoop * IL_MX],
            InputTensor[taskNumPerCurLoop * IL_C00], taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop * CL_T0], InputTensor[taskNumPerCurLoop * IL_MY],
            InputTensor[taskNumPerCurLoop * IL_C01], taskNumPerCurLoop);

        ClampMin(ComputingTensor[taskNumPerCurLoop * CL_SQX], ComputingTensor[taskNumPerCurLoop * CL_T1],
                 ZERO_FLOAT_VALUE, taskNumPerCurLoop);
        ClampMin(ComputingTensor[taskNumPerCurLoop * CL_SQY], ComputingTensor[taskNumPerCurLoop * CL_T0],
                 ZERO_FLOAT_VALUE, taskNumPerCurLoop);

        ClampMax(InputTensor[taskNumPerCurLoop * IL_RMIN_W], ComputingTensor[taskNumPerCurLoop * CL_SQX],
                 imageWidth - ONE_FLOAT_VALUE, taskNumPerCurLoop);
        ClampMax(InputTensor[taskNumPerCurLoop * IL_RMIN_H], ComputingTensor[taskNumPerCurLoop * CL_SQY],
                 imageHeight - ONE_FLOAT_VALUE, taskNumPerCurLoop);

        Add(InputTensor[taskNumPerCurLoop * IL_C00], InputTensor[taskNumPerCurLoop * IL_MX],
            InputTensor[taskNumPerCurLoop * IL_C00], taskNumPerCurLoop);
        Add(InputTensor[taskNumPerCurLoop * IL_C01], InputTensor[taskNumPerCurLoop * IL_MY],
            InputTensor[taskNumPerCurLoop * IL_C01], taskNumPerCurLoop);

        ClampMin(ComputingTensor[taskNumPerCurLoop * CL_SQX], InputTensor[taskNumPerCurLoop * IL_C00],
                 ZERO_FLOAT_VALUE, taskNumPerCurLoop);
        ClampMin(ComputingTensor[taskNumPerCurLoop * CL_SQY], InputTensor[taskNumPerCurLoop * IL_C01],
                 ZERO_FLOAT_VALUE, taskNumPerCurLoop);

        ClampMax(InputTensor[taskNumPerCurLoop * IL_RMAX_W], ComputingTensor[taskNumPerCurLoop * CL_SQX],
                 imageWidth - ONE_FLOAT_VALUE, taskNumPerCurLoop);
        ClampMax(InputTensor[taskNumPerCurLoop * IL_RMAX_H], ComputingTensor[taskNumPerCurLoop * CL_SQY],
                 imageHeight - ONE_FLOAT_VALUE, taskNumPerCurLoop);
    }

    // =======================================================================
    // CopyIn —— 第二级：gather_mask GM 本核行槽 0–11 → UB（12 路）
    // =======================================================================
    // Read adjacent pre-gathered channels in one MTE2 command. Source stride
    // is in bytes; UB blocks are rounded to 32B, matching the Ceil8 lane size.
    // Copy only valid elements so N=206613 and the last scratch row stay in bounds.
    __aicore__ inline void CopyGatherChannels(uint32_t loop, uint32_t slot,
                                             uint16_t channels, const LocalTensor<T>& dst)
    {
        const uint32_t offset = loop * gaussNumPerLoop;
        const uint32_t count = curValidLen_;
        DataCopyExtParams params{channels, count * static_cast<uint32_t>(sizeof(T)),
                                 (gaussNum - count) * static_cast<uint32_t>(sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(dst, gathermaskGM[gatherBlockStartAddr + static_cast<uint64_t>(slot) * gaussNum + offset],
                    params, pad);
    }

    __aicore__ inline void CopyInFromYFilter(uint32_t taskLoopIndex)
    {
        if (taskLoopIndex == 0) {
            SetFlag<HardEvent::MTE3_V>(evMte3V_);   // 级间：第一级末批 CopyOut（MTE3）读毕
            WaitFlag<HardEvent::MTE3_V>(evMte3V_);  // 后 V 方可重写压缩 lane
        }
        SetFlag<HardEvent::V_MTE2>(evVmte2_);
        WaitFlag<HardEvent::V_MTE2>(evVmte2_);

        CopyGatherChannels(taskLoopIndex, SLOT_RMIN_W, 4, InputTensor[taskNumPerCurLoop * IL_RMIN_W]);
        SetFlag<HardEvent::MTE2_V>(evMte2V_);
        WaitFlag<HardEvent::MTE2_V>(evMte2V_);
        if (taskLoopIndex > 0) {
            SetFlag<HardEvent::MTE3_V>(evMte3V_);
            WaitFlag<HardEvent::MTE3_V>(evMte3V_);
        }
        SetFlag<HardEvent::V_MTE2>(evVmte2_);
        WaitFlag<HardEvent::V_MTE2>(evVmte2_);
        CopyGatherChannels(taskLoopIndex, SLOT_MX, 6, InputTensor);
        GaussIndexCopyIn(taskLoopIndex, taskNumPerCurLoop, 0);
        DepthCopyIn(taskLoopIndex, taskNumPerCurLoop, 0);
    }

    // =======================================================================
    // CopyIn —— 第三级：gather_mask GM 本核行槽 12–19 → UB（8 路）。
    // 尾批 pad 由计数参数化 VF 掩码排除（§9.1 ② 等价机制），无需显式 pad 门控。
    // =======================================================================
    __aicore__ inline void CopyInFromBoundFilter(uint32_t taskLoopIndex)
    {
        if (taskLoopIndex == 0) {
            SetFlag<HardEvent::MTE3_V>(evMte3V_);
            WaitFlag<HardEvent::MTE3_V>(evMte3V_);
        }
        SetFlag<HardEvent::V_MTE2>(evVmte2_);
        WaitFlag<HardEvent::V_MTE2>(evVmte2_);

        CopyGatherChannels(taskLoopIndex, SLOT2_MX, 6, InputTensor);
        SetFlag<HardEvent::V_MTE2>(evVmte2_);
        WaitFlag<HardEvent::V_MTE2>(evVmte2_);
        GaussIndexCopyIn(taskLoopIndex, taskNumPerCurLoop, 1);
        DepthCopyIn(taskLoopIndex, taskNumPerCurLoop, 1);
    }

    // =======================================================================
    // 第二级 · BoundFilter（S4 谓词 + 压缩写槽 12–19）
    // =======================================================================
    __aicore__ inline void BoundFilter(uint32_t taskLoopIndex, float tileGridX, float tileGridY)
    {
        // all_in 谓词 → 0/1 掩码物化（§5.3 S4 ← 公式 2)）
        BoundMask(tileGridX, tileGridY, curValidLen_);
        
        SetFlag<HardEvent::MTE2_V>(evMte2V_);
        WaitFlag<HardEvent::MTE2_V>(evMte2V_);

        uint64_t rsvdGSCnt = 0;
        uint64_t realGSCnt = GaussIndexGather(taskLoopIndex, 1);
        uint64_t copyLength = Ceil8(realGSCnt);
        // copyout: depths（先出：压缩 lane 随后被 means/opacity/conics 复用；
        // 索引 lane 的压缩回写在 GaussIndexGather stage-1 内部完成）
        DepthGather(rsvdGSCnt);

        SetFlag<HardEvent::V_MTE3>(evVmte3_);
        WaitFlag<HardEvent::V_MTE3>(evVmte3_);

        DepthCopyOut(copyLength, 1);
        SetFlag<HardEvent::MTE3_V>(evMte3V_);
        WaitFlag<HardEvent::MTE3_V>(evMte3V_);

        Means2dGather(rsvdGSCnt);
        OpacityGather(rsvdGSCnt);
        ConicsGather(rsvdGSCnt);

        SetFlag<HardEvent::V_MTE3>(evVmte3_);
        WaitFlag<HardEvent::V_MTE3>(evVmte3_);

        Means2dCopyOut(copyLength, 1);
        OpacityCopyOut(copyLength, 1);
        ConicsCopyOut(copyLength, 1);
        GaussIndexCopyOut(copyLength, 1);

        gatherStartIndex += realGSCnt;
    }

    // =======================================================================
    // 第三级 · FlagFilter（S5–S9 ← 公式 3)/4)/5)）—— 精过滤 + 输出压缩
    // =======================================================================
    __aicore__ inline void FlagFilter(float tileGridX, float tileGridY, uint32_t taskLoopIndex)
    {
        // 级内到达屏障：CopyInFromBoundFilter 的 MTE2 拷贝（means/opacity/conics/gidx/
        // depths）读毕方可进入 VF 谓词求值（参考实现 BlockContainsCenter 前置同款）
        SetFlag<HardEvent::MTE2_V>(evMte2V_);
        WaitFlag<HardEvent::MTE2_V>(evMte2V_);

        // mask = center ∨ isect 谓词 → 0/1 掩码物化（闭区间 GE + 椭圆判别链）
        FineMask(tileGridX, tileGridY, curValidLen_);

        SetFlag<HardEvent::MTE2_V>(evMte2V_);
        WaitFlag<HardEvent::MTE2_V>(evMte2V_);

        uint64_t rsvdGSCnt = 0;
        uint64_t realGSCnt = GaussIndexGather(taskLoopIndex, 1);
        uint64_t copyLength = Ceil8(realGSCnt);
        DepthGather(rsvdGSCnt);  // depths 直通压缩（无放缩，NaN bitwise 保留）

        // 输出段基址：tile_sum 平铺索引 = 任务索引（(b·C+c)·numTile + t 段内前缀偏移）
        uint32_t bcIdx = curBcIdx_;
        uint32_t tIdx = (taskStart_ + curTaskIdx_) % numTile;
        uint64_t segBase = (static_cast<uint64_t>(bcIdx) * numTile + tIdx) * gaussNum;

        SetFlag<HardEvent::V_MTE3>(evVmte3_);
        WaitFlag<HardEvent::V_MTE3>(evVmte3_);

        CopyUb2GmPad(gaussindexGM, segBase + gatherStartIndex, segBase + gaussNum, GatherIndexTensor, copyLength);
        CopyUb2GmPad(tiledepthsGM, segBase + gatherStartIndex, segBase + gaussNum,
                     ComputingTensor[taskNumPerCurLoop * GL_DEPTH], copyLength);

        SetFlag<HardEvent::MTE3_V>(evMte3V_);
        WaitFlag<HardEvent::MTE3_V>(evMte3V_);
        gatherStartIndex += realGSCnt;
    }

    // =======================================================================
    // 谓词物化入口（0/1 fp32 掩码 → MaskTensor，count = 本批有效长度）
    // =======================================================================
    __aicore__ inline void YFilterMask(float minTileGridY, float maxTileGridY, uint32_t validLen)
    {
        constexpr uint32_t vlF = GetVecLen() / sizeof(T);
        uint16_t rep = static_cast<uint16_t>((validLen + vlF - 1) / vlF);
        asc_vf_call<FlashGaussianBuildMaskYFilterMaskVF<T>>(
            (__ubuf__ T*)MaskTensor.GetPhyAddr(),
            (__ubuf__ T*)InputTensor[taskNumPerCurLoop * IL_RMIN_H].GetPhyAddr(),
            (__ubuf__ T*)InputTensor[taskNumPerCurLoop * IL_RMAX_H].GetPhyAddr(),
            static_cast<T>(minTileGridY), static_cast<T>(maxTileGridY), validLen, rep);
    }

    __aicore__ inline void BoundMask(float tileGridX, float tileGridY, uint32_t validLen)
    {
        constexpr uint32_t vlF = GetVecLen() / sizeof(T);
        uint16_t rep = static_cast<uint16_t>((validLen + vlF - 1) / vlF);
        asc_vf_call<FlashGaussianBuildMaskBoundMaskVF<T>>(
            (__ubuf__ T*)MaskTensor.GetPhyAddr(),
            (__ubuf__ T*)InputTensor[taskNumPerCurLoop * IL_RMIN_W].GetPhyAddr(),
            (__ubuf__ T*)InputTensor[taskNumPerCurLoop * IL_RMIN_H].GetPhyAddr(),
            (__ubuf__ T*)InputTensor[taskNumPerCurLoop * IL_RMAX_W].GetPhyAddr(),
            (__ubuf__ T*)InputTensor[taskNumPerCurLoop * IL_RMAX_H].GetPhyAddr(),
            static_cast<T>(tileGridX), static_cast<T>(tileGridY), static_cast<T>(tileSize), validLen, rep);
    }

    __aicore__ inline void FineMask(float tileGridX, float tileGridY, uint32_t validLen)
    {
        constexpr uint32_t vlF = GetVecLen() / sizeof(T);
        uint16_t rep = static_cast<uint16_t>((validLen + vlF - 1) / vlF);
        asc_vf_call<FlashGaussianBuildMaskFineMaskVF<T>>(
            (__ubuf__ T*)MaskTensor.GetPhyAddr(),
            (__ubuf__ T*)InputTensor[taskNumPerCurLoop * IL_MX].GetPhyAddr(),
            (__ubuf__ T*)InputTensor[taskNumPerCurLoop * IL_MY].GetPhyAddr(),
            (__ubuf__ T*)InputTensor[taskNumPerCurLoop * IL_OP].GetPhyAddr(),
            (__ubuf__ T*)InputTensor[taskNumPerCurLoop * IL_C00].GetPhyAddr(),
            (__ubuf__ T*)InputTensor[taskNumPerCurLoop * IL_C01].GetPhyAddr(),
            (__ubuf__ T*)InputTensor[taskNumPerCurLoop * IL_C11].GetPhyAddr(),
            static_cast<T>(tileGridX), static_cast<T>(tileGridY), static_cast<T>(tileSize), validLen, rep);
    }

    // =======================================================================
    // CompressWithCount — 启动 CompressVF 并回读 SPR.AR 命中计数
    // =======================================================================
    __aicore__ inline uint64_t CompressWithCount(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                                  uint32_t count)
    {
        if (count == 0) {
            return 0;  // 本批紧凑集为空：无元素可压缩
        }
        constexpr uint32_t vlCmp = GetVecLen() / sizeof(T);
        uint16_t repCmp = static_cast<uint16_t>((count + vlCmp - 1) / vlCmp);
        asc_vf_call<FlashGaussianBuildMaskCompressVF<T>>(
            (__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ T*)src.GetPhyAddr(),
            (__ubuf__ T*)MaskTensor.GetPhyAddr(), count, repCmp);
        return static_cast<uint64_t>(AscendC::Reg::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(T);
    }

    // =======================================================================
    // GaussIndexGather — 按 mask 压缩 gaussIndex lane，返回本批命中计数 rsvdGSCnt
    // （Kernel.md §9.4；stageId：0 = Y 预过滤首建索引向量、1 = 后继级随数据重压缩）
    // =======================================================================
    __aicore__ inline uint64_t GaussIndexGather(uint32_t taskLoopIndex, uint32_t stageId)
    {
        int32_t vecStartIndex = static_cast<int32_t>(taskLoopIndex) * static_cast<int32_t>(gaussNumPerLoop);
        uint32_t validLen = curValidLen_;
        if (validLen == 0) {
            return 0;
        }
        constexpr uint32_t vlIdx = GetVecLen() / sizeof(T);
        uint16_t repIdx = static_cast<uint16_t>((taskNumPerCurLoop + vlIdx - 1) / vlIdx);
        if (stageId == 0) {
            // 原始索引向量（CreateVecIndex 的 asc_vf_call 等价落地）→ 计算暂存 lane，
            // 再按掩码压缩进 GatherIndexTensor（掩码 lane 不被覆盖，供数据 lane 复用）
            asc_vf_call<FlashGaussianBuildMaskArangeVF<T>>(
                (__ubuf__ T*)ComputingTensor[taskNumPerCurLoop * CL_T0].GetPhyAddr(), vecStartIndex,
                taskNumPerCurLoop, repIdx);
            return CompressWithCount(GatherIndexTensor, ComputingTensor[taskNumPerCurLoop * CL_T0], validLen);
        }
        // 后继级：读回的紧凑索引随数据重压缩（原始序不变量载体）→ 暂存 lane，
        // 再 Adds(+0) 折回 GatherIndexTensor（参考实现 FlagFilter/BoundFilter 同款）
        uint64_t cnt = CompressWithCount(ComputingTensor[taskNumPerCurLoop * CL_T0], GatherIndexTensor, validLen);
        Adds(GatherIndexTensor, ComputingTensor[taskNumPerCurLoop * CL_T0], ZERO_FLOAT_VALUE,
             Ceil8(static_cast<uint32_t>(cnt)));
        return cnt;
    }

    // =======================================================================
    // 逐 lane Gather / CopyIn / CopyOut（槽位映射见上部 SLOT_* 常量表）
    //
    // GM↔UB 搬运统一走 DataCopyPad（1-B 粒度）+ 自然边界裁剪：3510 整张量 DataCopy
    // 以 32B 为 burst 单位且要求 GM 侧地址 32B 对齐，而本算子 channel-first 通道基址
    // （元素偏移 N、2N、3N）、gather_mask 槽位前缀与输出段前缀在 N / 命中计数非 8
    // 倍数时不对齐或越过张量末端（GM overflow 检查 → MTE 507035）；被裁剪的尾段
    // 必为各级掩码排除的 pad 区（validLen ≤ 边界余量恒成立），不损失有效数据。
    // =======================================================================
    __aicore__ inline void CopyGm2UbPad(const GlobalTensor<T>& gm, uint64_t gmOffset, uint64_t gmEnd,
                                         const LocalTensor<T>& dst, uint32_t elemCount)
    {
        uint64_t readable = (gmEnd > gmOffset) ? (gmEnd - gmOffset) : 0;
        uint32_t readLen = (static_cast<uint64_t>(elemCount) <= readable)
                               ? elemCount : static_cast<uint32_t>(readable);
        if (readLen == 0) {
            return;  // 零长搬运（本批紧凑集为空）
        }
        DataCopyPadParams padParams{false, 0, 0, 0};  // 纯搬运，不注入 pad 值
        DataCopyParams copyParams{1, static_cast<uint16_t>(readLen * sizeof(T)), 0, 0};
        DataCopyPad(dst, gm[gmOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyUb2GmPad(GlobalTensor<T>& gm, uint64_t gmOffset, uint64_t gmEnd,
                                         const LocalTensor<T>& src, uint32_t elemCount)
    {
        uint64_t writable = (gmEnd > gmOffset) ? (gmEnd - gmOffset) : 0;
        uint32_t writeLen = (static_cast<uint64_t>(elemCount) <= writable)
                                ? elemCount : static_cast<uint32_t>(writable);
        if (writeLen == 0) {
            return;  // 零长写出（本批紧凑集为空）
        }
        DataCopyParams copyParams{1, static_cast<uint16_t>(writeLen * sizeof(T)), 0, 0};
        DataCopyPad(gm[gmOffset], src, copyParams);
    }

    __aicore__ inline void GaussIndexCopyIn(uint64_t taskLoopIndex, uint64_t copyLength, uint32_t stageId)
    {
        if (stageId == 0) {
            CopyGm2UbPad(gathermaskGM, gatherBlockStartAddr + gaussNum * SLOT_GIDX + taskLoopIndex * gaussNumPerLoop,
                         gatherBlockStartAddr + gaussNum * (SLOT_GIDX + 1), GatherIndexTensor, copyLength);
        } else {
            CopyGm2UbPad(gathermaskGM, gatherBlockStartAddr + gaussNum * SLOT2_GIDX + taskLoopIndex * gaussNumPerLoop,
                         gatherBlockStartAddr + gaussNum * (SLOT2_GIDX + 1), GatherIndexTensor, copyLength);
        }
    }

    __aicore__ inline void GaussIndexCopyOut(uint64_t copyLength, uint32_t stageId)
    {
        if (stageId == 0) {
            CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gaussNum * SLOT_GIDX + gatherStartIndex,
                         gatherBlockStartAddr + gaussNum * (SLOT_GIDX + 1), GatherIndexTensor, copyLength);
        } else {
            CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gaussNum * SLOT2_GIDX + gatherStartIndex,
                         gatherBlockStartAddr + gaussNum * (SLOT2_GIDX + 1), GatherIndexTensor, copyLength);
        }
    }

    __aicore__ inline void Means2dGather(uint64_t rsvdGSCnt)
    {
        rsvdGSCnt = CompressWithCount(ComputingTensor[taskNumPerCurLoop * GL_M0],
                                      InputTensor[taskNumPerCurLoop * IL_MX], curValidLen_);
        rsvdGSCnt = CompressWithCount(ComputingTensor[taskNumPerCurLoop * GL_M1],
                                      InputTensor[taskNumPerCurLoop * IL_MY], curValidLen_);
    }


    __aicore__ inline void Means2dCopyOut(uint64_t copyLength, uint32_t stageId)
    {
        if (stageId == 0) {
            CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex,
                         gatherBlockStartAddr + gaussNum * (SLOT_MX + 1),
                         ComputingTensor[taskNumPerCurLoop * GL_M0], copyLength);
            CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex + gaussNum,
                         gatherBlockStartAddr + gaussNum * (SLOT_MY + 1),
                         ComputingTensor[taskNumPerCurLoop * GL_M1], copyLength);
        } else {
            CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex + gaussNum * SLOT2_MX,
                         gatherBlockStartAddr + gaussNum * (SLOT2_MX + 1),
                         ComputingTensor[taskNumPerCurLoop * GL_M0], copyLength);
            CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex + gaussNum * SLOT2_MY,
                         gatherBlockStartAddr + gaussNum * (SLOT2_MY + 1),
                         ComputingTensor[taskNumPerCurLoop * GL_M1], copyLength);
        }
    }

    __aicore__ inline void OpacityGather(uint64_t rsvdGSCnt)
    {
        rsvdGSCnt = CompressWithCount(ComputingTensor[taskNumPerCurLoop * GL_OP],
                                      InputTensor[taskNumPerCurLoop * IL_OP], curValidLen_);
    }


    __aicore__ inline void OpacityCopyOut(uint64_t copyLength, uint32_t stageId)
    {
        if (stageId == 0) {
            CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex + gaussNum * SLOT_OP,
                         gatherBlockStartAddr + gaussNum * (SLOT_OP + 1),
                         ComputingTensor[taskNumPerCurLoop * GL_OP], copyLength);
        } else {
            CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex + gaussNum * SLOT2_OP,
                         gatherBlockStartAddr + gaussNum * (SLOT2_OP + 1),
                         ComputingTensor[taskNumPerCurLoop * GL_OP], copyLength);
        }
    }

    __aicore__ inline void ConicsGather(uint64_t rsvdGSCnt)
    {
        rsvdGSCnt = CompressWithCount(ComputingTensor[taskNumPerCurLoop * GL_C00],
                                      InputTensor[taskNumPerCurLoop * IL_C00], curValidLen_);
        rsvdGSCnt = CompressWithCount(ComputingTensor[taskNumPerCurLoop * GL_C01],
                                      InputTensor[taskNumPerCurLoop * IL_C01], curValidLen_);
        rsvdGSCnt = CompressWithCount(ComputingTensor[taskNumPerCurLoop * GL_C11],
                                      InputTensor[taskNumPerCurLoop * IL_C11], curValidLen_);
    }


    __aicore__ inline void ConicsCopyOut(uint64_t copyLength, uint32_t stageId)
    {
        if (stageId == 0) {
            CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex + gaussNum * SLOT_C00,
                         gatherBlockStartAddr + gaussNum * (SLOT_C00 + 1),
                         ComputingTensor[taskNumPerCurLoop * GL_C00], copyLength);
            CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex + gaussNum * SLOT_C01,
                         gatherBlockStartAddr + gaussNum * (SLOT_C01 + 1),
                         ComputingTensor[taskNumPerCurLoop * GL_C01], copyLength);
            CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex + gaussNum * SLOT_C11,
                         gatherBlockStartAddr + gaussNum * (SLOT_C11 + 1),
                         ComputingTensor[taskNumPerCurLoop * GL_C11], copyLength);
        } else {
            CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex + gaussNum * SLOT2_C00,
                         gatherBlockStartAddr + gaussNum * (SLOT2_C00 + 1),
                         ComputingTensor[taskNumPerCurLoop * GL_C00], copyLength);
            CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex + gaussNum * SLOT2_C01,
                         gatherBlockStartAddr + gaussNum * (SLOT2_C01 + 1),
                         ComputingTensor[taskNumPerCurLoop * GL_C01], copyLength);
            CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex + gaussNum * SLOT2_C11,
                         gatherBlockStartAddr + gaussNum * (SLOT2_C11 + 1),
                         ComputingTensor[taskNumPerCurLoop * GL_C11], copyLength);
        }
    }

    __aicore__ inline void RectGather(uint64_t rsvdGSCnt)
    {
        rsvdGSCnt = CompressWithCount(ComputingTensor[taskNumPerCurLoop * GL_M0],
                                      InputTensor[taskNumPerCurLoop * IL_RMIN_W], curValidLen_);
        rsvdGSCnt = CompressWithCount(ComputingTensor[taskNumPerCurLoop * GL_M1],
                                      InputTensor[taskNumPerCurLoop * IL_RMIN_H], curValidLen_);
        rsvdGSCnt = CompressWithCount(ComputingTensor[taskNumPerCurLoop * GL_OP],
                                      InputTensor[taskNumPerCurLoop * IL_RMAX_W], curValidLen_);
        rsvdGSCnt = CompressWithCount(ComputingTensor[taskNumPerCurLoop * GL_C00],
                                      InputTensor[taskNumPerCurLoop * IL_RMAX_H], curValidLen_);
    }


    __aicore__ inline void RectCopyOut(uint64_t copyLength)
    {
        CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex + gaussNum * SLOT_RMIN_W,
                     gatherBlockStartAddr + gaussNum * (SLOT_RMIN_W + 1),
                     ComputingTensor[taskNumPerCurLoop * GL_M0], copyLength);
        CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex + gaussNum * SLOT_RMIN_H,
                     gatherBlockStartAddr + gaussNum * (SLOT_RMIN_H + 1),
                     ComputingTensor[taskNumPerCurLoop * GL_M1], copyLength);
        CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex + gaussNum * SLOT_RMAX_W,
                     gatherBlockStartAddr + gaussNum * (SLOT_RMAX_W + 1),
                     ComputingTensor[taskNumPerCurLoop * GL_OP], copyLength);
        CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex + gaussNum * SLOT_RMAX_H,
                     gatherBlockStartAddr + gaussNum * (SLOT_RMAX_H + 1),
                     ComputingTensor[taskNumPerCurLoop * GL_C00], copyLength);
    }

    __aicore__ inline void DepthGather(uint64_t rsvdGSCnt)
    {
        rsvdGSCnt = CompressWithCount(ComputingTensor[taskNumPerCurLoop * GL_DEPTH],
                                      InputTensor[taskNumPerCurLoop * IL_DEPTHS], curValidLen_);
    }

    __aicore__ inline void DepthCopyIn(uint64_t taskLoopIndex, uint64_t copyLength, uint32_t stageId)
    {
        if (stageId == 0) {
            CopyGm2UbPad(gathermaskGM, gatherBlockStartAddr + gaussNum * SLOT_DEPTHS + taskLoopIndex * gaussNumPerLoop,
                         gatherBlockStartAddr + gaussNum * (SLOT_DEPTHS + 1), InputTensor[taskNumPerCurLoop * IL_DEPTHS],
                         copyLength);
        } else {
            CopyGm2UbPad(gathermaskGM,
                         gatherBlockStartAddr + gaussNum * SLOT2_DEPTHS + taskLoopIndex * gaussNumPerLoop,
                         gatherBlockStartAddr + gaussNum * (SLOT2_DEPTHS + 1),
                         InputTensor[taskNumPerCurLoop * IL_DEPTHS], copyLength);
        }
    }

    __aicore__ inline void DepthCopyOut(uint64_t copyLength, uint32_t stageId)
    {
        if (stageId == 0) {
            CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex + gaussNum * SLOT_DEPTHS,
                         gatherBlockStartAddr + gaussNum * (SLOT_DEPTHS + 1),
                         ComputingTensor[taskNumPerCurLoop * GL_DEPTH], copyLength);
        } else {
            CopyUb2GmPad(gathermaskGM, gatherBlockStartAddr + gatherStartIndex + gaussNum * SLOT2_DEPTHS,
                         gatherBlockStartAddr + gaussNum * (SLOT2_DEPTHS + 1),
                         ComputingTensor[taskNumPerCurLoop * GL_DEPTH], copyLength);
        }
    }

    // =======================================================================
    // CopyOutTileSum — tile_sum 分块批量写回（int32 计数直出；块内任务连续 →
    // GM 段连续；blockLen 不要求 32B 对齐，DataCopyPad 口径）
    // =======================================================================
    __aicore__ inline void CopyOutTileSum(uint32_t taskBase, uint32_t tileCnt)
    {
        SetFlag<HardEvent::V_MTE3>(evVmte3_);
        WaitFlag<HardEvent::V_MTE3>(evVmte3_);

        DataCopyParams dataCopyParams{1, static_cast<uint16_t>(tileCnt * sizeof(int32_t)), 0, 0};
        DataCopyPad(tilesumGM[taskBase], TileSumTensor, dataCopyParams);

        SetFlag<HardEvent::MTE3_V>(evMte3V_);
        WaitFlag<HardEvent::MTE3_V>(evMte3V_);
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> InputTensorBuffer, MaskTensorBuffer;
    TBuf<TPosition::VECCALC> ComputingTensorBuffer, GatherIndexTensorBuffer, TileSumTensorBuffer;
    LocalTensor<T> InputTensor, MaskTensor, ComputingTensor, GatherIndexTensor;
    LocalTensor<int32_t> TileSumTensor;

    GlobalTensor<T> means2dGM;
    GlobalTensor<T> opacityGM;
    GlobalTensor<T> conicsGM;
    GlobalTensor<T> covars2dGM;
    GlobalTensor<T> depthsGM;
    GlobalTensor<int32_t> cntGM;
    GlobalTensor<T> tilegridGM;
    GlobalTensor<T> gathermaskGM;
    GlobalTensor<T> gaussindexGM;
    GlobalTensor<T> tiledepthsGM;
    GlobalTensor<int32_t> tilesumGM;

    uint64_t means2dEnd_ = 0;    // 各输入张量元素总数（GM OOB 裁剪边界）
    uint64_t opacityEnd_ = 0;
    uint64_t conicsEnd_ = 0;
    uint64_t covars2dEnd_ = 0;
    uint64_t depthsEnd_ = 0;

    const FlashGaussianBuildMaskTilingData* td_;
    event_t evMte2V_;  // MTE2 → V（CopyIn 后、Compute 前；Init 经 FetchEventID 赋值）
    event_t evVmte2_;  // V → MTE2（Compute 读毕、下批 CopyIn 重写前，单缓冲复用）
    event_t evVmte3_;  // V → MTE3（压缩 lane 写完、写 GM 前）
    event_t evMte3V_;  // MTE3 → V（写 GM 读毕、下批 V 重写前）

    float imageWidth;
    float imageHeight;
    float tileSize;
    uint32_t taskStart_;      // 本核任务段基址（长核 blockIdx×taskNumPerLcore / 短核段式）
    uint32_t taskNum_;        // 本核任务段长（长短核差 ≤ 1）
    uint32_t curBcIdx_;       // 当前视图（UINT32_MAX 促使首任务触发视图切换）
    uint32_t curTaskIdx_;     // 段内任务序（FlagFilter 输出段 tIdx 解码用）
    uint32_t batchSize;
    uint32_t cameraNum;
    uint32_t numTile;
    uint32_t gaussNum;
    uint32_t gaussNumPerLoop;
    uint32_t cntGaussNum;     // 当前级紧凑集总长（批界）
    uint32_t curValidLen_;    // 本批有效长度（VF count；pad 区掩码排除）
    uint32_t yFilterGaussNum; // 第一级 Y 紧凑集总长（第二级批界）
    uint32_t tailNum;         // 尾批 pad 数（VF count 掩码排除区段长）
    int32_t taskLoop;         // 当前级批数
    uint32_t taskNumPerCurLoop;  // 当前批 Ceil8 对齐 lane 长
    uint64_t blockIndex;
    uint64_t gatherStartIndex;    // 列表前缀偏移（下一级中转 / 输出写回基址）
    uint64_t gatherBlockStartAddr;  // 本核独占 gather_mask 行基址 blockIndex·gaussNum·20
};
