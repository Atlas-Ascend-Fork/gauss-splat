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
// flash_gaussian_build_mask_package/op_kernel/arch35/flash_gaussian_build_mask_empty.h
// =============================================================================
//
// ROLE: empty 短路模板（tilingKey=1）设备侧 kernel，FlashGaussianBuildMask on
//   arch35（Ascend 950 / A5，DAV_3510，AIV-only，regbase 纯 Vector）。
//   实现 docs/flash_gaussian_build_mask/design/Kernel.md §9.3 空 tensor kernel 行与
//   docs/flash_gaussian_build_mask/design/branches/DESIGN-BRANCH-1.md §3–§5：
//   tile_sum 零填充直出（reduction 范式 EMPTY_R「Duplicate 固化值 → CopyOut」退化
//   形态——公式 5) 空集求和 tile_sum ≡ 0），gauss_index/tile_depths 为 0 元素空列表
//   无写回，fillTotal=0（B=0/C=0）唯一在环核空转早退。
//
//   进入条件（DESIGN-BRANCH-1.md §0 / BranchRoute.md §2）：任一规模维为 0
//   （N == 0 或 B == 0 或 C == 0）——host TilingFunc 置
//   SetTilingKey(GET_TPL_TILING_KEY(1)) 并填写 FlashGaussianBuildMaskEmptyTilingData。
//
// 3510 落地差异（相对 DESIGN-BRANCH-1.md §5 regbase 伪码，数学链与 §5.1–§5.5
// 逐步一致，仅 API 承载按 3510 SDK 实际形态调整，与 normal 主模板同款）：
//   1. TBuf 分配：伪码 pipe_->AllocTBuf<TPosition::VECCALC>(bytes) 在本版 SDK
//      TPipe 无此接口，改 pipe_->InitBuffer(fillTBuf_, bytes)（Branch-0 同款）；
//   2. 事件 ID 预取：FetchEventID 以 HardEvent 方向参数经 GetTPipePtr() 动态获取
//      （A5 规范禁止硬编码 EVENT_ID；Branch-0 同款）；
//   3. GM 写回统一 DataCopyPad（blockLen 不要求 32B 对齐，1B 粒度）——尾块
//      chunkLen=remain 任意余数直写，无对齐补齐（§5.3 要点 ①）。
//
// UB 内存规划（DESIGN-BRANCH-1.md §4，TBuf 单缓冲 DB=1、无 TQue）：1 路 int32
// 填充 lane × FILL_BUF_ELEMS=8192 元素 = 32,768 B ≤ 253,952 B（A5 UB）；
// blockLen 上界 32,768 ≤ uint16_t 65,535（§4 三重上界第 ② 条）。
//
// 同步（§5.5）：无 CopyIn / 无 MTE2 通道活动；每填充块两对事件
// （V_MTE3：Duplicate 写 fillBuf 后 DataCopyPad 读；MTE3_V：CopyOut 读毕后
// fillBuf 复用于下一块）；无跨核 SyncAll（各核填充段互不相交）；fillTotal=0
// 早退路径在任何 buffer/事件操作前 return（零事件、零 GM 读写）。
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : FlashGaussianBuildMask   — FlashGaussianBuildMaskEmptyKernel
//   snake_case   : flash_gaussian_build_mask  — filename
//   UPPER_SNAKE  : FLASH_GAUSSIAN_BUILD_MASK  — (not used directly)
// =============================================================================
// FORMULA ANCHOR (docs/flash_gaussian_build_mask/design/branches/DESIGN-BRANCH-1.md §5):
//   tile_sum[b,c,t] = Σ_{g∈∅} mask[g,t] = 0   (公式 5) 空集求和退化)
//   gauss_index / tile_depths: 尾维 N=0 → 0 元素空列表，无写回
// =============================================================================
// FORMULA ANCHOR (docs/flash_gaussian_build_mask/design/branches/DESIGN-BRANCH-1.md §5.1):
//   fillStart_/fillNum_ 长短核段映射: 长核 blockIdx·fillNumPerLcore /
//   短核 numLcore·fillNumPerLcore + (blockIdx−numLcore)·fillNumPerScore (§2)
// =============================================================================
// FORMULA ANCHOR (docs/flash_gaussian_build_mask/design/branches/DESIGN-BRANCH-1.md §5.3):
//   Process 填充循环: chunkLen=min(remain,FILL_BUF_ELEMS) → Duplicate 0 →
//   CopyOut(fillStart_+offset) → offset/remain 推进，直至本核段耗尽
// =============================================================================

#pragma once

#include "kernel_operator.h"                          // Ascend C kernel framework (AscendC:: namespace)
#include "flash_gaussian_build_mask_tiling_struct.h"  // FlashGaussianBuildMaskEmptyTilingData

using namespace AscendC;

// ---------------------------------------------------------------------------
// 常量（DESIGN-BRANCH-1.md §4：1 路 int32 填充 lane 的填充块粒度）
//   8192×4 = 32,768 B ≤ 253,952 B（UB，占 12.9%）；≤ 65,535（DataCopyParams
//   .blockLen uint16 上界，单位字节——16,384 元素 = 65,536 B 恰越界，故不取）；
//   2¹³ 幂次 × 4B = 32B 对齐（DMA burst 友好）。任意合法 fillNum 以
//   CeilDiv(fillNum, 8192) 块循环覆盖（§4 三重上界第 ③ 条机械保证）。
// ---------------------------------------------------------------------------
constexpr uint32_t FILL_BUF_ELEMS = 8192;  // 单填充块 int32 元素数（32,768 B）

// ===========================================================================
// class FlashGaussianBuildMaskEmptyKernel<T> — empty 短路模板（tilingKey=1）
//
// 模板参数 T = DTYPE_MEANS2D（与 normal 主模板同一编译期 dtype 实例化口径；
// empty 分支实际仅绑定 tile_sum 一路 int32 输出，BranchRoute.md 路由表；T 不进
// TilingKey，Kernel.md §1「禁止加 typename D_T」约束不冲突）。
// ===========================================================================
template <typename T>
class FlashGaussianBuildMaskEmptyKernel {
public:
    __aicore__ inline FlashGaussianBuildMaskEmptyKernel() {}

    // -----------------------------------------------------------------------
    // Init — PreInit（blockDim 校验 + blockIdx 长短核填充段映射）→ GM 绑定
    //        （仅 tile_sum，本分支唯一绑定）→ TBuf 分配（§4 lane 表）→ 事件预取
    // （DESIGN-BRANCH-1.md §5.1）
    // -----------------------------------------------------------------------
    __aicore__ inline void Init(GM_ADDR tile_sum, const FlashGaussianBuildMaskEmptyTilingData* tilingData,
                                TPipe* pipe)
    {
        // ── PreInit：blockDim 非零校验（host fillTotal=0 时置 1，严禁 0）──
        ASSERT((tilingData->blockDim != 0) && "Block Dim can not be Zero!");
        td_ = tilingData;
        blockIndex_ = GetBlockIdx();
        if (td_->fillTotal == 0) {                       // B=0/C=0：三输出均 0 元素，唯一在环核空转早退
            fillNum_ = 0;
            fillStart_ = 0;
        } else if (blockIndex_ < td_->numLcore) {        // 长核段（§2 段映射）
            fillNum_ = td_->fillNumPerLcore;
            fillStart_ = static_cast<uint32_t>(blockIndex_) * td_->fillNumPerLcore;
        } else {                                         // 短核段
            fillNum_ = td_->fillNumPerScore;
            fillStart_ = td_->numLcore * td_->fillNumPerLcore
                       + (static_cast<uint32_t>(blockIndex_) - td_->numLcore) * td_->fillNumPerScore;
        }

        // ── GM 绑定：本分支唯一绑定 tile_sum（int32 [B,C,numTile,1]，Kernel.md §4
        //    「empty 仅绑定 1 路」；其余 7 输入 + 2 输出均为空 Tensor / 空列表，
        //    不绑定、不访问）──
        gmTileSum_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tile_sum), td_->fillTotal);

        // ── TBuf 分配（§4 划分表：1 路 int32 填充 lane，8192 元素 × 4B；
        //    3510 SDK 无 TPipe::AllocTBuf，改 InitBuffer——Branch-0 同款承载）──
        pipe_ = pipe;
        pipe_->InitBuffer(fillTBuf_, FILL_BUF_ELEMS * sizeof(int32_t));
        fillTensor_ = fillTBuf_.Get<int32_t>();

        // ── 事件 ID 预取（A5 规范：FetchEventID 动态获取，禁止硬编码 EVENT_ID；
        //    §5.5 事件分配：本分支仅 V↔MTE3 两向）──
        evVmte3_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));  // V → MTE3
        evMte3V_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));  // MTE3 → V
    }

    // -----------------------------------------------------------------------
    // Process — 填充循环（§5.3 编排）：fillTotal==0（B=0/C=0）空转早退（不发起
    // 任何 GM 读写）；否则逐 FILL_BUF_ELEMS 块「Duplicate 0 → CopyOut」串行链
    // 直至本核段耗尽。gauss_index/tile_depths 为 0 元素空列表，全程无写回
    // （公式 5) 压缩域为空）。
    // -----------------------------------------------------------------------
    __aicore__ inline void Process()
    {
        if (fillNum_ == 0) {
            return;  // B=0/C=0：fillTotal=0，唯一在环核空转早退，无任何 GM 读写
        }
        uint32_t remain = fillNum_;
        uint32_t offset = 0;
        while (remain > 0) {
            uint32_t chunkLen = (remain > FILL_BUF_ELEMS) ? FILL_BUF_ELEMS : remain;  // 尾块 = remain
            // ── Compute：tile_sum[b,c,t] = Σ_{g∈∅} mask = 0（公式 5) 退化；
            //    int32 标量 0 广播（3510 DuplicateImpl 计数参数化掩码，
            //    任意 chunkLen 安全，含非 8 对齐尾块）──
            // 执行前: 持有=[]                       // fillBuf 空闲（上块 MTE3_V 已释放）
            // 执行中: 持有=[fillBuf]               ← P_int32=1 峰值（本分支唯一 lane，≤ P=1）
            // 执行后: 持有=[fillBuf]               // 值就绪，待 CopyOut 消费
            Duplicate(fillTensor_, static_cast<int32_t>(0), static_cast<int32_t>(chunkLen));
            CopyOut(fillStart_ + offset, chunkLen);  // §5.4（含 V_MTE3 / MTE3_V 同步对）
            offset += chunkLen;
            remain -= chunkLen;
        }
        // gauss_index / tile_depths：尾维 N=0 空列表，无写回
    }

private:
    // =======================================================================
    // CopyOut — DataCopyPad 单路径 dense 写回 tile_sum GM 段（§5.4；参考实现
    // tile_sum 写回同款）：blockLen = chunkLen × sizeof(int32_t) ≤ 32,768 B <
    // uint16 上界 65,535，不要求 32B 对齐；写回范围 = 本核段
    // [fillStart_, fillStart_+fillNum_) 逐块零填充，全 fillTotal = B·C·numTile
    // 元素跨核无缝覆盖（§2 长短核段映射，段间不重叠、不留隙）。
    // =======================================================================
    __aicore__ inline void CopyOut(uint32_t gmOffset, uint32_t chunkLen)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVmte3_);    // fillBuf: V 写完成 → MTE3 可读（§5.5）
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVmte3_);
        AscendC::DataCopyParams dataCopyParams{1, static_cast<uint16_t>(chunkLen * sizeof(int32_t)), 0, 0};
        AscendC::DataCopyPad(gmTileSum_[gmOffset], fillTensor_, dataCopyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);    // fillBuf 释放 → 下一块 Duplicate 可复用
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> fillTBuf_;       // 1 路 int32 填充 lane（8192 元素，§4）
    LocalTensor<int32_t> fillTensor_;         // 零填充 lane：Duplicate 目的（V 写）→ DataCopyPad 源（MTE3 读）
    GlobalTensor<int32_t> gmTileSum_;         // tile_sum [B,C,numTile,1]（本分支唯一 GM 绑定）

    const FlashGaussianBuildMaskEmptyTilingData* td_;
    event_t evVmte3_;                         // V → MTE3（Duplicate 写 fillBuf 后 DataCopyPad 读）
    event_t evMte3V_;                         // MTE3 → V（CopyOut 完成后 fillBuf 复用于下一块）
    uint64_t blockIndex_;                     // 本核 blockIdx（长短核段判定）
    uint32_t fillStart_;                      // 本核填充段基址（长核 blockIdx×fillNumPerLcore / 短核段式）
    uint32_t fillNum_;                        // 本核填充段长（长短核差 ≤ 1；0 = 空转早退）
};
