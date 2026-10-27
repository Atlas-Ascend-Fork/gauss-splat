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
// operators/calc_render_simt_fwd_package/op_kernel/arch35/fwd_simt_vf.h
// =============================================================================
//
// calc_render_simt_fwd SIMT VF 线程体（设计事实源 docs/calc_render_simt_fwd/design/
// Kernel.md §9 / DESIGN-BRANCH-0.md §5.3；单分支 tilingKey=0 simt-universal）。
//
// - 全部 SIMT 编译期常量收敛于本头文件（DevView.md「编译期常量与归属」）：
//   THREAD_NUM = 1024 与 LAUNCH_BOUND / Simt::Dim3 同一 constexpr（SIMT 铁律，
//   禁止从 tiling 数据动态获取线程数）。
// - 唯一 VF 函数 FwdSimtCompute：1024 线程 × 1 像素/线程逐像素串行状态机，
//   承载 EWA 二次型、双 clip（clip 1 三目比较 NaN 传播 / clip 2 微贡献跳过）、
//   alpha blending 累积（T_{k-1} 前置透射率语义裁定）、per-pixel done、Kahan
//   保序累加（Overview.md 数学语义 #1–#6）。
// - VF 内辅助函数 TileDoneReduce：tile-done 两级 warp 归约（非 VF 入口，无
//   __simt_vf__ 标记；由 FwdSimtCompute 末尾调用，全体 1024 线程参与）。
//
// include 纪律（Kernel.md §9.1，T6：禁 PipeBarrier 模板实例化错误）：仅允许
// kernel_operator.h + math_functions.h + device_warp_functions.h +
// device_sync_functions.h；禁止 device_functions.h / asc_simt.h。
// =============================================================================

#ifndef CALC_RENDER_SIMT_FWD_SIMT_VF_H
#define CALC_RENDER_SIMT_FWD_SIMT_VF_H

#include "kernel_operator.h"
#include "simt_api/math_functions.h"         // expf / logf（1-ulp 硬件 intrinsic）
#include "simt_api/device_warp_functions.h"  // asc_any / asc_reduce_add(int32_t)
#include "simt_api/device_sync_functions.h"  // asc_threadfence_block / asc_syncthreads

// ===== SIMT 编译期常量（DevView.md「编译期常量与归属」）=====
constexpr uint32_t THREAD_NUM = 1024;               // = nPixel：threadIdx → 像素下标（1 像素/线程）
constexpr uint32_t PIXEL_PER_THREAD = 1;            // 每线程 1 像素
constexpr uint32_t WARP_SIZE = 32;                  // warp lane 数
constexpr uint32_t NUM_WARPS = THREAD_NUM / WARP_SIZE;  // 32 warp（1024/32）
constexpr uint32_t N_PIXEL_FIXED = 1024;            // tile 像素数合同（T1；host 断言同值）
constexpr uint32_t N_PIXEL_1D = 32;                 // tile 边长硬编码（基线 sqrt 改硬编码，T1）
constexpr uint32_t MAX_TILE_SIZE = 32;              // tile 边长上限（沿用基线）
constexpr uint32_t NUM_GS_ATTRIBUTES = 10;          // gs 属性维
constexpr uint32_t GS_ATTR_STRIDE = 16;             // 属性段 16 float（64B）对齐，前 10 列有效
constexpr uint32_t MIN_DATACOPY_LEN = 16;           // 属性段 DataCopy 定长（同 GS_ATTR_STRIDE，沿用基线）
constexpr uint32_t KERNEL_GS_PER_BATCH = 256;       // 每批高斯数（一次 VF_CALL / 一次协同载入，T12）
constexpr uint32_t SCALAR_PARAMS_COUNT = 5;         // VF 批参数个数（int64）
constexpr float ALPHA_CLIP = 0.999f;                // 公式 #2：α 上界裁剪（clip 1）
constexpr float ALPHA_THRESHOLD = 1.0f / 255.0f;    // 公式 #3：微贡献跳过阈值（clip 2）
constexpr float PER_PIXEL_DONE_THRESHOLD = 1.0e-4f; // 公式 #4：per-pixel done 阈值

// 属性布局（gsAttrBatch 16 float 段，前 10 列有效；沿用基线 ATTR_* 常量序）
constexpr uint8_t ATTR_MEAN_X = 0, ATTR_MEAN_Y = 1;
constexpr uint8_t ATTR_CONIC_0 = 2, ATTR_CONIC_1 = 3, ATTR_CONIC_2 = 4;
constexpr uint8_t ATTR_OPACITY = 5, ATTR_COLOR_R = 6, ATTR_COLOR_G = 7;
constexpr uint8_t ATTR_COLOR_B = 8, ATTR_DEPTH = 9;

// -----------------------------------------------------------------------------
// tile-done 两级 warp 归约（Kernel.md §9.4 / DESIGN-BRANCH-0.md §5.3）：
// 全体 1024 线程（含 P5 outbound 线程，其 doneMask=1 计入统计）各持 1 像素 done 位，
// warp 内 asc_reduce_add 归约 → lane0 分槽写 warpReduce[warpId]（32 warp × int32，
// 分槽写无并发冲突）→ tid0 串行汇总写 tileSync[0]（唯一写者）。同地址并发写以
// 写者分区规避，全程无原子。
// __simt_callee__ 标记：VF（__simt_vf__）函数只能调用 simt_callee 函数
// （arch35 编译器调用约定；simt_api 内部 asc_* 原语同为 __simt_callee__）。
// -----------------------------------------------------------------------------
__simt_callee__ __aicore__ inline void TileDoneReduce(uint64_t tid, uint32_t warpId, uint32_t laneId,
                                      __ubuf__ uint8_t* doneMask,
                                      __ubuf__ int32_t* warpReduce, __ubuf__ uint8_t* tileSync)
{
    asc_threadfence_block();                       // ① UB 写（doneMask / 累加器写回）对块内可见
    const int32_t myDone = (int32_t)doneMask[tid]; // 本线程 done 位（outbound 恒 1）
    asc_syncthreads();                             // ② 全 1024 线程到齐（含 warp 早退线程与 outbound）
    const int32_t warpSum = asc_reduce_add(myDone);  // ③ warp 内 32 lane 归约（硬件原语）
    if (laneId == 0) {                             // ④ lane0 写本 warp 槽（分槽写，无并发冲突）
        warpReduce[warpId] = warpSum;
    }
    asc_syncthreads();                             // ⑤ 32 个 warp 槽全部写入完成
    if (tid == 0) {                                // ⑥ tid0 串行汇总 32 warp（唯一写者，无冲突）
        int32_t total = 0;
        for (uint32_t w = 0; w < NUM_WARPS; w++) {
            total += warpReduce[w];
        }
        tileSync[0] = (total >= (int32_t)THREAD_NUM) ? (uint8_t)1 : (uint8_t)0;  // 全 1024 像素 done
    }
    asc_syncthreads();                             // ⑦ tid0 写完成对全线程可见；VF 返回后经 V_S
                                                   //    事件对 SIMD S 流水可见（§6 事件链，T3）
}

// -----------------------------------------------------------------------------
// FwdSimtCompute VF 线程体（Kernel.md §9.3 完整实现链；14 个 __ubuf__ 形参，无
// __gm__ / 无标量参数——三类输入全经 UB 中转、批参数经 scalarParams 区段下发，
// T12 消除 VF 线程 GM 随机读）：
//   [0] tileCoordX / [1] tileCoordY   像素坐标（只读共享，SIMD MTE2 写入）
//   [2..5] pixOutR/G/B/D              四路输出累加器（读改写，跨批持久化 T4）
//   [6] lastCumsum / [7] kahanErr     ln 域累积透射率 s / Kahan 补偿 e（读改写）
//   [8] lastIds                       int32 逐像素最后贡献高斯 id（读改写）
//   [9] doneMask                      per-pixel done（读改写，跨批持久化）
//   [10] scalarParams                 5×int64：prevOffset/currOffset/nPixel/calPixel/batchSize
//   [11] gsAttrBatch                  [batchSize][16] 高斯属性批区（只读，前 10 列有效）
//   [12] gsIdsBatch                   [batchSize] 高斯全局 id 批区（VF 不再读取——属性已按
//                                     排序连续搬入 gsAttrBatch；保留形参仅为与 host
//                                     asc_vf_call 14 实参签名对齐，正文不消费）
//   [13] tileSync                     tile-done/warp 归约联合区段（写）：tileSync[0] 为
//                                     tile-done 标志，(__ubuf__ int32_t*)&tileSync[32] 为
//                                     warpReduceUb_（布局约束：warpReduceUb_ 紧邻
//                                     tileDoneUb_ 32B 槽之后，见编排类 UbBuffInit）
// -----------------------------------------------------------------------------
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void FwdSimtCompute(
    __ubuf__ float*    tileCoordX,
    __ubuf__ float*    tileCoordY,
    __ubuf__ float*    pixOutR,
    __ubuf__ float*    pixOutG,
    __ubuf__ float*    pixOutB,
    __ubuf__ float*    pixOutD,
    __ubuf__ float*    lastCumsum,
    __ubuf__ float*    kahanErr,
    __ubuf__ int32_t*  lastIds,
    __ubuf__ uint8_t*  doneMask,
    __ubuf__ int64_t*  scalarParams,
    __ubuf__ float*    gsAttrBatch,
    __ubuf__ int64_t*  gsIdsBatch,
    __ubuf__ uint8_t*  tileSync)
{
    // ===== 头部：索引计算 + 标量参数首读（Kernel.md §9.2）=====
    const uint64_t tid    = AscendC::Simt::GetThreadIdx();   // 0..1023
    const uint64_t pixIdx = tid;                             // 1 像素/线程：线程号即像素下标
    const uint32_t warpId = (uint32_t)(tid / WARP_SIZE);     // tid >> 5
    const uint32_t laneId = (uint32_t)(tid % WARP_SIZE);     // tid & 31
    const int64_t calPixel  = scalarParams[3];               // 实际计算像素数（== 1024）
    const int64_t batchSize = scalarParams[4];               // 本批高斯数 ≤ KERNEL_GS_PER_BATCH(256)
    const int64_t posBase   = scalarParams[0];               // 本批在排序列表中的起始位置（bwd 同系）
    __ubuf__ int32_t* warpReduce = (__ubuf__ int32_t*)&tileSync[32];  // warp 归约区（紧邻 tile-done 槽）

    // ===== P5 越界像素防御（合同内恒 false；置 doneMask=1 并与全体线程共同走完归约，
    //       不参与则 tile-done 统计错，T9）=====
    const bool outbound = (pixIdx >= (uint64_t)calPixel);
    if (outbound) {
        doneMask[pixIdx] = 1;
    }

    if (!outbound) {
        // ===== 跨 VF 状态读回寄存器（T4：批界不重置，UB 累加器持久化；透射率寄存器化）=====
        float s    = lastCumsum[pixIdx];      // ln 域累积透射率 s = Σ ln(1−α)
        float err  = kahanErr[pixIdx];        // Kahan 补偿项 e
        float cR   = pixOutR[pixIdx];
        float cG   = pixOutG[pixIdx];
        float cB   = pixOutB[pixIdx];
        float cD   = pixOutD[pixIdx];
        int32_t lastId = lastIds[pixIdx];
        uint8_t done   = doneMask[pixIdx];
        float trans = expf(s);                // T_{k-1} = e^{s_{k-1}}（累加前透射率，T_ 保留位不传值）

        // ===== 像素坐标读入寄存器（批内只读，一次性提循环外；消除每高斯 2 次 per-thread UB
        //       读，对齐 Meta 5.3：px/py 进循环前定格，内层循环零 per-thread UB 读）=====
        const float px = tileCoordX[pixIdx];
        const float py = tileCoordY[pixIdx];

        // ===== 批内串行循环：高斯维保序（accumulation_order = sequential，T5/T11）
        //       + warp 安全一致退出（asc_any warp 级原语）。对齐 Meta 控制流：
        //       单层 if(active){ if (alpha>=thr){...} } 嵌套、无 continue （P0），
        //       10 属性命名变量一次读 + fminf 裁剪，让编译器更易调度内层循环 =====
        for (int64_t b = 0; b < batchSize; b++) {
            if (!asc_any(static_cast<int32_t>(done == 0))) {      // warp 全灭一致退出（warp 安全）
                break;
            }
            if (done == 0) {                                         // 已 done 线程空转，保持 warp 迭代对齐
                // 本高斯全局排序位置（lastIds 记排序位置，供 bwd 判界）
                const int64_t gsPos = posBase + b;
                // 10 属性命名变量一次读（gsAttrBatch 已按排序序连续搬入）
                const float gMeanX = gsAttrBatch[b * GS_ATTR_STRIDE + ATTR_MEAN_X];
                const float gMeanY = gsAttrBatch[b * GS_ATTR_STRIDE + ATTR_MEAN_Y];
                const float gConic0 = gsAttrBatch[b * GS_ATTR_STRIDE + ATTR_CONIC_0];
                const float gConic1 = gsAttrBatch[b * GS_ATTR_STRIDE + ATTR_CONIC_1];
                const float gConic2 = gsAttrBatch[b * GS_ATTR_STRIDE + ATTR_CONIC_2];
                const float gOpacity = gsAttrBatch[b * GS_ATTR_STRIDE + ATTR_OPACITY];
                const float gColorR = gsAttrBatch[b * GS_ATTR_STRIDE + ATTR_COLOR_R];
                const float gColorG = gsAttrBatch[b * GS_ATTR_STRIDE + ATTR_COLOR_G];
                const float gColorB = gsAttrBatch[b * GS_ATTR_STRIDE + ATTR_COLOR_B];
                const float gDepth = gsAttrBatch[b * GS_ATTR_STRIDE + ATTR_DEPTH];

                // 公式 #1：EWA二次型（四路累加器寄存器化；运算序镜像基线 Adds/Mul/Muls/Axpy 链）
                const float dx = px - gMeanX;                    // px - μx（二次型符号不变）
                const float dy = py - gMeanY;                    // py - μy
                float quad = (-gConic1) * (dx * dy);             // -c1·dx·dy
                const float xx = (-gConic0) * (dx * dx);         // -c0·dx²
                const float yy = (-gConic2) * (dy * dy);         // -c2·dy²
                quad += 0.5f * (xx + yy);
                const float w = expf(quad);                      // w = exp(−0.5·二次型)；quad→+∞ ⇒ w=0（P6）

                // 公式 #2：clip 1 上界裁剪（fminf，对齐Meta；NaN∈裁剪辑按硬件 fmin 语义）
                const float alpha = fminf(gOpacity * w, ALPHA_CLIP);

                // 公式 #3：clip 2 微贡献跳过（无 continue，用成立分支包裹）
                if (alpha >= ALPHA_THRESHOLD) {
                    // 公式 #4：T_{k-1} 捕获（贡献权重与 done 判据同用累加前透射率；颜色累加
                    //          先于 s 的 Kahan 更新）。经 calc_render-only 收敛实验确认：
                    //          fwd/bwd 必须同时用 T_{k-1}（标准自洽形式），勿改 Meta 片段的 T_k。
                    const float tPrev = trans;
                    const float nextT = tPrev * (1.0f - alpha);   // = T_k（数学等价）
                    if (nextT <= PER_PIXEL_DONE_THRESHOLD) {
                        done = 1;
                        lastId = (int32_t)gsPos;
                    } else {
                        const float weight = tPrev * alpha;
                        cR += gColorR * weight;
                        cG += gColorG * weight;
                        cB += gColorB * weight;
                        cD += gDepth * weight;
                        lastId = (int32_t)gsPos;
                    }
                    // Kahan 块（done 触发位亦计入 s；保序串行，禁止并行化/重排；
                    // α ≤ 0.999 ⇒ ln(0.001) ≈ −6.9078 有界，无 −∞）
                    const float y = logf(1.0f - alpha) - err;
                    const float tau = s + y;
                    err = (tau - s) - y;
                    s = tau;
                    trans = expf(s);                              // 下一高斯的累加前透射率
                }
            }
        }

        // ===== 末尾写回 UB 累加器（per-pixel 索引 pixIdx = tid 互斥，无并发写、无原子；
        //       T4 跨批持久化，批界不重置）=====
        lastCumsum[pixIdx] = s;
        // Kahan 补偿跨批状态写回（T4：本区段 tile 内承载补偿项、下批首读 e 续算；
        // 编排类侧阶段 5 写回 GM 前统一清零——error 输出为数学真值 0，spec.yaml
        // formula error = np.zeros(...)、golden.py 逐像素与 0 比对）；NaN 污染链
        // （P6：NaN 沿 active 传播使 e=NaN）的补偿项对后续批无意义，置 0
        // （NaN == NaN 判 false → 0），s 已按序传播 NaN、结果不受影响
        kahanErr[pixIdx] = (err == err) ? err : 0.0f;
        pixOutR[pixIdx] = cR;
        pixOutG[pixIdx] = cG;
        pixOutB[pixIdx] = cB;
        pixOutD[pixIdx] = cD;
        lastIds[pixIdx] = lastId;
        doneMask[pixIdx] = done;
    }

    // ===== tile-done 统计（全体线程含 outbound 参与；P4 判据来源，§9.4）=====
    TileDoneReduce(tid, warpId, laneId, doneMask, warpReduce, tileSync);
}

#endif // CALC_RENDER_SIMT_FWD_SIMT_VF_H
