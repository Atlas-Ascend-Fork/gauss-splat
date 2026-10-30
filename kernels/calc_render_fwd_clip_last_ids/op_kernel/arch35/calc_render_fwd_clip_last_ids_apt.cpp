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
// operators/calc_render_fwd_clip_last_ids_package/op_kernel/arch35/calc_render_fwd_clip_last_ids_apt.cpp
// =============================================================================
//
// calc_render_fwd_clip_last_ids kernel 入口 + SIMD 侧编排类（Ascend950 / arch35 / SIMT，
// 设计事实源 docs/calc_render_fwd_clip_last_ids/design/Kernel.md §1–§8 +
// DESIGN-BRANCH-0.md §3–§5；单分支 tilingKey=0 simt-universal，kernel 模板
// 无参、binary 实例数 = 1）。
//
// 执行流：核间跨核步进分区（loopProcess，Task 30 golden 覆盖语义）→ 逐 tile
// 阶段化流水线（Process）→ 批循环 VF_CALL FwdSimtCompute（1024 线程 × 1 像素/
// 线程，像素数学全在 VF，fwd_simt_vf.h）→ 批尾 tile-done 判定（P4）→ 五路写回。
// 单 buffer UB 布局 60640B ≤ 可用 UB 120KB（ubSize − DCACHE_RESERVED，T4）；
// 6 类 HardEvent 各 1 对组织 MTE2/V/MTE3/S 四流水数据依赖（T6，禁 PipeBarrier）。
//
// 输入 4 路（gs [NGauss,10] fp32 / tileCoords [TileNum,2,1024] fp32 /
// offsets 三段布局 int64 / gsIds [TotalGauss] int64）→ 输出 5 路（color
// [3,TileNum,1024] / depth / lastCumsum / error fp32 + lastIds int32）。
// workspace = 0（无跨核交换，核间 tile 分区互斥）。
// =============================================================================

#include "calc_render_fwd_clip_last_ids.h"

using namespace AscendC;

// 单 buffer UB 布局总字节数（DESIGN-BRANCH-0.md §4 逐段字节账合计；
// InitBuffer 按 120KB 全额申请，本常量仅作容量校验口径）
constexpr uint32_t UB_LAYOUT_BYTES = 60640;

// =============================================================================
// CalcRenderFwdClipLastidsKernel — SIMD 侧编排类（单 buffer，无 ping-pong；像素级
// 数学全部下沉 VF 线程体 fwd_simt_vf.h，编排类只做「阶段化流水线 + 批循环 +
// VF 启动 + 批尾判定 + 五路写回」；Kernel.md §2 类声明逐成员对齐）
// =============================================================================
class CalcRenderFwdClipLastidsKernel {
public:
    __aicore__ inline CalcRenderFwdClipLastidsKernel() {}

    __aicore__ inline void Init(GM_ADDR gs, GM_ADDR tileCoords, GM_ADDR offsets,
                                GM_ADDR gsIds, GM_ADDR color, GM_ADDR depth,
                                GM_ADDR lastCumsum, GM_ADDR error, GM_ADDR lastIds,
                                const CalcRenderFwdClipLastidsTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void loopProcess();                // 跨核步进分区调度（§3，Task 30 golden 覆盖语义）
    __aicore__ inline void Finish();                     // 末事件令牌消化（事件配对平衡）

private:
    __aicore__ inline void UbBuffInit(TPipe* pipe);      // 单 buffer UB 布局（T4）
    __aicore__ inline void Process(int64_t tileIdx);     // 逐 tile 阶段化流水线（§5）
    __aicore__ inline void CopyOutTile(GlobalTensor<float>& colorRGm,
                                       GlobalTensor<float>& colorGGm,
                                       GlobalTensor<float>& colorBGm,
                                       GlobalTensor<float>& depthGm,
                                       GlobalTensor<float>& lastCumsumGm,
                                       GlobalTensor<float>& errorGm,
                                       GlobalTensor<int32_t>& lastIdsGm);      // 五路写回（§8）

    // ===== 标量状态 =====
    int64_t vecIdx_;            // 核逻辑号 = GetBlockIdx()*GetSubBlockNum() + GetSubBlockIdx()
    int64_t vecNum_;            // AIV 核数（以 TilingData.blockDim 为准，T7）
    int64_t nPixel_;            // 断言/兜底校验用（== N_PIXEL_FIXED 1024，T9）
    int64_t tileNum_;           // offsets 三段布局段界推导
    int64_t nGauss_;            // gsIds 值域上界（仅校验用，不做运行时 gsIds 校验）
    int64_t nIds_;              // gsIds 长度 = TilingData.totalGauss（Task 30：段界防御
                                // 裁剪上界，对齐 golden.py:74-76 clip(segStart/segEnd,0,nIds)）
    int64_t calPixel_;          // 实际计算像素数 = min(nPixel_, MAX_TILE_SIZE*MAX_TILE_SIZE) = 1024
    int64_t nPixel1d_;          // tile 边长硬编码 32（基线运行时 sqrt 改硬编码，T1）
    // ===== 事件 ID（单 buffer，6 类各 1 对，T6；Init 一次性 Alloc、全程复用；
    //       AllocEventID 返回 TEventID(int8_t)，SetFlag/WaitFlag 形参 int32_t 直接兼容）=====
    TEventID mte3Mte2Id_;      // 单 buffer 跨 tile 串行点（上一 tile 写回 → 本 tile 载入）
    TEventID mte2VId_;         // MTE2 → V（数据搬运 → VF 计算）
    TEventID vMte3Id_;         // V → MTE3（VF 计算/清零 → 写回）
    TEventID mte2SId_;         // MTE2 → S（gsIds 段载入 → Scalar 读）
    TEventID vSId_;            // V → S（VF 写 tileDone → Scalar 读）
    TEventID sVId_;            // S → V（Scalar 写参数/复位 → VF 读）
    // ===== GM 绑定（4 输入 + 5 输出；color 按 R/G/B 三平面基地址拆分）=====
    GlobalTensor<float> gsGm_;            // [NGauss, 10]
    GlobalTensor<float> tileCoordsGm_;    // [TileNum, 2, 1024]
    GlobalTensor<int64_t> coreOffsetsGm_; // offsets 第 1 段：[0, vecNum) 各核累积调度 tile 数量
                                          //   （loopProcess 读[vecIdx-1]/[vecIdx] 定本核调度段）
    GlobalTensor<int64_t> scheduleGm_;    // offsets 第 2 段：[vecNum, vecNum+TileNum) 负载均衡 tile 调度序
    GlobalTensor<int64_t> tileOffsetsGm_; // offsets 第 3 段：[vecNum+TileNum, vecNum+2*TileNum) 累计高斯偏移 end[k]
    GlobalTensor<int64_t> gsIdsGm_;       // [TotalGauss] 深度排序高斯全局 id
    GlobalTensor<float> colorRGm_;        // color[0] 平面基址（R），[TileNum, 1024]
    GlobalTensor<float> colorGGm_;        // color[1] 平面基址（G）
    GlobalTensor<float> colorBGm_;        // color[2] 平面基址（B）
    GlobalTensor<float> depthGm_;         // [TileNum, 1024]
    GlobalTensor<float> lastCumsumGm_;    // [TileNum, 1024]
    GlobalTensor<float> errorGm_;         // [TileNum, 1024]
    GlobalTensor<int32_t> lastIdsGm_;     // [TileNum, 1024]（替代基线 gsClipIndex/alphaClipIndex，T2）
    // ===== UB 区段基址（单 buffer，TBuf VECCALC 统一划分，布局表见 UbBuffInit 注释）=====
    LocalTensor<float> tileCoordX_;       // 1024 像素 X 坐标
    LocalTensor<float> tileCoordY_;       // 1024 像素 Y 坐标
    LocalTensor<float> pixOutR_;          // R 通道输出累加器
    LocalTensor<float> pixOutG_;          // G 通道输出累加器
    LocalTensor<float> pixOutB_;          // B 通道输出累加器
    LocalTensor<float> pixOutD_;          // depth 输出累加器
    LocalTensor<float> lastCumsum_;       // ln 域累积透射率 s = Σ ln(1−α)
    LocalTensor<float> error_;            // Kahan 补偿误差累加器
    LocalTensor<int32_t> lastIds_;        // 逐像素最后贡献高斯全局索引
    LocalTensor<float> T_;                // Exp 辅助保留位（透射率实际寄存器化，不传值）
    LocalTensor<uint8_t> doneMaskUb_;     // per-pixel done（跨 VF 调用持久化）
    LocalTensor<int64_t> scalarParamsUb_; // 5×int64：prevOffset/currOffset/nPixel/calPixel/batchSize
    LocalTensor<float> gsAttrBatchUb_;    // 256 高斯 × 16 float 协同载入区（T12）
    LocalTensor<int64_t> gsIdsBatchUb_;   // 256×int64 批量 gsIds
    LocalTensor<uint8_t> tileDoneUb_;     // tile-done 标志：SIMT VF 写、SIMD Scalar 读
    LocalTensor<int32_t> warpReduceUb_;   // 32 warp × int32 归约暂存
};

// -----------------------------------------------------------------------------
// Init — GM 绑定 / 核号 / 事件 / TilingData 消费（Kernel.md §4）
// -----------------------------------------------------------------------------
__aicore__ inline void CalcRenderFwdClipLastidsKernel::Init(GM_ADDR gs, GM_ADDR tileCoords, GM_ADDR offsets,
                                                     GM_ADDR gsIds, GM_ADDR color, GM_ADDR depth,
                                                     GM_ADDR lastCumsum, GM_ADDR error, GM_ADDR lastIds,
                                                     const CalcRenderFwdClipLastidsTilingData* tilingData,
                                                     TPipe* pipe)
{
    // ===== 核号（vecNum 以 TilingData.blockDim 为准，与 host SetBlockDim / offsets
    //       第一段长度同源同值，T7；AIV-only 下 GetBlockNum 与 blockDim 一致）=====
    vecIdx_ = GetBlockIdx() * GetSubBlockNum() + GetSubBlockIdx();
    vecNum_ = (int64_t)tilingData->blockDim;

    // ===== TilingData 消费（断言/兜底 + 运行时维，T9；nPixel_1d 硬编码 32，T1）=====
    nPixel_  = tilingData->nPixel;
    nPixel1d_ = (int64_t)N_PIXEL_1D;
    calPixel_ = (nPixel_ > (int64_t)(MAX_TILE_SIZE * MAX_TILE_SIZE))
                ? (int64_t)(MAX_TILE_SIZE * MAX_TILE_SIZE) : nPixel_;   // = 1024
    tileNum_ = tilingData->tileNum;
    nGauss_  = tilingData->nGauss;
    nIds_    = tilingData->totalGauss;   // gsIds 长度（Task 30 段界防御）

    // ===== 事件 ID（单 buffer 6 类各 1 对，T6；Init 一次性 Alloc、kernel 生命周期复用，
    //       不逐 tile 重复分配释放——AllocEventID/ReleaseEventID 成对纪律）=====
    mte3Mte2Id_ = pipe->AllocEventID<HardEvent::MTE3_MTE2>();
    mte2VId_    = pipe->AllocEventID<HardEvent::MTE2_V>();
    vMte3Id_    = pipe->AllocEventID<HardEvent::V_MTE3>();
    mte2SId_    = pipe->AllocEventID<HardEvent::MTE2_S>();
    vSId_       = pipe->AllocEventID<HardEvent::V_S>();
    sVId_       = pipe->AllocEventID<HardEvent::S_V>();

    // ===== GM 绑定：4 输入 =====
    gsGm_.SetGlobalBuffer((__gm__ float *)gs);                     // [NGauss, 10]
    tileCoordsGm_.SetGlobalBuffer((__gm__ float *)tileCoords);     // [TileNum, 2, 1024]
    gsIdsGm_.SetGlobalBuffer((__gm__ int64_t *)gsIds);             // [TotalGauss]
    // offsets 三段布局（基线同款：SetGlobalBuffer 基址 + 段内偏移视图）
    coreOffsetsGm_.SetGlobalBuffer((__gm__ int64_t *)offsets);     // 第 1 段 [0, vecNum)
    scheduleGm_   = coreOffsetsGm_[vecNum_];                       // 第 2 段（tile 调度序）
    tileOffsetsGm_= scheduleGm_[tileNum_];                         // 第 3 段（累计高斯偏移 end[k]）

    // ===== GM 绑定：5 输出（color 按三平面基地址拆分，与基线一致）=====
    colorRGm_.SetGlobalBuffer((__gm__ float *)color, tileNum_ * nPixel_);  // R 平面 [0, TileNum*1024)
    colorGGm_ = colorRGm_[tileNum_ * nPixel_];                             // G 平面
    colorBGm_ = colorGGm_[tileNum_ * nPixel_];                             // B 平面
    depthGm_.SetGlobalBuffer((__gm__ float *)depth, tileNum_ * nPixel_);
    lastCumsumGm_.SetGlobalBuffer((__gm__ float *)lastCumsum);
    errorGm_.SetGlobalBuffer((__gm__ float *)error);
    lastIdsGm_.SetGlobalBuffer((__gm__ int32_t *)lastIds);                 // 替代基线双 clip 数组（T2）

    // ===== 跨 tile 串行点预置令牌（单 buffer MTE3_MTE2 配对纪律：Init 预置 1 枚，
    //       每 tile 阶段 1 Wait 消费 + CopyOut 末 Set 补充，Finish 消化末枚/空载枚）=====
    SetFlag<HardEvent::MTE3_MTE2>(mte3Mte2Id_);

    // ===== 核内共享 UB（单 buffer，T4）=====
    UbBuffInit(pipe);
}

// -----------------------------------------------------------------------------
// UbBuffInit — 单 buffer UB 布局（Kernel.md §4 / DESIGN-BRANCH-0.md §4 逐段字节账）：
// TBuf VECCALC 一次性 120KB（= ubSize 248KB − DCache 预留 128KB；容量校验在入口
// 按 TilingData.ubSize 完成），16 区段平面布局合计 60640B，各段基址 32B 对齐
// （scalarParamsUb_ 取 64B 槽：40B 有效 + 24B padding，保证 gsAttrBatchUb_ 32B
// 基址——API.md §3 回改 F1；warpReduceUb_ 紧邻 tileDoneUb_ 32B 槽之后，与 VF 内
// tileSync[32] 联合区段一致）。
// -----------------------------------------------------------------------------
__aicore__ inline void CalcRenderFwdClipLastidsKernel::UbBuffInit(TPipe* pipe)
{
    TBuf<QuePosition::VECCALC> ubBuf;
    pipe->InitBuffer(ubBuf, 120 * 1024);             // 248KB − DCache 128KB（HostTiling.md §4）
    LocalTensor<uint8_t> ub = ubBuf.Get<uint8_t>();
    tileCoordX_    = ub[0].ReinterpretCast<float>();      // 基址 0B，4KB：1024 float 像素 X 坐标
    tileCoordY_    = ub[4096].ReinterpretCast<float>();   // 4096B，4KB：1024 float 像素 Y 坐标
    pixOutR_       = ub[8192].ReinterpretCast<float>();   // 8192B，4KB：R 输出累加器
    pixOutG_       = ub[12288].ReinterpretCast<float>();  // 12288B，4KB：G 输出累加器
    pixOutB_       = ub[16384].ReinterpretCast<float>();  // 16384B，4KB：B 输出累加器
    pixOutD_       = ub[20480].ReinterpretCast<float>();  // 20480B，4KB：depth 输出累加器
    lastCumsum_    = ub[24576].ReinterpretCast<float>();  // 24576B，4KB：ln 域累积透射率 s
    error_         = ub[28672].ReinterpretCast<float>();  // 28672B，4KB：Kahan 补偿跨批状态（tile 内）/
                                                          //   写回前清零的 error 输出区（阶段 5）
    lastIds_       = ub[32768].ReinterpretCast<int32_t>();// 32768B，4KB：int32×1024
    T_             = ub[36864].ReinterpretCast<float>();  // 36864B，4KB：Exp 辅助保留位（透射率寄存器化）
    doneMaskUb_    = ub[40960].ReinterpretCast<uint8_t>();// 40960B，1KB：uint8×1024 per-pixel done
    scalarParamsUb_ = ub[41984].ReinterpretCast<int64_t>();// 41984B，64B 槽：5×int64(40B) + 24B padding
    gsAttrBatchUb_ = ub[42048].ReinterpretCast<float>();  // 42048B，16KB：256 高斯 × 16 float
    gsIdsBatchUb_  = ub[58432].ReinterpretCast<int64_t>(); // 58432B，2KB：256×int64
    tileDoneUb_    = ub[60480].ReinterpretCast<uint8_t>();// 60480B，32B 槽（1 有效）：tile-done 标志
    warpReduceUb_  = ub[60512].ReinterpretCast<int32_t>();// 60512B，128B：32 warp × int32 归约暂存
    // 合计 60640B ≈ 59.2KB ≤ 120KB 可用 UB（T4，与 DESIGN-BRANCH-0.md §4 表逐值一致）
}

// -----------------------------------------------------------------------------
// loopProcess — 核间调度（Kernel.md）
// 各核按offsets调度表（gsplat get_render_schedule 排序后的负载均衡 schedule）
// 处理自己的连续调度段，尊重host侧以高斯球预做的工作均衡：
//   startIdx = (vecIdx>0) ? coreOffsetsGm_[vecIdx-1] : 0
//   endIdx   = coreOffsetsGm_[vecIdx]
//   for s in [startIdx, endIdx): tileIdx = scheduleGm_[s]; Process(tileIdx)
// 相比原[跨核步进分区（tileIdx = vecIdx + k·vecNum）]丢掉的负载均衡，本版
// 恢复强制要求的调度模型：每个核拿到一段按高斯数均衡的连续调度 tile，
// 避免重 tile 在个别核上扎堆导致整 op 卡在最慢核（load imbalance 放大耗时）。
// 空载核（coreOffset 区间为空 / P0）循环体不执行，Finish 消化 Init 预置令牌。
// 无跨核同步（核间 tile 分区互斥，不用 SyncAll / workspace）。
// -----------------------------------------------------------------------------
__aicore__ inline void CalcRenderFwdClipLastidsKernel::loopProcess()
{
    int64_t startScheduleIdx = 0;
    if (vecIdx_ > 0) {
        startScheduleIdx = coreOffsetsGm_.GetValue(vecIdx_ - 1);    // offsets[vecIdx-1]
    }
    int64_t endScheduleIdx = coreOffsetsGm_.GetValue(vecIdx_);      // offsets[vecIdx]
    for (int64_t s = startScheduleIdx; s < endScheduleIdx; ++s) {
        int64_t tileIdx = scheduleGm_.GetValue(s);                  // schedule[s] → tile
        Process(tileIdx);
    }
    Finish();
}

// -----------------------------------------------------------------------------
// Finish — 末事件令牌消化（MTE3_MTE2 配对平衡：消化末 tile CopyOut 补充的令牌，
// 或空载核 Init 预置的令牌；REF_SAMPLE 同款收尾纪律）。
// -----------------------------------------------------------------------------
__aicore__ inline void CalcRenderFwdClipLastidsKernel::Finish()
{
    WaitFlag<HardEvent::MTE3_MTE2>(mte3Mte2Id_);
}

// -----------------------------------------------------------------------------
// Process — 逐 tile 阶段化流水线（Kernel.md §5；SIMD 侧编排，像素数学在 VF）：
//   阶段 0 段界与 tile 平面视图 → 阶段 1 载像素坐标（MTE2）→ 阶段 2 清零累加器
//   → 阶段 3 空 tile 快速路径（P1）→ 阶段 4 批循环（P2/P3，256 高斯/批；4a 批参数
//   下发 / 4b 批式协同载入 / 4c VF_CALL / 4d tile-done 判定 P4）→ 阶段 5 五路写回。
// -----------------------------------------------------------------------------
__aicore__ inline void CalcRenderFwdClipLastidsKernel::Process(int64_t tileIdx)
{
    // ===== 阶段 0：段界与 tile 平面视图（Task 30：段界防御裁剪至 [0, nIds_]，
    //       对齐 golden.py:74-76 segStart/segEnd = clip(end[k-1]/end[k], 0, nIds)、
    //       segLen = max(segEnd−segStart, 0)——合同外 offsets（end 越过 gsIds 末尾 /
    //       负值 / 非单调）不再越界读 gsIds：裁剪后批循环区间 [prevOffset,
    //       currOffset) ⊆ [0, nIds_] 恒界内；segLen ≤ 0 走 P1 空 tile 快速路径）=====
    int64_t prevOffset = (tileIdx > 0) ? tileOffsetsGm_.GetValue(tileIdx - 1) : 0;  // end[k-1]（end[-1]=0）
    int64_t currOffset = tileOffsetsGm_.GetValue(tileIdx);                          // end[k]
    if (prevOffset < 0) { prevOffset = 0; }
    if (prevOffset > nIds_) { prevOffset = nIds_; }
    if (currOffset < 0) { currOffset = 0; }
    if (currOffset > nIds_) { currOffset = nIds_; }
    GlobalTensor<float> tileCoordsGm  = tileCoordsGm_[tileIdx * 2 * nPixel_];
    GlobalTensor<float> tileColorRGm = colorRGm_[tileIdx * nPixel_];   // tile 行平面视图（三平面 + 四路）
    GlobalTensor<float> tileColorGGm = colorGGm_[tileIdx * nPixel_];
    GlobalTensor<float> tileColorBGm = colorBGm_[tileIdx * nPixel_];
    GlobalTensor<float> tileDepthGm  = depthGm_[tileIdx * nPixel_];
    GlobalTensor<float> tileLastCumsumGm = lastCumsumGm_[tileIdx * nPixel_];
    GlobalTensor<float> tileErrorGm  = errorGm_[tileIdx * nPixel_];
    GlobalTensor<int32_t> tileLastIdsGm = lastIdsGm_[tileIdx * nPixel_];

    // ===== 阶段 1：载像素坐标（MTE2；上一 tile 写回完成后再启动，单 buffer 跨 tile
    //       串行点 §6——Wait 消费 Init 预置令牌（首 tile）或上一 tile CopyOut 末 Set）=====
    WaitFlag<HardEvent::MTE3_MTE2>(mte3Mte2Id_);
    DataCopy(tileCoordX_, tileCoordsGm[0], calPixel_);           // X 平面 [:, 0, :]（4KB 定长，32B 倍数）
    DataCopy(tileCoordY_, tileCoordsGm[nPixel_], calPixel_);     // Y 平面 [:, 1, :]

    // ===== 阶段 2：清零累加器（空 tile 快速路径 P1 的零值来源，T9；跨批持久化、
    //       tile 界复位，T4）=====
    SetFlag<HardEvent::MTE2_V>(mte2VId_);
    WaitFlag<HardEvent::MTE2_V>(mte2VId_);
    Duplicate(pixOutR_, (float)0.0, (int32_t)calPixel_);
    Duplicate(pixOutG_, (float)0.0, (int32_t)calPixel_);
    Duplicate(pixOutB_, (float)0.0, (int32_t)calPixel_);
    Duplicate(pixOutD_, (float)0.0, (int32_t)calPixel_);
    Duplicate(lastCumsum_, (float)0.0, (int32_t)calPixel_);
    Duplicate(error_, (float)0.0, (int32_t)calPixel_);
    Duplicate(lastIds_, (int32_t)0, (int32_t)calPixel_);         // 无贡献像素为 0（Interface.md）
    Duplicate(doneMaskUb_, (uint8_t)0, (int32_t)calPixel_);      // per-pixel done 复位（批界不复位）

    // ===== 阶段 3：空 tile 快速路径（P1）——不进高斯循环，直接写回全零
    //       （Task 30：判据从 prevOffset == currOffset 放宽为 segLen ≤ 0，即
    //       currOffset ≤ prevOffset——对齐 golden segLen = max(segEnd−segStart, 0)，
    //       合同外非单调 end 的倒挂段亦为空 tile）=====
    if (currOffset <= prevOffset) {
        SetFlag<HardEvent::V_MTE3>(vMte3Id_);
        WaitFlag<HardEvent::V_MTE3>(vMte3Id_);
        CopyOutTile(tileColorRGm, tileColorGGm, tileColorBGm, tileDepthGm,
                    tileLastCumsumGm, tileErrorGm, tileLastIdsGm);
        return;
    }

    // ===== 阶段 4：批循环（P2 单批 / P3 多批；256 高斯/批，KERNEL_GS_PER_BATCH）=====
    tileDoneUb_.SetValue(0, (uint8_t)0);             // tile-done 复位（S 写 → V 读，4c 前 S_V 保证）
    int64_t i = prevOffset;
    while (i < currOffset) {
        int64_t batchSize = (currOffset - i < (int64_t)KERNEL_GS_PER_BATCH)
                            ? (currOffset - i) : (int64_t)KERNEL_GS_PER_BATCH;  // 批尾可 < 256
        // 4a. 标量参数下发（5×int64，VF 内首读，§9.2；S 写 → VF 读，4c 前 S_V 保证）
        //     [0] 必须是“本批”在排序列表中的起始位置 i（运行游标），而非 tile 起始
        //     prevOffset：VF 侧 lastIds 记 posBase+b（全局排序位置），bwd 用
        //     batchStart+t（同为排序位置）判界，二者必须同系。
        scalarParamsUb_.SetValue(0, i);
        scalarParamsUb_.SetValue(1, currOffset);
        scalarParamsUb_.SetValue(2, nPixel_);
        scalarParamsUb_.SetValue(3, calPixel_);
        scalarParamsUb_.SetValue(4, batchSize);
        // 4b. 批式协同载入 gsIds 段 + gs 属性段（T12：消除 VF 线程对 GM 的随机读，§7）；
        //     gsIds 批段走 DataCopyPad（Ext 参数）：批尾 batchSize×8B 非 32B 倍数时由
        //     Pad 自动处理（API.md §3 回改 F2），VF 只读 [0, batchSize) 无害
        DataCopyExtParams gsIdsCopyParams{1, static_cast<uint32_t>(batchSize * sizeof(int64_t)), 0, 0, 0};
        DataCopyPadExtParams<int64_t> gsIdsPadParams{false, 0, 0, 0};
        DataCopyPad(gsIdsBatchUb_, gsIdsGm_[i], gsIdsCopyParams, gsIdsPadParams);  // MTE2：连续 int64 段
        SetFlag<HardEvent::MTE2_S>(mte2SId_);
        WaitFlag<HardEvent::MTE2_S>(mte2SId_);       // MTE2 → S：gsIds 段可被 Scalar 读
        for (int64_t b = 0; b < batchSize; b++) {    // Scalar 逐高斯发属性段搬运
            const int64_t gsId = gsIdsBatchUb_.GetValue((uint32_t)b);
            // Task 30：gid 域防御对齐 golden.py:98-99 gsPad[clip(gid, 0, nGauss)]——
            // lastIds 记原始 gsId（VF 侧 gsIdsBatch 直读，含越界原值），属性行按
            // 裁剪值取：gid ∈ [0, nGauss) → 行 gid；gid < 0 且 nGauss > 0 → 行 0
            // （clip 下界）；其余（gid ≥ nGauss，含 nGauss = 0 全体与 gid < 0 且
            // nGauss = 0——clip 至 pad 槽位 nGauss）→ pad 零行：本槽位 Duplicate
            // 清零、不读 GM（零行 opacity = 0 ⇒ alpha = 0 < 1/255，VF clip-2 微
            // 贡献跳过、零状态更新，与 golden gsPad[nGauss] 零行语义逐位一致）
            int64_t attrRow = gsId;
            if (gsId < 0) {
                attrRow = (nGauss_ > 0) ? 0 : -1;
            } else if (gsId >= nGauss_) {
                attrRow = -1;
            }
            if (attrRow < 0) {
                // pad 零行（16 float 槽全清零；V 流水指令，发射序先于本批 VF 体）
                Duplicate(gsAttrBatchUb_[(uint32_t)(b * GS_ATTR_STRIDE)],
                          (float)0.0, (int32_t)GS_ATTR_STRIDE);
            } else {
                const int64_t attrRowOffset = attrRow * (int64_t)NUM_GS_ATTRIBUTES;
                if (attrRow + 1 == nGauss_) {
                    // gs 末行仅 40B（10 float）有效：64B 定长段会越读 gs 张量末尾 24B
                    // （真实读越界，oom 插桩按分配边界校验 addr+blockLen 检出 → trap 286；
                    // 越读值仅落入 VF 不读的 [10,16) padding 列，数值无影响）。末行走
                    // DataCopyPad 40B（非 32B 倍数由 Pad 处理，与上 gsIds 批段同款，
                    // API.md §3 F2），addr+40B 恰等于 gs 末尾、界内通过
                    DataCopyExtParams attrTailCopyParams{
                        1, static_cast<uint32_t>(NUM_GS_ATTRIBUTES * sizeof(float)), 0, 0, 0};
                    DataCopyPadExtParams<float> attrTailPadParams{false, 0, 0, 0};
                    DataCopyPad(gsAttrBatchUb_[(uint32_t)(b * GS_ATTR_STRIDE)],
                                gsGm_[attrRowOffset], attrTailCopyParams, attrTailPadParams);
                } else {
                    DataCopy(gsAttrBatchUb_[(uint32_t)(b * GS_ATTR_STRIDE)],
                             gsGm_[attrRowOffset], (int64_t)MIN_DATACOPY_LEN);  // 64B 定长对齐段
                }
            }
        }
        // 4c. VF 启动（批区/坐标全部 MTE2 写完成 + S 写可见后发射，§6）：
        //     1024 线程 × 1 像素/线程（§9.1/§9.2/§9.3）。asc_vf_call<F>(dim3, args...)
        //     为 arch35 SIMT 启动 API（simt_api/common_functions.h，与
        //     AscendC::Simt::VF_CALL 同体：两者均为 cce::async_invoke<funcPtr> 的
        //     薄包装、函数体逐行一致），此处按一致性门控（ops-impl-review
        //     vf_reachability_check）认可的 SIMT VF 调用拼写发射
        SetFlag<HardEvent::MTE2_V>(mte2VId_);
        WaitFlag<HardEvent::MTE2_V>(mte2VId_);
        SetFlag<HardEvent::S_V>(sVId_);
        WaitFlag<HardEvent::S_V>(sVId_);
        // 14 个 UB 区段 __ubuf__ 指针（T5，LocalTensor → GetPhyAddr 转换；tileSync =
        // tile-done/warp 归约联合区段，warpReduceUb_ 紧邻 tileDoneUb_ 32B 槽之后）
        __ubuf__ float*    tileCoordXPtr   = (__ubuf__ float*)tileCoordX_.GetPhyAddr();
        __ubuf__ float*    tileCoordYPtr   = (__ubuf__ float*)tileCoordY_.GetPhyAddr();
        __ubuf__ float*    pixOutRPtr      = (__ubuf__ float*)pixOutR_.GetPhyAddr();
        __ubuf__ float*    pixOutGPtr      = (__ubuf__ float*)pixOutG_.GetPhyAddr();
        __ubuf__ float*    pixOutBPtr      = (__ubuf__ float*)pixOutB_.GetPhyAddr();
        __ubuf__ float*    pixOutDPtr      = (__ubuf__ float*)pixOutD_.GetPhyAddr();
        __ubuf__ float*    lastCumsumPtr   = (__ubuf__ float*)lastCumsum_.GetPhyAddr();
        __ubuf__ float*    errorPtr        = (__ubuf__ float*)error_.GetPhyAddr();
        __ubuf__ int32_t*  lastIdsPtr      = (__ubuf__ int32_t*)lastIds_.GetPhyAddr();
        __ubuf__ uint8_t*  doneMaskPtr     = (__ubuf__ uint8_t*)doneMaskUb_.GetPhyAddr();
        __ubuf__ int64_t*  scalarParamsPtr = (__ubuf__ int64_t*)scalarParamsUb_.GetPhyAddr();
        __ubuf__ float*    gsAttrBatchPtr  = (__ubuf__ float*)gsAttrBatchUb_.GetPhyAddr();
        __ubuf__ int64_t*  gsIdsBatchPtr   = (__ubuf__ int64_t*)gsIdsBatchUb_.GetPhyAddr();
        __ubuf__ uint8_t*  tileSyncPtr     = (__ubuf__ uint8_t*)tileDoneUb_.GetPhyAddr();
        asc_vf_call<FwdSimtCompute>(AscendC::Simt::Dim3(THREAD_NUM),
            tileCoordXPtr, tileCoordYPtr, pixOutRPtr, pixOutGPtr, pixOutBPtr, pixOutDPtr,
            lastCumsumPtr, errorPtr, lastIdsPtr, doneMaskPtr, scalarParamsPtr,
            gsAttrBatchPtr, gsIdsBatchPtr, tileSyncPtr);
        // V → S：Scalar 等 VF 完成（下一批 4a 覆写 scalarParams / 4d 读 tileDone 的前置）
        SetFlag<HardEvent::V_S>(vSId_);
        WaitFlag<HardEvent::V_S>(vSId_);
        i += batchSize;
        // 4d. 批尾 tile-done 判定（P4，T3）：全 1024 像素 done ⇒ tile 早停（数学结果不变）
        if (i < currOffset && tileDoneUb_.GetValue(0) == 1) {
            break;
        }
    }

    // ===== 阶段 5：五路写回（V → MTE3，§8）=====
    // error 输出 = Kahan 补偿项数学真值 0（spec.yaml math_semantics.formula：
    // error = np.zeros(...)，精确算术下补偿项恒零；golden.py:133 逐像素与 0 比对，
    // stat_rel_err 对零 golden 的有效容差仅 ~1e-10，fp32 残差不可暴露）。error_ 区段
    // 在批循环内承载 Kahan 补偿跨批状态（T4 批界不重置，VF 末尾写回、下批首读），
    // tile 计算完成后状态不再消费（下一 tile 阶段 2 全量复位），写回前清零输出
    Duplicate(error_, (float)0.0, (int32_t)calPixel_);
    SetFlag<HardEvent::V_MTE3>(vMte3Id_);
    WaitFlag<HardEvent::V_MTE3>(vMte3Id_);
    CopyOutTile(tileColorRGm, tileColorGGm, tileColorBGm, tileDepthGm,
                tileLastCumsumGm, tileErrorGm, tileLastIdsGm);
}

// -----------------------------------------------------------------------------
// CopyOutTile — 五路 7 段定长写回（Kernel.md §8；每 tile 一次，V_MTE3 事件后）：
// 恒单路径 dense（各 1024 元素 = 4KB，32B 倍数），无条件全量覆写（空 tile P1 全零行、
// tile 早停 P4 不跳过写回）；末尾 SetFlag<MTE3_MTE2> 为跨 tile 串行点补充令牌
// （写回完成才允许下一 tile 覆写单 buffer，§6）。
// -----------------------------------------------------------------------------
__aicore__ inline void CalcRenderFwdClipLastidsKernel::CopyOutTile(GlobalTensor<float>& colorRGm,
                                                            GlobalTensor<float>& colorGGm,
                                                            GlobalTensor<float>& colorBGm,
                                                            GlobalTensor<float>& depthGm,
                                                            GlobalTensor<float>& lastCumsumGm,
                                                            GlobalTensor<float>& errorGm,
                                                            GlobalTensor<int32_t>& lastIdsGm)
{
    DataCopy(colorRGm[0],     pixOutR_,    calPixel_);   // R 平面 tile 行（三平面分存单张量）
    DataCopy(colorGGm[0],     pixOutG_,    calPixel_);   // G 平面
    DataCopy(colorBGm[0],     pixOutB_,    calPixel_);   // B 平面
    DataCopy(depthGm[0],      pixOutD_,    calPixel_);
    DataCopy(lastCumsumGm[0], lastCumsum_, calPixel_);   // ln 域累积透射率 s_K
    DataCopy(errorGm[0],      error_,     calPixel_);    // Kahan 补偿项（数学真值 0，阶段 5 写回前清零）
    DataCopy(lastIdsGm[0],    lastIds_,   calPixel_);    // int32 逐像素最后贡献索引（T2）
    SetFlag<HardEvent::MTE3_MTE2>(mte3Mte2Id_);          // 跨 tile 串行点（§6）
}

// -----------------------------------------------------------------------------
// Kernel 入口（Kernel.md §1；11 形参与 Interface.md「kernel 入口形参」及 OpDef
// 签名顺序一致；SIMT 单分支无模板分发，入口为单实例直通；workspace = 0 占位形参
// 不消费）。TPipe 必须在 kernel 入口声明、经指针传入（T6：UB bus error 264 修复）。
// -----------------------------------------------------------------------------
extern "C" __global__ __aicore__ void calc_render_fwd_clip_lastids(
    GM_ADDR gs, GM_ADDR tileCoords, GM_ADDR offsets, GM_ADDR gsIds,
    GM_ADDR color, GM_ADDR depth, GM_ADDR lastCumsum, GM_ADDR error,
    GM_ADDR lastIds, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);   // 纯 AIV（SIMT asc_simt/VF），无 Cube
    SetAtomicNone();                                  // 无原子写（核间 tile 互斥分区）
    TPipe pipe;                                       // T6：TPipe 必须在 kernel 入口声明
    REGISTER_TILING_DEFAULT(CalcRenderFwdClipLastidsTilingData);
    GET_TILING_DATA_WITH_STRUCT(CalcRenderFwdClipLastidsTilingData, tilingData, tiling);
    if (tilingData.tileNum == 0) {                    // P0：空 tile 列表占位返回（T9）
        return;
    }
    if (tilingData.ubSize - DCACHE_RESERVED < UB_LAYOUT_BYTES) {   // 容量校验兜底（非契约平台，T4）
        return;
    }
    CalcRenderFwdClipLastidsKernel op;
    op.Init(gs, tileCoords, offsets, gsIds,
            color, depth, lastCumsum, error, lastIds, &tilingData, &pipe);
    op.loopProcess();
}
