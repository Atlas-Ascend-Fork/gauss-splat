/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {
class CalcRenderFwdClipLastids : public OpDef {
public:
    explicit CalcRenderFwdClipLastids(const char* name) : OpDef(name)
    {
        // —— Input 声明：ParamType + DataType + Format + UnknownShapeFormat（单一 dtype 组合，各 1 项）——
        this->Input("gs")            // 高斯属性表 [NGauss, 10]
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("tileCoords")    // 逐 tile 像素坐标 [TileNum, 2, 1024]
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("offsets")       // 核间调度表 [vecNum + 2*TileNum]（int64，三段布局）
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gsIds")         // 深度排序高斯全局 id [TotalGauss]（int64）
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // —— Output 声明：顺序 = aclnn/OpDef 签名顺序（color, depth, lastCumsum, error, lastIds）——
        this->Output("color")        // R/G/B 三平面 [3, TileNum, 1024]
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("depth")        // [TileNum, 1024]
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("lastCumsum")   // ln 域累积透射率 [TileNum, 1024]
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("error")        // Kahan 误差 [TileNum, 1024]
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("lastIds")      // 逐像素最后贡献高斯全局索引 [TileNum, 1024]（int32，T2/T8）
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // —— Attr 声明：无（proto.md：alpha 上界 0.999、ALPHA_THRESHOLD=1/255、DONE_THRESHOLD=1e-4、nPixel=1024 均为编译期固定常量，T1/T3）——

        // —— AICore 配置：仅 ascend950 ——
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)     // 单一 shape 无关 kernel（shape 经 TilingData 运行时下发）
            .DynamicFormatFlag(false)                   // 固定 ND，无动态格式转换
            .DynamicRankSupportFlag(false)              // 各输入 rank 契约固定（2/3/1/1）
            .DynamicShapeSupportFlag(true)              // TileNum/NGauss/TotalGauss/vecNum 为运行时可变维
            .NeedCheckSupportFlag(false)                // 跳过框架 support check，由 aclnn L2 自检
            .PrecisionReduceFlag(false)                 // 禁止降精度：fp32 保序 Kahan 数值契约（T10）
            .ExtendCfgInfo("opFile.value", "calc_render_fwd_clip_last_ids_apt"); // kernel 入口 op_kernel/arch35/calc_render_fwd_clip_last_ids_apt.cpp
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(CalcRenderFwdClipLastids);
} // namespace ops

// [REF_SAMPLE] Original sample retained for reference only.
// [REF_SAMPLE] /**
// [REF_SAMPLE]  * Copyright (c) 2026 Huawei Technologies Co., Ltd.
// [REF_SAMPLE]  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// [REF_SAMPLE]  * CANN Open Software License Agreement Version 2.0 (the "License").
// [REF_SAMPLE]  * Please refer to the License for details. You may not use this file except in compliance with the License.
// [REF_SAMPLE]  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// [REF_SAMPLE]  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// [REF_SAMPLE]  * See LICENSE in the root of the software repository for the full text of the License.
// [REF_SAMPLE]  */
// [REF_SAMPLE] 
// [REF_SAMPLE] /*!
// [REF_SAMPLE]  * \file calc_render_fwd_clip_last_ids_def.cpp
// [REF_SAMPLE]  * \brief CalcRenderFwdClipLastids 算子定义，声明输入输出和算子配置
// [REF_SAMPLE]  */
// [REF_SAMPLE] #include "register/op_def_registry.h"
// [REF_SAMPLE] 
// [REF_SAMPLE] namespace ops {
// [REF_SAMPLE] class CalcRenderFwdClipLastids : public OpDef {
// [REF_SAMPLE] public:
// [REF_SAMPLE]     explicit CalcRenderFwdClipLastids(const char* name) : OpDef(name)
// [REF_SAMPLE]     {
// [REF_SAMPLE]         this->Input("x1")                                       // 输入x1定义
// [REF_SAMPLE]             .ParamType(REQUIRED)                                // 必选输入
// [REF_SAMPLE]             .DataType({ge::DT_FLOAT, ge::DT_INT32})             // 支持数据类型
// [REF_SAMPLE]             .Format({ge::FORMAT_ND, ge::FORMAT_ND})             // 支持format格式
// [REF_SAMPLE]             .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}) // 未确定大小shape对应format格式
// [REF_SAMPLE]             .AutoContiguous();                                  // 内存自动连续化
// [REF_SAMPLE]         this->Input("x2")                                       // 输入x2定义
// [REF_SAMPLE]             .ParamType(REQUIRED)
// [REF_SAMPLE]             .DataType({ge::DT_FLOAT, ge::DT_INT32})
// [REF_SAMPLE]             .Format({ge::FORMAT_ND, ge::FORMAT_ND})
// [REF_SAMPLE]             .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
// [REF_SAMPLE]             .AutoContiguous();
// [REF_SAMPLE]         this->Output("y") // 输出y定义
// [REF_SAMPLE]             .ParamType(REQUIRED)
// [REF_SAMPLE]             .DataType({ge::DT_FLOAT, ge::DT_INT32})
// [REF_SAMPLE]             .Format({ge::FORMAT_ND, ge::FORMAT_ND})
// [REF_SAMPLE]             .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
// [REF_SAMPLE]             .AutoContiguous();
// [REF_SAMPLE] 
// [REF_SAMPLE]         OpAICoreConfig aicoreConfig910B;
// [REF_SAMPLE]         aicoreConfig910B.DynamicCompileStaticFlag(true)
// [REF_SAMPLE]             .DynamicFormatFlag(false)
// [REF_SAMPLE]             .DynamicRankSupportFlag(true)
// [REF_SAMPLE]             .DynamicShapeSupportFlag(true)
// [REF_SAMPLE]             .NeedCheckSupportFlag(false)
// [REF_SAMPLE]             .PrecisionReduceFlag(true);
// [REF_SAMPLE]         this->AICore().AddConfig("ascend910b", aicoreConfig910B);
// [REF_SAMPLE] 
// [REF_SAMPLE]         OpAICoreConfig aicoreConfig950;
// [REF_SAMPLE]         aicoreConfig950.DynamicCompileStaticFlag(true)
// [REF_SAMPLE]             .DynamicFormatFlag(false)
// [REF_SAMPLE]             .DynamicRankSupportFlag(true)
// [REF_SAMPLE]             .DynamicShapeSupportFlag(true)
// [REF_SAMPLE]             .NeedCheckSupportFlag(false)
// [REF_SAMPLE]             .PrecisionReduceFlag(true);
// [REF_SAMPLE]         this->AICore().AddConfig("ascend950", aicoreConfig950);
// [REF_SAMPLE]     }
// [REF_SAMPLE] };
// [REF_SAMPLE] OP_ADD(CalcRenderFwdClipLastids); // 添加算子信息库
// [REF_SAMPLE] } // namespace ops
