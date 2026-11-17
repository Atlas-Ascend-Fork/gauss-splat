/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_color_exposure_bwd.h"
#include "test_utils.h"

using namespace std;

namespace {

const int64_t kImageWidth = 320;
const int64_t kImageHeight = 240;
const int kBoundaryCount = 7;
const float kBoundaryXy[kBoundaryCount][2] = {
    {0.0f, 0.0f},
    {320.0f, 240.0f},
    {319.0f, 239.0f},
    {-4.0f, -2.0f},
    {323.0f, 241.0f},
    {320.0f, 0.0f},
    {0.0f, 240.0f},
};
const float kBoundaryColors[kBoundaryCount][3] = {
    {0.0f, 0.0f, 0.0f},
    {2.0f, 2.0f, 2.0f},
    {-2.0f, -2.0f, -2.0f},
    {0.5f, -0.3f, 0.2f},
    {0.1f, 0.2f, 0.3f},
    {-0.5f, 0.5f, 0.25f},
    {0.299f, 0.587f, 0.114f},
};

float NextUniform(uint32_t& state)
{
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state >> 8) / 16777216.0f;
}

float Clamp01(float v)
{
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

void GenRandomAffine(uint32_t& rng, float* e)
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            e[4 * i + j] = (NextUniform(rng) * 2.0f - 1.0f) * 0.1f;
        }
        e[4 * i + 3] = (NextUniform(rng) * 2.0f - 1.0f) * 0.1f;
    }
}

struct GridCoord {
    int32_t ix;
    int32_t iy;
    int32_t iz;
    float wx;
    float wy;
    float wz;
};

GridCoord MapToGrid(float x, float y, float grey, int32_t gx, int32_t gy, int32_t gz,
                    float invWidth, float invHeight)
{
    float fx = Clamp01(x * invWidth) * static_cast<float>(gx - 1);
    float fy = Clamp01(y * invHeight) * static_cast<float>(gy - 1);
    float fz = Clamp01(grey) * static_cast<float>(gz - 1);
    GridCoord c;
    c.ix = static_cast<int32_t>(fx);
    c.wx = fx - static_cast<float>(c.ix);
    c.iy = static_cast<int32_t>(fy);
    c.wy = fy - static_cast<float>(c.iy);
    c.iz = static_cast<int32_t>(fz);
    c.wz = fz - static_cast<float>(c.iz);
    if (c.ix == gx - 1) {
        c.ix -= 1;
        c.wx = 1.0f;
    }
    if (c.iy == gy - 1) {
        c.iy -= 1;
        c.wy = 1.0f;
    }
    if (c.iz == gz - 1) {
        c.iz -= 1;
        c.wz = 1.0f;
    }
    return c;
}

void InterpolateAffine(const float* grid, const GridCoord& c, int32_t gh, int32_t gd, float* out)
{
    const float wxl = 1.0f - c.wx;
    const float wxr = c.wx;
    const float wyl = 1.0f - c.wy;
    const float wyr = c.wy;
    const float wzl = 1.0f - c.wz;
    const float wzr = c.wz;
    const float w[8] = {
        wxl * wyl * wzl,
        wxl * wyl * wzr,
        wxl * wyr * wzl,
        wxl * wyr * wzr,
        wxr * wyl * wzl,
        wxr * wyl * wzr,
        wxr * wyr * wzl,
        wxr * wyr * wzr,
    };
    const int32_t id[8] = {
        (c.ix * gh + c.iy) * gd + c.iz,
        (c.ix * gh + c.iy) * gd + c.iz + 1,
        (c.ix * gh + c.iy + 1) * gd + c.iz,
        (c.ix * gh + c.iy + 1) * gd + c.iz + 1,
        ((c.ix + 1) * gh + c.iy) * gd + c.iz,
        ((c.ix + 1) * gh + c.iy) * gd + c.iz + 1,
        ((c.ix + 1) * gh + c.iy + 1) * gd + c.iz,
        ((c.ix + 1) * gh + c.iy + 1) * gd + c.iz + 1,
    };
    for (int j = 0; j < 12; j++) {
        float acc = 0.0f;
        for (int k = 0; k < 8; k++) {
            acc += w[k] * grid[id[k] * 12 + j];
        }
        out[j] = acc;
    }
}

float ApplyAffineRow(const float* e, int row, float r, float g, float b)
{
    const int base = 4 * row;
    return ((e[base] * r + e[base + 1] * g) + e[base + 2] * b) + e[base + 3];
}

float ApplyVjpCol(const float* e, int col, float vr, float vg, float vb)
{
    // v_c = A^T * v: v_{c_col} = Σ_i A[i][col] * v_i, A[i][j] = e[4i+j]
    return e[0 * 4 + col] * vr + e[1 * 4 + col] * vg + e[2 * 4 + col] * vb;
}

struct VerifyReport {
    bool pass;
    double maxAbsDiff;
    int64_t mismatchCount;
    int64_t firstBadIndex;
};

VerifyReport VerifyClose(const vector<float>& actual, const vector<float>& expected,
                         double rtol, double atol)
{
    VerifyReport rep;
    rep.pass = true;
    rep.maxAbsDiff = 0.0;
    rep.mismatchCount = 0;
    rep.firstBadIndex = -1;
    const int64_t n = static_cast<int64_t>(expected.size());
    for (int64_t i = 0; i < n; i++) {
        const double diff = fabs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        if (diff > rep.maxAbsDiff) {
            rep.maxAbsDiff = diff;
        }
        const double tol = atol + rtol * fabs(static_cast<double>(expected[i]));
        if (diff > tol) {
            rep.mismatchCount++;
            if (rep.firstBadIndex < 0) {
                rep.firstBadIndex = i;
            }
        }
    }
    rep.pass = rep.mismatchCount == 0;
    return rep;
}

struct BwdCase {
    const char* name;
    int64_t batch;
    int64_t gauss;
    int32_t gridW;
    int32_t gridH;
    int32_t gridD;
    bool clamp;
    bool calcExpGrad;
    uint32_t seed;
};

struct BwdTensors {
    void* colorsAddr = nullptr;
    void* vColorsInAddr = nullptr;
    void* exposureAddr = nullptr;
    void* indexAddr = nullptr;
    void* weightAddr = nullptr;
    void* vColorsOutAddr = nullptr;
    void* vExposureAddr = nullptr;
    aclTensor* colorsTensor = nullptr;
    aclTensor* vColorsInTensor = nullptr;
    aclTensor* exposureTensor = nullptr;
    aclTensor* indexTensor = nullptr;
    aclTensor* weightTensor = nullptr;
    aclTensor* vColorsOutTensor = nullptr;
    aclTensor* vExposureTensor = nullptr;
    void* workspaceAddr = nullptr;
};

void DestroyBwdTensors(BwdTensors& t)
{
    if (t.colorsTensor) aclDestroyTensor(t.colorsTensor);
    if (t.vColorsInTensor) aclDestroyTensor(t.vColorsInTensor);
    if (t.exposureTensor) aclDestroyTensor(t.exposureTensor);
    if (t.indexTensor) aclDestroyTensor(t.indexTensor);
    if (t.weightTensor) aclDestroyTensor(t.weightTensor);
    if (t.vColorsOutTensor) aclDestroyTensor(t.vColorsOutTensor);
    if (t.vExposureTensor) aclDestroyTensor(t.vExposureTensor);
    if (t.colorsAddr) aclrtFree(t.colorsAddr);
    if (t.vColorsInAddr) aclrtFree(t.vColorsInAddr);
    if (t.exposureAddr) aclrtFree(t.exposureAddr);
    if (t.indexAddr) aclrtFree(t.indexAddr);
    if (t.weightAddr) aclrtFree(t.weightAddr);
    if (t.vColorsOutAddr) aclrtFree(t.vColorsOutAddr);
    if (t.vExposureAddr) aclrtFree(t.vExposureAddr);
    if (t.workspaceAddr) aclrtFree(t.workspaceAddr);
}

bool RunBwdCase(const BwdCase& cs, aclrtStream stream)
{
    const int64_t b = cs.batch;
    const int64_t n = cs.gauss;
    const int64_t bcn = b * n;
    const bool useGrid = cs.gridW > 0;
    const int64_t gridCells = useGrid ?
        static_cast<int64_t>(cs.gridW) * cs.gridH * cs.gridD : 1;
    const int64_t expNumel = gridCells * 12;

    vector<float> colorsHost(bcn * 3, 0.0f);
    vector<float> vColorsInHost(bcn * 3, 0.0f);
    vector<float> means2dHost(bcn * 2, 0.0f);
    vector<float> exposureHost(expNumel, 0.0f);
    uint32_t rng = cs.seed;

    for (int64_t i = 0; i < bcn * 3; i++) {
        colorsHost[i] = (NextUniform(rng) * 2.0f - 1.0f) * 0.5f;
        vColorsInHost[i] = (NextUniform(rng) * 2.0f - 1.0f) * 0.5f;
    }
    for (int64_t i = 0; i < bcn; i++) {
        means2dHost[i] = (NextUniform(rng) * 1.2f - 0.1f) * static_cast<float>(kImageWidth);
        means2dHost[bcn + i] = (NextUniform(rng) * 1.2f - 0.1f) * static_cast<float>(kImageHeight);
    }
    for (int64_t c = 0; c < gridCells; c++) {
        GenRandomAffine(rng, exposureHost.data() + c * 12);
    }

    const int64_t k = min(static_cast<int64_t>(kBoundaryCount), n);
    for (int64_t bi = 0; bi < b; bi++) {
        for (int64_t j = 0; j < k; j++) {
            for (int c = 0; c < 3; c++) {
                colorsHost[(bi * 3 + c) * n + j] = kBoundaryColors[j][c];
            }
            for (int c = 0; c < 2; c++) {
                means2dHost[(bi * 2 + c) * n + j] = kBoundaryXy[j][c];
            }
        }
    }

    vector<int32_t> indexHost(useGrid ? bcn * 3 : 0, 0);
    vector<float> weightHost(useGrid ? bcn * 3 : 0.0f);
    const float invW = 1.0f / static_cast<float>(kImageWidth);
    const float invH = 1.0f / static_cast<float>(kImageHeight);
    if (useGrid) {
        for (int64_t bi = 0; bi < b; bi++) {
            for (int64_t j = 0; j < n; j++) {
                const float r = colorsHost[(bi * 3 + 0) * n + j];
                const float g = colorsHost[(bi * 3 + 1) * n + j];
                const float bl = colorsHost[(bi * 3 + 2) * n + j];
                const float x = means2dHost[(bi * 2 + 0) * n + j];
                const float y = means2dHost[(bi * 2 + 1) * n + j];
                const float grey = 0.299f * r + 0.587f * g + 0.114f * bl;
                const GridCoord c = MapToGrid(x, y, grey, cs.gridW, cs.gridH, cs.gridD, invW, invH);
                indexHost[(bi * 3 + 0) * n + j] = c.ix;
                indexHost[(bi * 3 + 1) * n + j] = c.iy;
                indexHost[(bi * 3 + 2) * n + j] = c.iz;
                weightHost[(bi * 3 + 0) * n + j] = c.wx;
                weightHost[(bi * 3 + 1) * n + j] = c.wy;
                weightHost[(bi * 3 + 2) * n + j] = c.wz;
            }
        }
    }

    vector<float> goldenVColors(bcn * 3, 0.0f);
    vector<double> goldenVExp(cs.calcExpGrad ? expNumel : 0, 0.0);
    for (int64_t bi = 0; bi < b; bi++) {
        for (int64_t j = 0; j < n; j++) {
            const float r = colorsHost[(bi * 3 + 0) * n + j];
            const float g = colorsHost[(bi * 3 + 1) * n + j];
            const float bl = colorsHost[(bi * 3 + 2) * n + j];
            float vr = vColorsInHost[(bi * 3 + 0) * n + j];
            float vg = vColorsInHost[(bi * 3 + 1) * n + j];
            float vb = vColorsInHost[(bi * 3 + 2) * n + j];
            float e[12];
            int32_t id[8];
            float w[8];
            if (useGrid) {
                const GridCoord c = {
                    indexHost[(bi * 3 + 0) * n + j],
                    indexHost[(bi * 3 + 1) * n + j],
                    indexHost[(bi * 3 + 2) * n + j],
                    weightHost[(bi * 3 + 0) * n + j],
                    weightHost[(bi * 3 + 1) * n + j],
                    weightHost[(bi * 3 + 2) * n + j]};
                InterpolateAffine(exposureHost.data(), c, cs.gridH, cs.gridD, e);
                if (cs.calcExpGrad) {
                    const float wxl = 1.0f - c.wx;
                    const float wxr = c.wx;
                    const float wyl = 1.0f - c.wy;
                    const float wyr = c.wy;
                    const float wzl = 1.0f - c.wz;
                    const float wzr = c.wz;
                    w[0] = wxl * wyl * wzl;
                    w[1] = wxl * wyl * wzr;
                    w[2] = wxl * wyr * wzl;
                    w[3] = wxl * wyr * wzr;
                    w[4] = wxr * wyl * wzl;
                    w[5] = wxr * wyl * wzr;
                    w[6] = wxr * wyr * wzl;
                    w[7] = wxr * wyr * wzr;
                    id[0] = (c.ix * cs.gridH + c.iy) * cs.gridD + c.iz;
                    id[1] = id[0] + 1;
                    id[2] = (c.ix * cs.gridH + c.iy + 1) * cs.gridD + c.iz;
                    id[3] = id[2] + 1;
                    id[4] = ((c.ix + 1) * cs.gridH + c.iy) * cs.gridD + c.iz;
                    id[5] = id[4] + 1;
                    id[6] = ((c.ix + 1) * cs.gridH + c.iy + 1) * cs.gridD + c.iz;
                    id[7] = id[6] + 1;
                }
            } else {
                for (int t = 0; t < 12; t++) {
                    e[t] = exposureHost[t];
                }
            }
            if (cs.clamp) {
                if (ApplyAffineRow(e, 0, r, g, bl) < 0.0f) {
                    vr = 0.0f;
                }
                if (ApplyAffineRow(e, 1, r, g, bl) < 0.0f) {
                    vg = 0.0f;
                }
                if (ApplyAffineRow(e, 2, r, g, bl) < 0.0f) {
                    vb = 0.0f;
                }
            }
            for (int col = 0; col < 3; col++) {
                goldenVColors[(bi * 3 + col) * n + j] = ApplyVjpCol(e, col, vr, vg, vb);
            }
            if (cs.calcExpGrad) {
                double grad[12];
                grad[0] = static_cast<double>(r) * vr;
                grad[1] = static_cast<double>(g) * vr;
                grad[2] = static_cast<double>(bl) * vr;
                grad[3] = vr;
                grad[4] = static_cast<double>(r) * vg;
                grad[5] = static_cast<double>(g) * vg;
                grad[6] = static_cast<double>(bl) * vg;
                grad[7] = vg;
                grad[8] = static_cast<double>(r) * vb;
                grad[9] = static_cast<double>(g) * vb;
                grad[10] = static_cast<double>(bl) * vb;
                grad[11] = vb;
                if (useGrid) {
                    for (int kk = 0; kk < 8; kk++) {
                        const int64_t base = static_cast<int64_t>(id[kk]) * 12;
                        for (int m = 0; m < 12; m++) {
                            goldenVExp[base + m] += static_cast<double>(w[kk]) * grad[m];
                        }
                    }
                } else {
                    for (int m = 0; m < 12; m++) {
                        goldenVExp[m] += grad[m];
                    }
                }
            }
        }
    }
    vector<float> goldenVExpF(goldenVExp.size(), 0.0f);
    for (size_t i = 0; i < goldenVExp.size(); i++) {
        goldenVExpF[i] = static_cast<float>(goldenVExp[i]);
    }

    BwdTensors t;
    vector<int64_t> colorsShape = {b, 3, n};
    vector<int64_t> expShape = useGrid ?
        vector<int64_t>{cs.gridW, cs.gridH, cs.gridD, 12} : vector<int64_t>{12};
    vector<float> zeroVColors(bcn * 3, 0.0f);
    vector<float> zeroVExp(expNumel, 0.0f);

    int ret = CreateAclTensor(colorsHost, colorsShape, &t.colorsAddr,
                              aclDataType::ACL_FLOAT, &t.colorsTensor);
    bool ok = CHECK_RET(ret == ACL_SUCCESS);
    if (!ok) {
        LOG_PRINT("CreateAclTensor colors failed. ERROR: %d\n", ret);
        DestroyBwdTensors(t);
        return false;
    }
    ret = CreateAclTensor(vColorsInHost, colorsShape, &t.vColorsInAddr,
                          aclDataType::ACL_FLOAT, &t.vColorsInTensor);
    ok = CHECK_RET(ret == ACL_SUCCESS);
    if (!ok) {
        LOG_PRINT("CreateAclTensor vColorsIn failed. ERROR: %d\n", ret);
        DestroyBwdTensors(t);
        return false;
    }
    ret = CreateAclTensor(exposureHost, expShape, &t.exposureAddr,
                          aclDataType::ACL_FLOAT, &t.exposureTensor);
    ok = CHECK_RET(ret == ACL_SUCCESS);
    if (!ok) {
        LOG_PRINT("CreateAclTensor exposure failed. ERROR: %d\n", ret);
        DestroyBwdTensors(t);
        return false;
    }
    if (useGrid) {
        ret = CreateAclTensor(indexHost, colorsShape, &t.indexAddr,
                              aclDataType::ACL_INT32, &t.indexTensor);
        ok = CHECK_RET(ret == ACL_SUCCESS);
        if (!ok) {
            LOG_PRINT("CreateAclTensor index failed. ERROR: %d\n", ret);
            DestroyBwdTensors(t);
            return false;
        }
        ret = CreateAclTensor(weightHost, colorsShape, &t.weightAddr,
                              aclDataType::ACL_FLOAT, &t.weightTensor);
        ok = CHECK_RET(ret == ACL_SUCCESS);
        if (!ok) {
            LOG_PRINT("CreateAclTensor weight failed. ERROR: %d\n", ret);
            DestroyBwdTensors(t);
            return false;
        }
    }
    ret = CreateAclTensor(zeroVColors, colorsShape, &t.vColorsOutAddr,
                          aclDataType::ACL_FLOAT, &t.vColorsOutTensor);
    ok = CHECK_RET(ret == ACL_SUCCESS);
    if (!ok) {
        LOG_PRINT("CreateAclTensor vColorsOut failed. ERROR: %d\n", ret);
        DestroyBwdTensors(t);
        return false;
    }
    ret = CreateAclTensor(zeroVExp, expShape, &t.vExposureAddr,
                          aclDataType::ACL_FLOAT, &t.vExposureTensor);
    ok = CHECK_RET(ret == ACL_SUCCESS);
    if (!ok) {
        LOG_PRINT("CreateAclTensor vExposure failed. ERROR: %d\n", ret);
        DestroyBwdTensors(t);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnColorExposureBwdGetWorkspaceSize(t.colorsTensor, t.vColorsInTensor, t.exposureTensor,
                                                t.indexTensor, t.weightTensor,
                                                cs.clamp ? 1 : 0, cs.calcExpGrad ? 1 : 0,
                                                t.vColorsOutTensor, t.vExposureTensor,
                                                &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnColorExposureBwdGetWorkspaceSize failed. ERROR: %d\n", ret);
        DestroyBwdTensors(t);
        return false;
    }
    if (workspaceSize > 0ULL) {
        ret = aclrtMalloc(&t.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            DestroyBwdTensors(t);
            return false;
        }
    }
    ret = aclnnColorExposureBwd(t.workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnColorExposureBwd failed. ERROR: %d\n", ret);
        DestroyBwdTensors(t);
        return false;
    }
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        DestroyBwdTensors(t);
        return false;
    }

    vector<float> actualVColors(bcn * 3, 0.0f);
    ret = aclrtMemcpy(actualVColors.data(), actualVColors.size() * sizeof(float), t.vColorsOutAddr,
                      actualVColors.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy vColorsOut to host failed. ERROR: %d\n", ret);
        DestroyBwdTensors(t);
        return false;
    }

    vector<float> actualVExp;
    if (cs.calcExpGrad) {
        actualVExp.resize(expNumel, 0.0f);
        ret = aclrtMemcpy(actualVExp.data(), actualVExp.size() * sizeof(float), t.vExposureAddr,
                          actualVExp.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("copy vExposure to host failed. ERROR: %d\n", ret);
            DestroyBwdTensors(t);
            return false;
        }
    }

    std::string expDesc = "[12]";
    if (useGrid) {
        expDesc = "[" + std::to_string(cs.gridW) + "," + std::to_string(cs.gridH) + "," +
                  std::to_string(cs.gridD) + ",12]";
    }

    const VerifyReport vColorsRep = VerifyClose(actualVColors, goldenVColors, 1e-5, 1e-5);
    bool pass = vColorsRep.pass;
    VerifyReport vExpRep;
    vExpRep.pass = true;
    vExpRep.maxAbsDiff = 0.0;
    vExpRep.mismatchCount = 0;
    vExpRep.firstBadIndex = -1;
    if (cs.calcExpGrad) {
        const double vExpRtol = 2e-4;
        const double vExpAtol = 1e-5 + 5e-8 * static_cast<double>(bcn);
        vExpRep = VerifyClose(actualVExp, goldenVExpF, vExpRtol, vExpAtol);
        pass = pass && vExpRep.pass;
    }

    DestroyBwdTensors(t);

    if (pass) {
        LOG_PRINT("[bwd:%s] B=%ld N=%ld exposure=%s clamp=%d calc=%d A=rand PASS vcolors_max_diff=%.3e",
                  cs.name, static_cast<long>(b), static_cast<long>(n), expDesc.c_str(),
                  cs.clamp ? 1 : 0, cs.calcExpGrad ? 1 : 0, vColorsRep.maxAbsDiff);
        if (cs.calcExpGrad) {
            LOG_PRINT(" vexp_max_diff=%.3e", vExpRep.maxAbsDiff);
        }
        LOG_PRINT("\n");
    } else {
        LOG_PRINT("[bwd:%s] B=%ld N=%ld exposure=%s clamp=%d calc=%d A=rand FAIL",
                  cs.name, static_cast<long>(b), static_cast<long>(n), expDesc.c_str(),
                  cs.clamp ? 1 : 0, cs.calcExpGrad ? 1 : 0);
        if (!vColorsRep.pass) {
            const int64_t i = vColorsRep.firstBadIndex;
            LOG_PRINT(" vcolors mismatch=%ld first@%ld act=%.7f exp=%.7f diff=%.3e",
                      static_cast<long>(vColorsRep.mismatchCount), static_cast<long>(i),
                      static_cast<double>(actualVColors[i]),
                      static_cast<double>(goldenVColors[i]),
                      fabs(static_cast<double>(actualVColors[i]) -
                           static_cast<double>(goldenVColors[i])));
        }
        if (cs.calcExpGrad && !vExpRep.pass) {
            const int64_t i = vExpRep.firstBadIndex;
            LOG_PRINT(" vexp mismatch=%ld first@%ld act=%.7f exp=%.7f diff=%.3e",
                      static_cast<long>(vExpRep.mismatchCount), static_cast<long>(i),
                      static_cast<double>(actualVExp[i]), static_cast<double>(goldenVExpF[i]),
                      fabs(static_cast<double>(actualVExp[i]) -
                           static_cast<double>(goldenVExpF[i])));
        }
        LOG_PRINT("\n");
    }
    return pass;
}

} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    int ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    const BwdCase cases[] = {
        {"global_clamp_calc", 1, 257, 0, 0, 0, true, true, 11},
        {"global_nocalc", 2, 128, 0, 0, 0, false, false, 22},
        {"grid_clamp_calc", 2, 517, 5, 5, 6, true, true, 33},
        {"grid_nocalc", 1, 4096, 3, 5, 4, false, false, 44},
        {"grid_large", 1, 100000, 12, 12, 8, true, true, 55},
    };
    const int caseCount = static_cast<int>(sizeof(cases) / sizeof(cases[0]));
    int failed = 0;
    for (int i = 0; i < caseCount; i++) {
        if (!RunBwdCase(cases[i], stream)) {
            failed++;
        }
    }
    LOG_PRINT("\n=== aclnnColorExposureBwd: %d/%d cases passed ===\n", caseCount - failed, caseCount);

    CleanupResources(deviceId, stream);
    return failed == 0 ? 0 : 1;
}
