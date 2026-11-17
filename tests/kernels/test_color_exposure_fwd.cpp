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
#include "aclnnop/aclnn_color_exposure_fwd.h"
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

VerifyReport VerifyExactInt(const vector<int32_t>& actual, const vector<int32_t>& expected)
{
    VerifyReport rep;
    rep.pass = true;
    rep.maxAbsDiff = 0.0;
    rep.mismatchCount = 0;
    rep.firstBadIndex = -1;
    const int64_t n = static_cast<int64_t>(expected.size());
    for (int64_t i = 0; i < n; i++) {
        if (actual[i] != expected[i]) {
            rep.mismatchCount++;
            if (rep.firstBadIndex < 0) {
                rep.firstBadIndex = i;
            }
        }
    }
    rep.pass = rep.mismatchCount == 0;
    return rep;
}

struct FwdCase {
    const char* name;
    int64_t batch;
    int64_t gauss;
    int32_t gridW;
    int32_t gridH;
    int32_t gridD;
    bool clamp;
    bool exposure5d;
    bool identityExposure;
    uint32_t seed;
};

struct FwdTensors {
    void* colorsAddr = nullptr;
    void* means2dAddr = nullptr;
    void* exposureAddr = nullptr;
    void* outputAddr = nullptr;
    void* indexAddr = nullptr;
    void* weightAddr = nullptr;
    aclTensor* colorsTensor = nullptr;
    aclTensor* means2dTensor = nullptr;
    aclTensor* exposureTensor = nullptr;
    aclTensor* outputTensor = nullptr;
    aclTensor* indexTensor = nullptr;
    aclTensor* weightTensor = nullptr;
    void* workspaceAddr = nullptr;
};

void DestroyFwdTensors(FwdTensors& t)
{
    if (t.colorsTensor) aclDestroyTensor(t.colorsTensor);
    if (t.means2dTensor) aclDestroyTensor(t.means2dTensor);
    if (t.exposureTensor) aclDestroyTensor(t.exposureTensor);
    if (t.outputTensor) aclDestroyTensor(t.outputTensor);
    if (t.indexTensor) aclDestroyTensor(t.indexTensor);
    if (t.weightTensor) aclDestroyTensor(t.weightTensor);
    if (t.colorsAddr) aclrtFree(t.colorsAddr);
    if (t.means2dAddr) aclrtFree(t.means2dAddr);
    if (t.exposureAddr) aclrtFree(t.exposureAddr);
    if (t.outputAddr) aclrtFree(t.outputAddr);
    if (t.indexAddr) aclrtFree(t.indexAddr);
    if (t.weightAddr) aclrtFree(t.weightAddr);
    if (t.workspaceAddr) aclrtFree(t.workspaceAddr);
}

bool RunFwdCase(const FwdCase& cs, aclrtStream stream)
{
    const int64_t b = cs.batch;
    const int64_t n = cs.gauss;
    const int64_t bcn = b * n;
    const bool useGrid = cs.gridW > 0;
    const int64_t expNumel = useGrid ?
        static_cast<int64_t>(cs.gridW) * cs.gridH * cs.gridD * 12 : 12;

    vector<float> colorsHost(bcn * 3, 0.0f);
    vector<float> means2dHost(bcn * 2, 0.0f);
    vector<float> exposureHost(expNumel, 0.0f);
    uint32_t rng = cs.seed;

    for (int64_t i = 0; i < bcn * 3; i++) {
        colorsHost[i] = (NextUniform(rng) * 2.0f - 1.0f) * 0.5f;
    }
    for (int64_t i = 0; i < bcn; i++) {
        means2dHost[i] = (NextUniform(rng) * 1.2f - 0.1f) * static_cast<float>(kImageWidth);
        means2dHost[bcn + i] = (NextUniform(rng) * 1.2f - 0.1f) * static_cast<float>(kImageHeight);
    }
    if (cs.identityExposure) {
        const float identity[12] = {1.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 1.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 1.0f, 0.0f};
        for (int64_t i = 0; i < expNumel; i++) {
            exposureHost[i] = identity[i % 12];
        }
    } else {
        for (int64_t i = 0; i < expNumel; i++) {
            exposureHost[i] = (NextUniform(rng) * 2.0f - 1.0f) * 0.1f;
        }
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

    vector<float> goldenOut(bcn * 3, 0.0f);
    vector<int32_t> goldenIndex(useGrid ? bcn * 3 : 0, 0);
    vector<float> goldenWeight(useGrid ? bcn * 3 : 0.0f);
    const float invW = 1.0f / static_cast<float>(kImageWidth);
    const float invH = 1.0f / static_cast<float>(kImageHeight);

    for (int64_t bi = 0; bi < b; bi++) {
        for (int64_t j = 0; j < n; j++) {
            const float r = colorsHost[(bi * 3 + 0) * n + j];
            const float g = colorsHost[(bi * 3 + 1) * n + j];
            const float bl = colorsHost[(bi * 3 + 2) * n + j];
            float e[12];
            if (useGrid) {
                const float x = means2dHost[(bi * 2 + 0) * n + j];
                const float y = means2dHost[(bi * 2 + 1) * n + j];
                const float grey = 0.299f * r + 0.587f * g + 0.114f * bl;
                const GridCoord c = MapToGrid(x, y, grey, cs.gridW, cs.gridH, cs.gridD, invW, invH);
                InterpolateAffine(exposureHost.data(), c, cs.gridH, cs.gridD, e);
                goldenIndex[(bi * 3 + 0) * n + j] = c.ix;
                goldenIndex[(bi * 3 + 1) * n + j] = c.iy;
                goldenIndex[(bi * 3 + 2) * n + j] = c.iz;
                goldenWeight[(bi * 3 + 0) * n + j] = c.wx;
                goldenWeight[(bi * 3 + 1) * n + j] = c.wy;
                goldenWeight[(bi * 3 + 2) * n + j] = c.wz;
            } else {
                for (int t = 0; t < 12; t++) {
                    e[t] = exposureHost[t];
                }
            }
            for (int row = 0; row < 3; row++) {
                float out = ApplyAffineRow(e, row, r, g, bl);
                if (cs.clamp && out < 0.0f) {
                    out = 0.0f;
                }
                goldenOut[(bi * 3 + row) * n + j] = out;
            }
        }
    }

    FwdTensors t;
    vector<int64_t> colorsShape = {b, 3, n};
    vector<int64_t> means2dShape = {b, 2, n};
    vector<int64_t> expShape;
    if (useGrid) {
        if (cs.exposure5d) {
            expShape = {cs.gridW, cs.gridH, cs.gridD, 4, 3};
        } else {
            expShape = {cs.gridW, cs.gridH, cs.gridD, 12};
        }
    } else {
        expShape = {12};
    }
    vector<int64_t> outShape = {b, 3, n};
    vector<int64_t> indexShape = {b, 3, n};
    vector<float> zeroOut(bcn * 3, 0.0f);
    vector<int32_t> zeroIndex(bcn * 3, 0);
    vector<float> zeroWeight(bcn * 3, 0.0f);

    int ret = CreateAclTensor(colorsHost, colorsShape, &t.colorsAddr,
                              aclDataType::ACL_FLOAT, &t.colorsTensor);
    bool ok = CHECK_RET(ret == ACL_SUCCESS);
    if (!ok) {
        LOG_PRINT("CreateAclTensor colors failed. ERROR: %d\n", ret);
        DestroyFwdTensors(t);
        return false;
    }
    ret = CreateAclTensor(means2dHost, means2dShape, &t.means2dAddr,
                          aclDataType::ACL_FLOAT, &t.means2dTensor);
    ok = CHECK_RET(ret == ACL_SUCCESS);
    if (!ok) {
        LOG_PRINT("CreateAclTensor means2d failed. ERROR: %d\n", ret);
        DestroyFwdTensors(t);
        return false;
    }
    ret = CreateAclTensor(exposureHost, expShape, &t.exposureAddr,
                          aclDataType::ACL_FLOAT, &t.exposureTensor);
    ok = CHECK_RET(ret == ACL_SUCCESS);
    if (!ok) {
        LOG_PRINT("CreateAclTensor exposure failed. ERROR: %d\n", ret);
        DestroyFwdTensors(t);
        return false;
    }
    ret = CreateAclTensor(zeroOut, outShape, &t.outputAddr,
                          aclDataType::ACL_FLOAT, &t.outputTensor);
    ok = CHECK_RET(ret == ACL_SUCCESS);
    if (!ok) {
        LOG_PRINT("CreateAclTensor output failed. ERROR: %d\n", ret);
        DestroyFwdTensors(t);
        return false;
    }
    ret = CreateAclTensor(zeroIndex, indexShape, &t.indexAddr,
                          aclDataType::ACL_INT32, &t.indexTensor);
    ok = CHECK_RET(ret == ACL_SUCCESS);
    if (!ok) {
        LOG_PRINT("CreateAclTensor index failed. ERROR: %d\n", ret);
        DestroyFwdTensors(t);
        return false;
    }
    ret = CreateAclTensor(zeroWeight, indexShape, &t.weightAddr,
                          aclDataType::ACL_FLOAT, &t.weightTensor);
    ok = CHECK_RET(ret == ACL_SUCCESS);
    if (!ok) {
        LOG_PRINT("CreateAclTensor weight failed. ERROR: %d\n", ret);
        DestroyFwdTensors(t);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnColorExposureFwdGetWorkspaceSize(t.colorsTensor, t.means2dTensor, t.exposureTensor,
                                                kImageWidth, kImageHeight, cs.clamp ? 1 : 0,
                                                t.outputTensor, t.indexTensor, t.weightTensor,
                                                &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnColorExposureFwdGetWorkspaceSize failed. ERROR: %d\n", ret);
        DestroyFwdTensors(t);
        return false;
    }
    if (workspaceSize > 0ULL) {
        ret = aclrtMalloc(&t.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            DestroyFwdTensors(t);
            return false;
        }
    }
    ret = aclnnColorExposureFwd(t.workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnColorExposureFwd failed. ERROR: %d\n", ret);
        DestroyFwdTensors(t);
        return false;
    }
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        DestroyFwdTensors(t);
        return false;
    }

    vector<float> actualOut(bcn * 3, 0.0f);
    ret = aclrtMemcpy(actualOut.data(), actualOut.size() * sizeof(float), t.outputAddr,
                      actualOut.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy output to host failed. ERROR: %d\n", ret);
        DestroyFwdTensors(t);
        return false;
    }

    std::string expDesc = "[12]";
    if (useGrid) {
        expDesc = "[" + std::to_string(cs.gridW) + "," + std::to_string(cs.gridH) + "," +
                  std::to_string(cs.gridD) + (cs.exposure5d ? ",4,3]" : ",12]");
    }

    const double outRtol = cs.identityExposure ? 0.0 : 1e-5;
    const double outAtol = cs.identityExposure ? 0.0 : 1e-5;
    const VerifyReport outRep = VerifyClose(actualOut, goldenOut, outRtol, outAtol);
    bool pass = outRep.pass;

    VerifyReport indexRep;
    indexRep.pass = true;
    indexRep.maxAbsDiff = 0.0;
    indexRep.mismatchCount = 0;
    indexRep.firstBadIndex = -1;
    VerifyReport weightRep = indexRep;
    if (useGrid) {
        vector<int32_t> actualIndex(bcn * 3, 0);
        vector<float> actualWeight(bcn * 3, 0.0f);
        ret = aclrtMemcpy(actualIndex.data(), actualIndex.size() * sizeof(int32_t), t.indexAddr,
                          actualIndex.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        ok = CHECK_RET(ret == ACL_SUCCESS);
        if (ok) {
            ret = aclrtMemcpy(actualWeight.data(), actualWeight.size() * sizeof(float), t.weightAddr,
                              actualWeight.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            ok = CHECK_RET(ret == ACL_SUCCESS);
        }
        if (!ok) {
            LOG_PRINT("copy index/weight to host failed. ERROR: %d\n", ret);
            DestroyFwdTensors(t);
            return false;
        }
        indexRep = VerifyExactInt(actualIndex, goldenIndex);
        weightRep = VerifyClose(actualWeight, goldenWeight, 1e-5, 1e-5);
        pass = pass && indexRep.pass && weightRep.pass;
    }

    DestroyFwdTensors(t);

    if (pass) {
        LOG_PRINT("[fwd:%s] B=%ld N=%ld exposure=%s clamp=%d PASS output_max_diff=%.3e",
                  cs.name, static_cast<long>(b), static_cast<long>(n), expDesc.c_str(),
                  cs.clamp ? 1 : 0, outRep.maxAbsDiff);
        if (useGrid) {
            LOG_PRINT(" index_exact weight_max_diff=%.3e", weightRep.maxAbsDiff);
        }
        LOG_PRINT("\n");
    } else {
        LOG_PRINT("[fwd:%s] B=%ld N=%ld exposure=%s clamp=%d FAIL", cs.name, static_cast<long>(b),
                  static_cast<long>(n), expDesc.c_str(), cs.clamp ? 1 : 0);
        if (!outRep.pass) {
            const int64_t i = outRep.firstBadIndex;
            LOG_PRINT(" output mismatch=%ld first@%ld act=%.7f exp=%.7f diff=%.3e",
                      static_cast<long>(outRep.mismatchCount), static_cast<long>(i),
                      static_cast<double>(actualOut[i]), static_cast<double>(goldenOut[i]),
                      fabs(static_cast<double>(actualOut[i]) - static_cast<double>(goldenOut[i])));
        }
        if (useGrid && !indexRep.pass) {
            LOG_PRINT(" index mismatch=%ld first@%ld", static_cast<long>(indexRep.mismatchCount),
                      static_cast<long>(indexRep.firstBadIndex));
        }
        if (useGrid && !weightRep.pass) {
            LOG_PRINT(" weight mismatch=%ld first@%ld",
                      static_cast<long>(weightRep.mismatchCount),
                      static_cast<long>(weightRep.firstBadIndex));
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

    const FwdCase cases[] = {
        {"global_basic", 1, 4096, 0, 0, 0, false, false, false, 42},
        {"global_identity_exact", 2, 257, 0, 0, 0, false, false, true, 7},
        {"global_clamp_tail", 2, 123, 0, 0, 0, true, false, false, 1234},
        {"grid_aligned", 1, 4096, 7, 7, 8, false, false, false, 99},
        {"grid_clamp_tail", 3, 13, 2, 2, 2, true, false, false, 555},
        {"grid_tiny", 1, 7, 5, 4, 3, false, false, false, 77},
        {"grid_5d_exposure", 2, 512, 4, 5, 6, true, true, false, 2026},
        {"grid_large", 1, 100000, 12, 12, 8, true, false, false, 8888},
    };
    const int caseCount = static_cast<int>(sizeof(cases) / sizeof(cases[0]));
    int failed = 0;
    for (int i = 0; i < caseCount; i++) {
        if (!RunFwdCase(cases[i], stream)) {
            failed++;
        }
    }
    LOG_PRINT("\n=== aclnnColorExposureFwd: %d/%d cases passed ===\n", caseCount - failed, caseCount);

    CleanupResources(deviceId, stream);
    return failed == 0 ? 0 : 1;
}
