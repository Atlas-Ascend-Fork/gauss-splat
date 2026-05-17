/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <omp.h>
#include <torch/torch.h>

#include <algorithm>
#include <numeric>
#include <string>

#include "OpApiCommon.h"
#include "functions.h"

using namespace NPU_NAME_SPACE;
using namespace std;

namespace {
// 应对贪心算法分配不均的边界情况，预留1-2额外空间
constexpr int RESERVE_CAPACITY_MARGIN = 2;
// schedule:[num_bins,tile_ids(T),tile_offsets(T)] 每个tile需要存储两个字段，tile_id和累积offset
constexpr int SCHEDULE_DATA_SECTION_PER_TILE = 2;
// 张量维度常量
constexpr int SINGLE_TASK_DIM = 1;  // 单任务输入维度[T]
constexpr int BATCH_TASK_DIM = 3;   // 批量任务输入维度[B,C,T]
// 张量维度索引
constexpr int BATCH_DIM_INDEX = 0;   // batch维度索引
constexpr int CAMERA_DIM_INDEX = 1;  // camera维度索引
constexpr int TILE_DIM_INDEX = 2;    // tile维度索引
}  // namespace

std::vector<int> get_render_schedule_impl(const std::vector<int>& nums, int num_bins)
{
    int T = nums.size();
    if (T == 0) {
        return std::vector<int>(num_bins, 0);
    }
    // 动态bin_num，避免空bin
    int effective_bins = std::min(T, num_bins);
    // 1. 对nums进行排序，获取排序后的索引
    std::vector<int> tile_idxes(T);
    std::iota(tile_idxes.begin(), tile_idxes.end(), 0);
    std::sort(tile_idxes.begin(), tile_idxes.end(), [&nums](int a, int b) { return nums[a] < nums[b]; });
    // 2. 贪心分配
    std::vector<std::vector<int>> bins_new(effective_bins);
    // 预分配内存
    int avg_tiles_per_bin = (T + effective_bins - 1) / effective_bins;
    for (int i = 0; i < effective_bins; i++) {
        bins_new[i].reserve(avg_tiles_per_bin + RESERVE_CAPACITY_MARGIN);
    }
    // 3. 使用优先队列进行bin分配
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> pq;
    for (int i = 0; i < effective_bins; i++) {
        pq.push({0, i});
    }
    // 从大到小分配 tile
    for (auto it = tile_idxes.rbegin(); it != tile_idxes.rend(); ++it) {
        int tile_idx = *it;
        auto [bin_num, idx] = pq.top();
        pq.pop();
        bins_new[idx].push_back(tile_idx);
        pq.push({bin_num + nums[tile_idx], idx});
    }
    // 4. 构建调度数组
    int M = num_bins + SCHEDULE_DATA_SECTION_PER_TILE * T;
    std::vector<int> schedule(M, 0);
    // 填充 bin 大小和 tile 索引
    int cumsum = 0;
    for (int i = 0; i < effective_bins; i++) {
        int bin_size = bins_new[i].size();
        schedule[i] = cumsum + bin_size;
        int base = num_bins + cumsum;
        for (size_t j = 0; j < bin_size; j++) {
            schedule[base + j] = bins_new[i][j];
        }
        cumsum += bin_size;
    }
    for (int i = effective_bins; i < num_bins; i++) {
        schedule[i] = cumsum;
    }
    // 填充累积和
    int offset = num_bins + T;
    schedule[offset] = nums[0];
    for (int i = 1; i < T; i++) {
        schedule[offset + i] = schedule[offset + i - 1] + nums[i];
    }
    return schedule;
}

at::Tensor single_get_render_schedule(const at::Tensor& nums_tensor, int num_bins)
{
    TORCH_CHECK(nums_tensor.dim() == SINGLE_TASK_DIM, "nums_tensor must be 1D for single task");
    TORCH_CHECK(nums_tensor.scalar_type() == at::kLong, "must be int64 dtype");

    auto device = nums_tensor.device();
    // 转换为 std::vector<int>
    at::Tensor cpu_tensor = nums_tensor.cpu();
    std::vector<int> nums(cpu_tensor.data_ptr<long>(), cpu_tensor.data_ptr<long>() + cpu_tensor.numel());

    auto schedule = get_render_schedule_impl(nums, num_bins);

    return at::tensor(schedule, at::dtype(torch::kInt64)).to(device);
}

at::Tensor batch_get_render_schedule(const at::Tensor& tile_sums, int num_bins)
{
    TORCH_CHECK(tile_sums.dim() == BATCH_TASK_DIM, "tile_sums must be 3D [B, C, T]");
    TORCH_CHECK(tile_sums.device().is_cpu(), "must be CPU tensor");
    TORCH_CHECK(tile_sums.scalar_type() == at::kLong, "must be int64 dtype");

    int B = tile_sums.size(BATCH_DIM_INDEX);
    int C = tile_sums.size(CAMERA_DIM_INDEX);
    int T = tile_sums.size(TILE_DIM_INDEX);
    int M = num_bins + SCHEDULE_DATA_SECTION_PER_TILE * T;

    auto schedules = at::zeros({B, C, M}, tile_sums.options());

    const long* input_ptr = tile_sums.data_ptr<long>();
    long* output_ptr = schedules.data_ptr<long>();

// 并行处理
#pragma omp parallel
    {
        // 线程局部存储
        std::vector<int> nums(T);
        std::vector<int> schedule;
#pragma omp for schedule(dynamic, 1)
        for (int idx = 0; idx < B * C; idx++) {
            // 计算偏移量
            int offset_in = idx * T;
            int offset_out = idx * M;
            // 提取数据
            for (int t = 0; t < T; t++) {
                nums[t] = static_cast<int>(input_ptr[offset_in + t]);
            }
            schedule = get_render_schedule_impl(nums, num_bins);
            // 写入结果
            for (int m = 0; m < M; m++) {
                output_ptr[offset_out + m] = schedule[m];
            }
        }
    }
    return schedules;
}

// 动态路由 -- 根据输入维度自动选择最优实现
at::Tensor get_render_schedule(const at::Tensor& nums_tensor, int num_bins)
{
    TORCH_CHECK(num_bins > 0, "num_bins must be positive");
    if (nums_tensor.dim() == SINGLE_TASK_DIM) {
        // 单任务场景
        return single_get_render_schedule(nums_tensor, num_bins);
    } else if (nums_tensor.dim() == BATCH_TASK_DIM) {
        // 多任务场景
        int B = nums_tensor.size(BATCH_DIM_INDEX);
        int C = nums_tensor.size(CAMERA_DIM_INDEX);
        int total_tasks = B * C;
        // 避免并行开销，BxC=1时，使用单任务实现
        if (total_tasks == 1) {
            auto input_cpu = nums_tensor.cpu();
            std::vector<int> nums(input_cpu.data_ptr<long>(),
                                  input_cpu.data_ptr<long>() + nums_tensor.size(TILE_DIM_INDEX));
            auto schedule_vec = get_render_schedule_impl(nums, num_bins);
            auto schedule = at::tensor(schedule_vec, at::dtype(torch::kInt64));
            return schedule.to(nums_tensor.device()).unsqueeze(0).unsqueeze(0);
        } else {
            return batch_get_render_schedule(nums_tensor, num_bins);
        }
    } else {
        TORCH_CHECK(false, "nums_tensor must be 1D[T] or 3D [B, C, T], got dim=", nums_tensor.dim());
    }
}