#include <string>
#include <torch/torch.h>
#include "functions.h"
#include "OpApiCommon.h"

using namespace NPU_NAME_SPACE;
using namespace std;


at::Tensor get_render_schedule(const at::Tensor &nums_tensor, int num_bins)
{
    auto device = nums_tensor.device();
    // 转换为 std::vector<int>
    at::Tensor cpu_tensor = nums_tensor.cpu();
    std::vector<int> nums(cpu_tensor.data_ptr<long>(), cpu_tensor.data_ptr<long>() + cpu_tensor.numel());
    // 1. 对nums进行排序，获取排序后的索引
    std::vector<int> tile_idxes(nums.size());
    for (size_t i = 0; i < nums.size(); ++i) {
        tile_idxes[i]=i;
    }
    std::sort(tile_idxes.begin(), tile_idxes.end(), [&nums](int a, int b) {
        return nums[a] < nums[b];
    });
    // 2. 初始化bins_new, 存储每个 bin 的 tile索引
    std::vector<std::vector<int>> bins_new(num_bins);

    // 3. 使用优先队列进行bin分配
    using Pair = std::pair<int, int>;
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> pq;
    for (int i = 0; i < num_bins; ++i) {
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
    std::vector<int> schedule(num_bins + nums.size() + nums.size(), 0);

    // 填充 bin 大小和 tile 索引
    for (int i = 0; i < num_bins; ++i) {
        if (i == 0) {
            schedule[i] = bins_new[i].size();
            for (size_t j = 0; j < bins_new[i].size(); ++j) {
                schedule[num_bins + j] = bins_new[i][j];
            }
        } else {
            schedule[i] = schedule[i-1] + bins_new[i].size();
            for (size_t j = 0; j < bins_new[i].size(); ++j) {
                schedule[num_bins+schedule[i-1]+j] = bins_new[i][j];
            }
        }
    }
    // 填充累积和
    int offset = num_bins + nums.size();
    for (size_t i = 0; i<nums.size(); ++i) {
        if (i == 0) {
            schedule[offset + i] = nums[i];
        } else {
            schedule[offset + i] = schedule[offset + i -1] + nums[i];
        }
    }
    at::Tensor schedule_npu = at::tensor(schedule, at::dtype(torch::kInt64)).to(device);
    return schedule_npu;
}