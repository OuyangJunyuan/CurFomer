#include <cub/cub.cuh>

#include "common.h"


__global__
void InitVoxels(const float3 init_voxel, float3 (*__restrict__ voxel_infos)[3]) {
    voxel_infos[threadIdx.x][0] = {.0f, .0f, .0f};
    voxel_infos[threadIdx.x][1] = init_voxel;
    voxel_infos[threadIdx.x][2] = init_voxel + init_voxel;
}

__global__
void InitHashTables(const uint32_t num_hash,
                    const uint32_t *__restrict__ batch_masks,
                    uint32_t *__restrict__ hash_tables) {
    if (batch_masks[blockIdx.y]) {
        return;
    }
    hash_tables[num_hash * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x] = kEmpty;
}

__global__
void CountNonEmptyVoxel(const uint32_t num_src, const uint32_t num_hash,
                        const uint32_t *__restrict__ batch_masks,
                        const float3 *__restrict__ sources, const float3 (*__restrict__ voxel_infos)[3],
                        uint32_t *__restrict__ hash_tables, uint32_t *__restrict__ num_sampled) {
    const uint32_t bid = blockIdx.y;
    const uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (batch_masks[bid] || pid >= num_src) {
        return;
    }

    const auto voxel = voxel_infos[bid][1];
    const auto point = sources[num_src * bid + pid];
    const auto table = hash_tables + num_hash * bid;

    const uint32_t hash_key = coord_hash_32((int) roundf(point.x / voxel.x),
                                            (int) roundf(point.y / voxel.y),
                                            (int) roundf(point.z / voxel.z));

    const uint32_t kHashMax = num_hash - 1;
    uint32_t hash_slot = hash_key & kHashMax;

    while (true) {
        const uint32_t old = atomicCAS(table + hash_slot, kEmpty, hash_key);
        if (old == hash_key) {
            return;
        }
        if (old == kEmpty) {
            atomicAdd(num_sampled + bid, 1);
            return;
        }
        hash_slot = (hash_slot + 1) & kHashMax;
    }
}

__global__
void UpdateVoxelsSizeIfNotConverge(const uint32_t num_batch, const uint32_t num_trg,
                                   const uint32_t lower_bound, const uint32_t upper_bound,
                                   uint32_t *__restrict__ batch_masks,
                                   float3 (*__restrict__ voxel_infos)[3],
                                   uint32_t *__restrict__ num_sampled) {
    uint32_t bid = threadIdx.x;
    if (batch_masks[bid])
        return;

    const auto num = num_sampled[bid];
    if (lower_bound <= num and num <= upper_bound) {   // fall into tolerance.
        batch_masks[bid] = 1;
        atomicAdd(&batch_masks[num_batch], 1);
        dbg("%d", num);
        dbg("%f %f %f", voxel_infos[bid].c.x, voxel_infos[bid].c.y, voxel_infos[bid].c.z);
    } else {  // has not converged yet.
        if (num > num_trg) {
            voxel_infos[bid][0] = voxel_infos[bid][1];
        }
        if (num < num_trg) {
            voxel_infos[bid][2] = voxel_infos[bid][1];
        }
        // update current voxel by the average of left and right voxels.
        voxel_infos[bid][1] = (voxel_infos[bid][0] + voxel_infos[bid][2]) / 2.0f;
        num_sampled[bid] = 0;
    }
}

__global__
void FindMiniDistToCenterForEachNonEmptyVoxels(const uint32_t num_src, const uint32_t num_hash,
                                               const float3 *__restrict__ sources,
                                               const float3 (*__restrict__ voxel_infos)[3],
                                               uint32_t *__restrict__ hash_tables,
                                               float *__restrict__ dist_tables,
                                               uint32_t *__restrict__ point_slots,
                                               float *__restrict__ point_dists) {
    const uint32_t bid = blockIdx.y;
    const uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= num_src) {
        return;
    }

    const auto pid_global = num_src * bid + pid;
    const auto point = sources[pid_global];
    const auto voxel = voxel_infos[bid][1];

    const auto coord_x = roundf(point.x / voxel.x);
    const auto coord_y = roundf(point.y / voxel.y);
    const auto coord_z = roundf(point.z / voxel.z);
    const auto d1 = point.x - coord_x * voxel.x;
    const auto d2 = point.y - coord_y * voxel.y;
    const auto d3 = point.z - coord_z * voxel.z;
    const auto noise = (float) pid * FLT_MIN;  // to ensure all point distances are different.
    const auto dist = d1 * d1 + d2 * d2 + d3 * d3 + noise;
    point_dists[pid_global] = dist;

    const auto dist_table = dist_tables + num_hash * bid;
    const auto hash_table = hash_tables + num_hash * bid;
    const uint32_t hash_key = coord_hash_32((int) coord_x, (int) coord_y, (int) coord_z);

    const uint32_t kHashMax = num_hash - 1;
    uint32_t hash_slot = hash_key & kHashMax;
    while (true) {
        const uint32_t old = atomicCAS(hash_table + hash_slot, kEmpty, hash_key);
        assert(old != kEmpty); // should never meet.
        if (old == hash_key) {
            atomicMin(dist_table + hash_slot, dist);
            point_slots[pid_global] = hash_slot;
            return;
        }
        hash_slot = (hash_slot + 1) & kHashMax;
    }
}

__global__
void MaskSourceWithMinimumDistanceToCenter(const uint32_t num_src, const uint32_t num_hash,
                                           const uint32_t *__restrict__ point_slots,
                                           const float *__restrict__ point_dists,
                                           const float *__restrict__ dist_tables,
                                           uint32_t *__restrict__ point_masks) {
    const uint32_t bid = blockIdx.y;
    const uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= num_src) {
        return;
    }

    const uint32_t pid_global = num_src * bid + pid;
    const auto min_dist = dist_tables[num_hash * bid + point_slots[pid_global]];
    point_masks[pid_global] = min_dist == point_dists[pid_global];
}

inline
void ExclusivePrefixSum(const uint32_t num_batch, const uint32_t num_src, const uint32_t num_hash,
                        void *temp_mem, uint32_t *point_masks, uint32_t *point_masks_sum, cudaStream_t stream) {
    size_t temp_mem_size = num_batch * num_hash;  // must be higher than expected.

    for (int bid = 0; bid < num_batch; ++bid) {
        cub::DeviceScan::ExclusiveSum(
                temp_mem, temp_mem_size,
                point_masks + bid * num_src,
                point_masks_sum + bid * num_src,
                num_src, stream
        );
    }
}

__global__
void MaskOutSubsetIndices(const uint32_t num_src, const uint32_t num_trg,
                          const uint32_t *__restrict__ point_masks,
                          const uint32_t *__restrict__ point_masks_sum,
                          uint64_t *__restrict__ sampled_ids) {
    const uint32_t bid = blockIdx.y;
    const uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= num_src) {
        return;
    }

    const auto pid_global = num_src * bid + pid;
    const auto mask_sum = point_masks_sum[pid_global];
    if (point_masks[pid_global] and mask_sum < num_trg) {
        sampled_ids[num_trg * bid + mask_sum] = pid;
    }
}

__global__
void MaskOutSubsetIndices(const uint32_t num_src, const uint32_t num_trg, const uint32_t num_hash,
                          const uint32_t *__restrict__ point_slots,
                          const uint32_t *__restrict__ point_masks,
                          const uint32_t *__restrict__ point_masks_sum,
                          uint64_t *__restrict__ sampled_ids, uint32_t *__restrict__ hash2subset) {
    const uint32_t bid = blockIdx.y;
    const uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;

    if (pid >= num_src) {
        return;
    }
    const auto pid_global = num_src * bid + pid;
    const auto mask_sum = point_masks_sum[pid_global];
    if (point_masks[pid_global] and mask_sum < num_trg) {
        sampled_ids[num_trg * bid + mask_sum] = pid;
        hash2subset[num_hash * bid + point_slots[pid_global]] = mask_sum;
    }
}

void HAVSamplingBatchLauncher(const int num_batch, const int num_src,
                              const int num_trg, const int num_hash,
                              const float3 init_voxel, const float tolerance, const int max_iterations,
                              const float3 *sources, uint32_t *batch_masks,
                              uint32_t *num_sampled, float3 (*voxel_infos)[3],
                              uint32_t *hash_tables, float *dist_tables,
                              uint32_t *point_slots, float *point_dists, uint32_t *point_masks,
                              uint64_t *sampled_ids,
                              const bool return_hash2subset = false,
                              cudaStream_t stream = nullptr) {
    InitVoxels<<<1, num_batch, 0, stream>>>(init_voxel, voxel_infos);

    const auto src_grid = BLOCKS2D(num_src, num_batch);
    const auto table_grid = BLOCKS2D(num_hash, num_batch);
    const auto block = THREADS();

    const auto lower_bound = uint32_t((float) num_trg * (1.0f + 0.0f));
    const auto upper_bound = uint32_t((float) num_trg * (1.0f + tolerance));

    uint32_t num_complete = 0;
    uint32_t cur_iteration = 1;
    while (max_iterations >= cur_iteration++ and num_complete != num_batch) {
        InitHashTables<<<table_grid, block, 0, stream>>>(
                num_hash, batch_masks, hash_tables
        );
        CountNonEmptyVoxel<<<src_grid, block, 0, stream>>>(
                num_src, num_hash, batch_masks, sources, voxel_infos, hash_tables, num_sampled
        );

        if (max_iterations >= cur_iteration) {  // voxels should not be updated in last iteration.
            UpdateVoxelsSizeIfNotConverge<<<1, num_batch, 0, stream>>>(
                    num_batch, num_trg, lower_bound, upper_bound, batch_masks, voxel_infos, num_sampled
            );
        }
        cudaMemcpyAsync(
                &num_complete, &batch_masks[num_batch],
                sizeof(num_complete), cudaMemcpyDeviceToHost, stream
        );
        cudaStreamSynchronize(stream);
    }
    FindMiniDistToCenterForEachNonEmptyVoxels<<< src_grid, block, 0, stream>>>(
            num_src, num_hash, sources, voxel_infos, hash_tables, dist_tables, point_slots, point_dists
    );
    MaskSourceWithMinimumDistanceToCenter<<<src_grid, block, 0, stream>>>(
            num_src, num_hash, point_slots, point_dists, dist_tables, point_masks
    );

    auto *temp_mem = (void *) dist_tables;  // reuse dist_tables as temporary memories.
    auto *point_masks_sum = (uint32_t *) point_dists;  // reuse point_dists as point_masks_sum
    ExclusivePrefixSum(
            num_batch, num_src, num_hash, temp_mem, point_masks, point_masks_sum, stream
    );
    if (return_hash2subset) {
        auto *hash2subset = (uint32_t *) dist_tables;  // reuse dist_tables as hash2subset.
        // cudaMemsetAsync(hash2subset, 0x00, num_batch * num_hash * sizeof(uint32_t), stream);
        MaskOutSubsetIndices<<<src_grid, block, 0, stream>>>(
                num_src, num_trg, num_hash, point_slots, point_masks, point_masks_sum, sampled_ids, hash2subset
        );
    } else {
        MaskOutSubsetIndices<<<src_grid, block, 0, stream>>>(
                num_src, num_trg, point_masks, point_masks_sum, sampled_ids
        );
    }
}

std::vector<at::Tensor> HAVSamplingBatchWrapper(at::Tensor &sources,  // [in]
                                                at::Tensor &sampled_ids,  // [out]
                                                const double init_voxel_x,
                                                const double init_voxel_y,
                                                const double init_voxel_z,
                                                const double tolerance,
                                                const long max_iterations,
                                                const bool return_sample_infos = false,
                                                const bool return_query_infos = false) {
    const int64_t num_batch = sources.size(0);
    const int64_t num_src = sources.size(1);
    const int64_t num_trg = sampled_ids.size(1);
    const int64_t num_hash = get_table_size(num_src);
    const float3 init_voxel = {(float) init_voxel_x,
                               (float) init_voxel_y,
                               (float) init_voxel_z};

    auto batch_masks_tensor = sources.new_empty({num_batch + 1}, TorchType<int>);  // 哪个batch已完成体素搜索
    auto num_sampled_tensor = sources.new_empty({num_batch}, TorchType<int>);  // 当前体素对应的非空体素个数
    auto hash_tables_tensor = sources.new_empty({num_batch, num_hash}, TorchType<int>);  // 哈希表：key
    auto dist_tables_tensor = sources.new_empty({num_batch, num_hash}, TorchType<float>);  // 哈希表：val-dist
    auto point_slots_tensor = sources.new_empty({num_batch, num_src}, TorchType<int>);  // 输入点的哈希表slot
    auto point_dists_tensor = sources.new_empty({num_batch, num_src}, TorchType<float>);  // 输入点到其体素的体素
    auto point_masks_tensor = sources.new_empty({num_batch, num_src}, TorchType<int>);  // 输入点是否被采样
    auto voxel_infos_tensor = sources.new_empty({num_batch, 3 * 3}, TorchType<float>);  // 体素尺寸信息

    sampled_ids.zero_();
    batch_masks_tensor.zero_();
    num_sampled_tensor.zero_();
    dist_tables_tensor.fill_(at::Scalar(FLT_MAX));

    HAVSamplingBatchLauncher(
            num_batch, num_src, num_trg, num_hash,
            init_voxel, tolerance, max_iterations,
            (float3 *) sources.data_ptr(),
            (uint32_t *) batch_masks_tensor.data_ptr(),
            (uint32_t *) num_sampled_tensor.data_ptr(),
            (float3 (*)[3]) voxel_infos_tensor.data_ptr(),
            (uint32_t *) hash_tables_tensor.data_ptr(),
            (float *) dist_tables_tensor.data_ptr(),
            (uint32_t *) point_slots_tensor.data_ptr(),
            (float *) point_dists_tensor.data_ptr(),
            (uint32_t *) point_masks_tensor.data_ptr(),
            (uint64_t *) sampled_ids.data_ptr(),
            return_query_infos
    );

    auto ret = std::vector<at::Tensor>{voxel_infos_tensor};
    if (return_sample_infos) {
        ret.emplace_back(num_sampled_tensor);  // the number of points actually sampled.
        ret.emplace_back(point_masks_tensor);  // the mask of which sources point is sampled.
    }
    if (return_query_infos) {
        ret.emplace_back(hash_tables_tensor);  // reuse as the hashtable of input points.
        ret.emplace_back(dist_tables_tensor);  // reuse as the hashtable with subset indices.
    }
    return ret;
}

__global__
void dynamic_voxelization_kernel(const uint32_t num_points, const uint32_t num_feats, const uint32_t num_hash,
                                 const float *__restrict__ points,
                                 int4 *__restrict__ coors,
                                 float *__restrict__ feats,
                                 uint *__restrict__ table,
                                 uint *__restrict__ value,
                                 uint *num_voxels,
                                 uint *indices,
                                 const float3 voxel_size,
                                 const float3 coors_range_min,
                                 const float3 coors_range_max,
                                 int32_t bid) {
    const uint32_t pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= num_points) {
        return;
    }
    const auto point = *(float3 *) (points + num_feats * pid);
    if ((point.x < coors_range_min.x or point.x >= coors_range_max.x) or
        (point.y < coors_range_min.y or point.y >= coors_range_max.y) or
        (point.z < coors_range_min.z or point.z >= coors_range_max.z)) {
        indices[pid] = kEmpty;
        return;
    }
    const auto c = int4{(int) bid,
                        (int) floor((point.x - coors_range_min.x) / voxel_size.x),
                        (int) floor((point.y - coors_range_min.y) / voxel_size.y),
                        (int) floor((point.z - coors_range_min.z) / voxel_size.z)};

    const uint32_t key = coord_hash_32(c.y, c.z, c.w);
    const uint32_t kHashMax = num_hash - 1;
    uint32_t slot = key & kHashMax;
    while (true) {
        const uint32_t old = atomicCAS(table + slot, kEmpty, key);
        if (old == key) {
            const auto vid = value[slot];
            for (int i = 0; i < num_feats; ++i) {
                atomicAdd(feats + (num_feats + 1) * vid + i, points[num_feats * pid + i]);
            }
            atomicAdd(feats + (num_feats + 1) * vid + num_feats, 1.0f);
            return;
        }
        if (old == kEmpty) {
            const auto vid = value[slot] = atomicInc(num_voxels, kEmpty);
            coors[vid] = c;
            return;
        }
        slot = (slot + 1) & kHashMax;
    }
}


void DynamicVoxelizationLauncher(const uint num_points,
                                 const uint num_feats,
                                 const uint num_hash,
                                 const float *points,
                                 int4 *coors,
                                 float *feats,
                                 uint *table,
                                 uint *value,
                                 uint *num_voxels,
                                 uint *indices,
                                 const float3 voxel_size,
                                 const float3 coors_range_min,
                                 const float3 coors_range_max,
                                 const int batch_id) {
    const auto grid = BLOCKS1D(num_points);
    const auto block = THREADS();
    dynamic_voxelization_kernel<<<grid, block >>>(
            num_points,
            num_feats,
            num_hash,
            points,
            coors,
            feats,
            table,
            value,
            num_voxels,
            indices,
            voxel_size,
            coors_range_min,
            coors_range_max,
            batch_id
    );
}

std::vector<at::Tensor> DynamicVoxelizationBatchWrapper(const at::Tensor &points,
                                                        const std::vector<double> &voxel_size,
                                                        const std::vector<double> &coors_range,
                                                        long batch_id) {
    const int64_t num_points = points.size(0);
    const int64_t num_feats = points.size(1);
    const int64_t num_hash = get_table_size(num_points);

    auto table = points.new_empty({num_hash}, TorchType<int>);
    auto value = points.new_empty({num_hash}, TorchType<int>);
    auto num_voxels = points.new_zeros({1}, TorchType<int>);

    auto coors = points.new_empty({num_points, 4}, TorchType<int>);
    auto feats = points.new_zeros({num_points, num_feats + 1}, TorchType<float>);
    auto indices = points.new_empty({num_points, 1}, TorchType<int>);
    table.fill_(at::Scalar(-1));

    DynamicVoxelizationLauncher(num_points, num_feats, num_hash,
                                (float *) points.data_ptr(),
                                (int4 *) coors.data_ptr(),
                                (float *) feats.data_ptr(),
                                (uint *) table.data_ptr(),
                                (uint *) value.data_ptr(),
                                (uint *) num_voxels.data_ptr(),
                                (uint *) indices.data_ptr(),
                                float3{(float) voxel_size[0], (float) voxel_size[1], (float) voxel_size[2]},
                                float3{(float) coors_range[0], (float) coors_range[1], (float) coors_range[2]},
                                float3{(float) coors_range[3], (float) coors_range[4], (float) coors_range[5]},
                                (int) batch_id);
    return {indices, coors, feats, num_voxels};
}
