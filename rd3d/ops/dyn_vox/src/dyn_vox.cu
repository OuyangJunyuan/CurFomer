#include <cuda.h>
#include <cub/cub.cuh>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "common.h"


__global__
void dynamic_voxelization_kernel(const uint32_t num_points,
                                 const uint32_t num_feats,
                                 const uint32_t max_slot,
                                 const int bid,
                                 const float3 voxel_size,
                                 const float3 coors_range_min,
                                 const float3 coors_range_max,
                                 const float *__restrict__ points,
                                 int4 *__restrict__ coors,
                                 float *__restrict__ feats,
                                 uint *__restrict__ num_voxels,
                                 uint2 *__restrict__ table) {
    const uint pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= num_points) {
        return;
    }
    const auto *const point = points + num_feats * pid;
    const auto xyz = *(float3 *) point;
    if ((xyz.x < coors_range_min.x or xyz.x >= coors_range_max.x) or
        (xyz.y < coors_range_min.y or xyz.y >= coors_range_max.y) or
        (xyz.z < coors_range_min.z or xyz.z >= coors_range_max.z)) {
        return;
    }
    const auto c = int4{(int) bid,
                        (int) floor((xyz.x - coors_range_min.x) / voxel_size.x),
                        (int) floor((xyz.y - coors_range_min.y) / voxel_size.y),
                        (int) floor((xyz.z - coors_range_min.z) / voxel_size.z)};

    const uint key = coord_hash_32(c.y, c.z, c.w);
    uint32_t slot = key & max_slot;
    while (true) {
        const uint32_t old = atomicCAS(&table[slot].x, kEmpty, key);
        if (old == kEmpty) {
            auto vid = atomicAdd(num_voxels, 1);
            table[slot].y = vid;
            coors[vid] = c;
            return;
        }
        if (old == key) {
            return;
        }
        slot = (slot + 1) & max_slot;
    }

    // hash.insert(coord_hash_32(c.y, c.z, c.w), [&](auto &val) {
    //     vid = val = atomicAdd(num_voxels, 1);
    //     coors[vid] = c;
    // }, [&](auto &val) {
    //     vid = val;
    // });
    // if (vid > num_points) {
    //     printf("%u\n", vid);
    // }
    // auto *const feat = feats + (num_feats + 1) * vid;
    // for (int i = 0; i < num_feats; ++i) {
    //     atomicAdd(feat + i, point[i]);
    // }
    // atomicAdd(feat + num_feats, 1.0f);
}

__global__
void dynamic_voxelization_kernel2(const uint32_t num_points,
                                  const uint32_t num_feats,
                                  const uint32_t max_slot,
                                  const int bid,
                                  const float3 voxel_size,
                                  const float3 coors_range_min,
                                  const float3 coors_range_max,
                                  const float *__restrict__ points,
                                  int4 *__restrict__ coors,
                                  float *__restrict__ feats,
                                  uint *__restrict__ num_voxels,
                                  uint2 *__restrict__ table) {
    const uint pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= num_points) {
        return;
    }
    const auto *const point = points + num_feats * pid;
    const auto xyz = *(float3 *) point;

    const auto c = int4{(int) bid,
                        (int) floor((xyz.x - coors_range_min.x) / voxel_size.x),
                        (int) floor((xyz.y - coors_range_min.y) / voxel_size.y),
                        (int) floor((xyz.z - coors_range_min.z) / voxel_size.z)};

    const uint key = coord_hash_32(c.y, c.z, c.w);
    uint32_t slot = key & max_slot;

    while (true) {
        if (table[slot].x == kEmpty) {
            return;
        }
        if (table[slot].x == key) {
            auto *const feat = feats + (num_feats + 1) * table[slot].y;
            for (int i = 0; i < num_feats; ++i) {
                atomicAdd(feat + i, point[i]);
            }
            atomicAdd(feat + num_feats, 1.0f);
            break;
        }
        slot = (slot + 1) & max_slot;
    }
}

void DynamicVoxelizationLauncher(const uint num_points,
                                 const uint num_feats,
                                 const uint max_slot,
                                 const int batch_id,
                                 const float3 voxel_size,
                                 const float3 coors_range_min,
                                 const float3 coors_range_max,
                                 const float *points,
                                 int4 *coors,
                                 float *feats,
                                 uint *num_voxels,
                                 uint2 *table) {
    const auto grid = BLOCKS1D(num_points);
    const auto block = THREADS();
    dynamic_voxelization_kernel<<<grid, block >>>(
            num_points,
            num_feats,
            max_slot,
            batch_id,
            voxel_size,
            coors_range_min,
            coors_range_max,
            points,
            coors,
            feats,
            num_voxels,
            table
    );
    dynamic_voxelization_kernel2<<<grid, block >>>(
            num_points,
            num_feats,
            max_slot,
            batch_id,
            voxel_size,
            coors_range_min,
            coors_range_max,
            points,
            coors,
            feats,
            num_voxels,
            table
    );
}

std::vector<at::Tensor> DynamicVoxelizationWrapper(const at::Tensor &points,
                                                   const std::vector<double> &voxel_size,
                                                   const std::vector<double> &coors_range,
                                                   long batch_id) {
    const int64_t num_points = points.size(0);
    const int64_t num_feats = points.size(1);
    const int64_t num_hash = get_table_size(num_points);

    auto table = points.new_empty({num_hash, 2}, TorchType<int>);
    auto num_voxels = points.new_zeros({1}, TorchType<int>);

    auto coors = points.new_empty({num_points, 4}, TorchType<int>);
    auto feats = points.new_zeros({num_points, num_feats + 1}, TorchType<float>);
    auto indices = points.new_empty({num_points, 1}, TorchType<int>);
    torch::zeros(static_cast<at::IntArrayRef>((uint *) table.data_ptr(), 1));
    table.fill_(at::Scalar(-1));

    auto voxel = float3{(float) voxel_size[0], (float) voxel_size[1], (float) voxel_size[2]};
    auto coors_range_min = float3{(float) coors_range[0], (float) coors_range[1], (float) coors_range[2]};
    auto coors_range_max = float3{(float) coors_range[3], (float) coors_range[4], (float) coors_range[5]};
    DynamicVoxelizationLauncher(num_points, num_feats,
                                (uint) num_hash - 1, batch_id,
                                voxel, coors_range_min, coors_range_max,
                                (float *) points.data_ptr(),
                                (int4 *) coors.data_ptr(),
                                (float *) feats.data_ptr(),
                                (uint *) num_voxels.data_ptr(),
                                (uint2 *) table.data_ptr());
    return {indices, coors, feats, num_voxels};
}

struct PointsList {
    float *const points;
    const ulong num_points;

    PointsList(float *const ptr, const uint n) : points(ptr), num_points(n) {}
};

__global__
void dyn_vox_batch_insert_hash_kernel(uint batch_size, const PointsList *const pl) {
    const uint bid = threadIdx.x;
    if (bid >= batch_size) {
        return;
    }
    printf("%u %lu\n", bid, pl[bid].num_points);
}

void DynamicVoxelizationBatchLauncher(const uint batch_size, const PointsList *const pl) {
    dyn_vox_batch_insert_hash_kernel<<<1, batch_size>>>(batch_size, pl);
}

inline auto prepare_input(const std::vector<at::Tensor> &points_list) {
    const auto &points1 = points_list[0];
    const uint batch_size = points_list.size();
    const long bytes = (long) (batch_size * sizeof(PointsList));
    auto dpl = points1.new_empty({bytes}, TorchType<uint8_t>);
    uint8_t hpl[bytes];
    for (int i = 0; i < batch_size; ++i) {
        new(hpl + i) PointsList((float *) points_list[i].data_ptr(), points_list[i].size(0));
    }
    cudaMemcpyAsync((void *) dpl.data_ptr(), (void *) hpl, bytes, cudaMemcpyHostToDevice);
    return dpl;
}


std::vector<at::Tensor> DynamicVoxelizationBatchWrapper(const std::vector<at::Tensor> &points_list,
                                                        const std::vector<double> &voxel_size,
                                                        const std::vector<double> &coors_range) {

    const uint batch_size = points_list.size();
    auto pl = prepare_input(points_list);
    DynamicVoxelizationBatchLauncher(batch_size, (PointsList *) pl.data_ptr());
}
