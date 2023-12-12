#include <cuda.h>
#include <cub/cub.cuh>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "common.h"

struct PointsInfo {
    const uint num_point;
    const uint num_feat;
};

struct VoxInfo {
    const float3 voxel_size;
    const float3 coors_min;
    const float3 coors_max;
};

inline bool check_valid(const float3 xyz) {
    return not(isnan(xyz.x) or isnan(xyz.y) or isnan(xyz.z)) or (isnan(xyz.x) or isnan(xyz.y) or isnan(xyz.z));
}

template<class table_t>
__global__
void voxel_counting_kernel(const PointsInfo pf, const VoxInfo vf,
                           const float *__restrict__ points,
                           uint *__restrict__ count,
                           table_t table) {
    const uint pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= pf.num_point) {
        return;
    }
    const float3 xyz = *(float3 *) (points + pf.num_feat * pid);
    const uint key = table.coord_hash_32((uint) floor((xyz.x - vf.coors_min.x) / vf.voxel_size.x),
                                         (uint) floor((xyz.y - vf.coors_min.y) / vf.voxel_size.y),
                                         (uint) floor((xyz.z - vf.coors_min.z) / vf.voxel_size.z));
    table.insert(key, [&](auto &val) {
        val = atomicAdd(count, 1);
    });
}

template<class table_t>
__global__
void voxel_feats_scatter(const uint bid, const PointsInfo pf, const VoxInfo vf,
                         const float *__restrict__ points,
                         uint4 *__restrict__ coors,
                         float *__restrict__ feats,
                         table_t table) {
    const uint pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= pf.num_point) {
        return;
    }
    const float3 xyz = *(float3 *) (points + pf.num_feat * pid);
    if ((xyz.x < vf.coors_min.x or xyz.x >= vf.coors_max.x) or
        (xyz.y < vf.coors_min.y or xyz.y >= vf.coors_max.y) or
        (xyz.z < vf.coors_min.z or xyz.z >= vf.coors_max.z)) {
        return;
    }
    uint4 coor{bid,
               (uint) floor((xyz.x - vf.coors_min.x) / vf.voxel_size.x),
               (uint) floor((xyz.y - vf.coors_min.y) / vf.voxel_size.y),
               (uint) floor((xyz.z - vf.coors_min.z) / vf.voxel_size.z)};

    const uint key = table.coord_hash_32(coor.y, coor.z, coor.w);
    const auto vid = table.lookup(key);
    if (vid == kEmpty) {
        return;
    }
    auto *point = (points + pf.num_feat * pid);
    auto *const feat = feats + (pf.num_feat + 1) * vid;
    for (int i = 0; i < pf.num_feat; ++i) {
        atomicAdd(feat + i, point[i]);
    }
    if (0 == atomicAdd(feat + pf.num_feat, 1.0f)) {
        coors[vid] = coor;
    }
}

std::vector<at::Tensor> DynamicVoxelizationWrapper(const at::Tensor &points,
                                                   const std::vector<float> &voxel_size,
                                                   const std::vector<float> &coor_range,
                                                   uint batch_id) {

    const uint num_point = points.size(0);
    const uint num_feat = points.size(1);

    PointsInfo pf{num_point, num_feat};
    VoxInfo vf{{voxel_size[0], voxel_size[1], voxel_size[2]},
               {coor_range[0], coor_range[1], coor_range[2]},
               {coor_range[3], coor_range[4], coor_range[5]}};

    HashTable<uint> table(num_point);
    auto count = points.new_zeros({1}, torch::ScalarType::Int);
    auto coors = points.new_empty({num_point, 4}, torch::ScalarType::Int);
    auto feats = points.new_zeros({num_point, num_feat + 1}, torch::ScalarType::Float);
    auto table_data = points.new_empty({table.num_bytes()}, torch::ScalarType::Char);
    table_data.fill_(-1);
    table.from_blob(table_data.data_ptr());

    const auto grid = BLOCKS1D(num_point);
    const auto block = THREADS();
    voxel_counting_kernel<<<grid, block>>>(pf, vf,
                                           (float *) points.data_ptr(),
                                           (uint *) count.data_ptr(),
                                           table);
    voxel_feats_scatter<<<grid, block>>>(
            (uint) batch_id, pf, vf,
            (float *) points.data_ptr(),
            (uint4 *) coors.data_ptr(),
            (float *) feats.data_ptr(),
            table
    );
    return {coors, feats, count};
}

