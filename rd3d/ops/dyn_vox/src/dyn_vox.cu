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


__device__ __inline__ uint4 voxelize(const uint bid, const float3 xyz, const VoxInfo vf) {
    return {bid,
            (uint) floor((xyz.x - vf.coors_min.x) / vf.voxel_size.x),
            (uint) floor((xyz.y - vf.coors_min.y) / vf.voxel_size.y),
            (uint) floor((xyz.z - vf.coors_min.z) / vf.voxel_size.z)};
}

template<class table_t>
__global__
void voxel_counting_kernel(const uint bid, const PointsInfo pf, const VoxInfo vf,
                           const float *__restrict__ points,
                           uint *__restrict__ count,
                           table_t table) {
    const uint pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= pf.num_point) {
        return;
    }
    const float3 xyz = *(float3 *) (points + pf.num_feat * pid);
    const uint key = voxel_hash(voxelize(bid, xyz, vf));
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
    const uint4 coor = voxelize(bid, xyz, vf);
    const auto vid = table.lookup(voxel_hash(coor));
    if (vid != table_t::EMPTY_HASH) {
        auto *point = (points + pf.num_feat * pid);
        auto *const feat = feats + (pf.num_feat + 1) * vid;
        const auto v_pid = atomicAdd(feat + pf.num_feat, 1.0f);
        for (int i = 0; i < pf.num_feat; ++i) {
            atomicAdd(feat + i, point[i]);
        }
        if (0 == v_pid) {
            coors[vid] = coor;
        }
    }
}

__global__
void feature_normalize_kernel(const uint num_voxel, const uint num_feat,
                              float *__restrict__ feats) {
    const uint vid = blockDim.x * blockIdx.x + threadIdx.x;
    if (vid >= num_voxel) {
        return;
    }
    auto *const feat = feats + (num_feat + 1) * vid;
    const float v_num_point = feat[num_feat];
    for (int i = 0; i < num_feat; ++i) {
        feat[i] /= v_num_point;
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

    auto count = points.new_zeros({1}, torch::ScalarType::Int);

    auto table = HashTable<uint>(num_point);
    auto table_data = points.new_full({table.bytes}, -1, torch::ScalarType::Char);
    table.from_blob(table_data.data_ptr());

    const auto grid = BLOCKS1D(num_point);
    const auto block = THREADS();
    voxel_counting_kernel<<<grid, block>>>((uint) batch_id, pf, vf,
                                           (float *) points.data_ptr(),
                                           (uint *) count.data_ptr(),
                                           table);
    auto num_voxel = count.item<int>();
    auto coors = points.new_empty({num_voxel, 4}, torch::ScalarType::Int);
    auto feats = points.new_zeros({num_voxel, num_feat + 1}, torch::ScalarType::Float);
    voxel_feats_scatter<<<grid, block>>>(
            (uint) batch_id, pf, vf,
            (float *) points.data_ptr(),
            (uint4 *) coors.data_ptr(),
            (float *) feats.data_ptr(),
            table
    );
    return {coors, feats, count};
}

std::vector<at::Tensor> DynamicVoxelizationBatchWrapper(const std::vector<at::Tensor> &points_list,
                                                        const std::vector<float> &voxel_size,
                                                        const std::vector<float> &coor_range) {
    const uint num_batch = points_list.size();
    const auto &points_0 = points_list[0];
    if (num_batch == 1) {
        return DynamicVoxelizationWrapper(points_0, voxel_size, coor_range, 0);
    }

    const uint num_feat = points_0.size(1);
    VoxInfo vf{{voxel_size[0], voxel_size[1], voxel_size[2]},
               {coor_range[0], coor_range[1], coor_range[2]},
               {coor_range[3], coor_range[4], coor_range[5]}};

    auto counts = points_0.new_zeros({num_batch}, torch::ScalarType::Int);

    uint cum_num_point = 0;
    for (const auto &points: points_list) {
        cum_num_point += points.size(0);
    }

    auto table = HashTable<uint>(cum_num_point);
    auto table_data = points_0.new_full({table.bytes}, -1, torch::ScalarType::Char);
    table.from_blob(table_data.data_ptr());
    for (uint bid = 0; bid < num_batch; ++bid) {
        const uint num_point = points_list[bid].size(0);
        PointsInfo pf{num_point, num_feat};
        voxel_counting_kernel<<<BLOCKS1D(num_point), THREADS()>>>(
                bid, pf, vf,
                (float *) points_list[bid].data_ptr(),
                ((uint *) counts.data_ptr()) + bid,
                table
        );
    }

    const auto h_count = counts.cpu();
    uint cum_num_voxel[num_batch + 1];
    for (uint bid = cum_num_voxel[0] = 0; bid < num_batch; ++bid) {
        cum_num_voxel[bid + 1] = cum_num_voxel[bid] + h_count[bid].item<int>();
    }
    const uint num_voxel = cum_num_voxel[num_batch];
    auto coors = points_0.new_empty({num_voxel, 4}, torch::ScalarType::Int);
    auto feats = points_0.new_zeros({num_voxel, num_feat + 1}, torch::ScalarType::Float);
    for (int bid = 0; bid < num_batch; ++bid) {
        const uint num_point = points_list[bid].size(0);
        PointsInfo pf{num_point, num_feat};
        voxel_feats_scatter<<<BLOCKS1D(num_point), THREADS()>>>(
                bid, pf, vf,
                (float *) points_list[bid].data_ptr(),
                ((uint4 *) coors.data_ptr()) + cum_num_voxel[bid],
                ((float *) feats.data_ptr()) + cum_num_voxel[bid] * (num_feat + 1),
                table
        );
    }
    feature_normalize_kernel<<<BLOCKS1D(num_voxel), THREADS()>>>(
            num_voxel, num_feat, (float *) feats.data_ptr()
    );
    return {coors, feats, counts};
}


