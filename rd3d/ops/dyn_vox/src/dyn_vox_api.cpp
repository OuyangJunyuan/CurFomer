#include <torch/extension.h>
#include <torch/serialize/tensor.h>

std::vector<at::Tensor> DynamicVoxelizationWrapper(const at::Tensor &points,
                                                   const std::vector<float> &voxel_size,
                                                   const std::vector<float> &coor_range,
                                                   uint batch_id = -1);

std::vector<at::Tensor> DynamicVoxelizationBatchWrapper(const std::vector<at::Tensor> &points_list,
                                                        const std::vector<float> &voxel_size,
                                                        const std::vector<float> &coor_range);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dynamic_voxelization", &DynamicVoxelizationWrapper, "dynamic voxelization");
    m.def("dynamic_voxelization_batch", &DynamicVoxelizationBatchWrapper, "dynamic voxelization");
}
