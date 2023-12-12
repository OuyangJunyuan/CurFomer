#include <torch/extension.h>
#include <torch/serialize/tensor.h>

std::vector<at::Tensor> DynamicVoxelizationWrapper(const at::Tensor &points,
                                                   const std::vector<double> &voxel_size,
                                                   const std::vector<double> &coors_range,
                                                   long batch_id);

std::vector<at::Tensor> DynamicVoxelizationBatchWrapper(const std::vector<at::Tensor> &points_list,
                                                        const std::vector<double> &voxel_size,
                                                        const std::vector<double> &coors_range);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dynamic_voxelization", &DynamicVoxelizationWrapper, "dynamic voxelization");
    m.def("dynamic_voxelization_batch", &DynamicVoxelizationBatchWrapper, "dynamic voxelization");
}

TORCH_LIBRARY(havs_cuda, m) {
    m.def("dynamic_voxelization", &DynamicVoxelizationWrapper);
    m.def("dynamic_voxelization_batch", &DynamicVoxelizationBatchWrapper);
}