#include <torch/extension.h>
#include <torch/serialize/tensor.h>

std::tuple<at::Tensor, at::Tensor, at::Tensor>
DynamicVoxelizationWrapper(const at::Tensor &points,
                           const std::vector<float> &voxel_size,
                           const std::vector<float> &coor_range,
                           uint batch_id = -1);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
DynamicVoxelizationBatchWrapper(const std::vector<at::Tensor> &points_list,
                                const std::vector<float> &voxel_size,
                                const std::vector<float> &coor_range);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dynamic_voxelization", &DynamicVoxelizationWrapper, " ");
    m.def("dynamic_voxelization_batch", &DynamicVoxelizationBatchWrapper, " ");
}
