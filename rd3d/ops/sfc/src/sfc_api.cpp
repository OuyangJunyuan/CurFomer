#include <torch/extension.h>
#include <torch/serialize/tensor.h>


void HilbertCurveEncodeWrapper(at::Tensor &coords,
                               at::Tensor &codes,
                               int64_t order);

void HilbertCurveDecodeWrapper(at::Tensor &codes,
                               at::Tensor &coords,
                               int64_t order);

void ArgSortHilbertWrapper(at::Tensor &coords,
                           at::Tensor &key,
                           at::Tensor &val,
                           at::Tensor &key_out,
                           at::Tensor &val_out,
                           at::Tensor &bs_offsets,
                           int64_t order);

std::vector<torch::Tensor>
IndicesGroupingBatchWrapper(const at::Tensor &batch_end, const at::Tensor &indices,
                            const at::Tensor &padded_batch_end, at::Tensor &padded_indices);
/**
 * @brief TORCH_EXTENSION_NAME will be automatically replaced by the extension name writen in "setup.py", i.e., havs_cuda.
 * @example from havs import havs_cuda
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hilbert_curve_encoder", &HilbertCurveEncodeWrapper,
          "transform the 3-D space coordinates to 1-D hilbert coordinates");
    // m.def("hilbert_curve_decoder", &HilbertCurveDecodeWrapper,
    //       "transform the 1-D hilbert coordinates to 3-D space coordinates");
    // m.def("argsort_hilbert", &ArgSortHilbertWrapper,
    //       "transform the 3-D space coordinates to 1-D hilbert coordinates and conduct argsort");
    m.def("indices_grouping_batch", &IndicesGroupingBatchWrapper, " ");
}
/**
 * @brief Register to TorchScript operation set.
 * @example cuda_spec = importlib.machinery.PathFinder().find_spec(library, [os.path.dirname(__file__)])
 * @example torch.ops.load_library(cuda_spec.origin)
 */
TORCH_LIBRARY(sfc_cuda, m) {
    m.def("hilbert_curve_encoder", &HilbertCurveEncodeWrapper);
    // m.def("hilbert_curve_decoder", &HilbertCurveDecodeWrapper);
    // m.def("argsort_hilbert", &ArgSortHilbertWrapper);
}