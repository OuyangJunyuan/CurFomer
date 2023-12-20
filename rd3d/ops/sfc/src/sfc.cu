#include "sfc.h"
#include <boost/preprocessor/repetition/repeat.hpp>
#include <cub/cub.cuh>

__device__ __inline__ uint8_t get_bit(const uint32_t bits, const uint32_t b) {
    return (bits & 1 << b) >> b;
}

#ifdef USE_MAPPING_METHOD

__device__ __inline__ uint8_t encode_code_transfer(const uint8_t state, const uint8_t order_code) {
    static const uint32_t compact_phm[12] = {
            1416045328, 270755668, 838944118, 1984233778, 1411606288, 275194708,
            1979794738, 843383158, 1382105968, 1883464018, 369567028, 371525428,
    };
    return (compact_phm[state] >> (order_code << 2)) & 15u;
}

__device__ __inline__ uint8_t encode_state_transfer(const uint8_t state, uint8_t order_code) {
    static const uint32_t compact_pnm[12] = {
            1398424392, 1498565442, 2858518641, 1620668603, 369498888, 1903204778,
            874813866, 1397928834, 1953798432, 1697744945, 543498427, 1953771946,
    };
    return (compact_pnm[state] >> (order_code << 2)) & 15u;  // compact_pnm[state] >> (code*4)
}

__device__ __inline__ uint8_t decode_code_transfer(const uint8_t state, const uint8_t order_code) {
    static const uint32_t compact_hpm[12] = {
            1165370128, 588268918, 270759493, 1984237603, 594953488, 1158685558,
            1982009413, 272987683, 359867968, 1930772518, 1176507253, 641734003,
    };
    return (compact_hpm[state] >> (order_code << 2)) & 15u;
}

__device__ __inline__ uint8_t decode_state_transfer(const uint8_t state, uint8_t order_code) {
    static const uint32_t compact_hnm[12] = {
            2773693512, 3024233817, 1904781846, 1620687623, 2971033608, 128620913,
            882485826, 2183484219, 662279232, 1748177285, 1342943782, 340436851,
    };
    return (compact_hnm[state] >> (order_code << 2)) & 15u;  // compact_pnm[state] >> (code*4)
}

#else

__device__ __inline__ uint8_t rotate_right(const uint8_t bits, uint8_t b) {
    b = b % 3u;
    return ((bits >> b) | (bits << (3u - b))) & 7u;
}

__device__ __inline__ uint8_t rotate_left(const uint8_t bits, uint8_t b) {
    return ((bits << b) | (bits >> (3 - b))) & 7u;
}

__device__ __inline__ uint8_t transformation(const uint8_t bits, const uint8_t entry, const uint8_t direction) {
    return rotate_right(bits ^ entry, direction + 1u);
}

__device__ __inline__ uint8_t inverse_transformation(const uint8_t bits, const uint8_t entry, const uint8_t direction) {
    return transformation(bits, rotate_right(entry, direction + 1u), 1u - direction);
}


__device__ __inline__ uint8_t inverse_graycode(const uint8_t graycode) {
    uint8_t res = graycode;
#pragma unroll
    for (int i = 1; i < 3; ++i) {
        res = res ^ (graycode >> i);
    }
    return res;
}

__device__ __inline__ uint8_t gray_code(const uint8_t bits) {
    return bits ^ (bits >> 1);
}

__device__ __inline__ uint8_t entry_point(const uint8_t bits) {
    if (bits) {
        return gray_code(((bits - 1u) >> 1) << 1);
    } else {
        return 0;
    }
    // return uint8_t(bool(bits)) * gray_code(((bits - 1) >> 1) << 1);
    // return ((uint8_t) (0) - uint8_t(bool(bits))) & gray_code(((bits - 1) >> 1) << 1);
}

__device__ __inline__ uint8_t direction_between(const uint8_t bits) {
    // return int(log2(gray_code(bits) ^ gray_code(bits + 1)));
    return 31u - __clz(gray_code(bits) ^ gray_code(bits + 1u));
}

__device__ __inline__ uint8_t direction_point(const uint8_t bits) {
    if (bits) {
        // return direction_between(bits - !bool(bits & 1u)) % 3;
        // if(bits%2){
        //     return direction_between(bits) % 3;
        // }else{
        //     return direction_between(bits -1) % 3;
        // }
        return direction_between(bits - 1u + (bits % 2u)) % 3u;
    } else {
        return 0;
    }
}

template<uint32_t order, class ...T>
__global__ void HilbertCurveEncodeKernel_(T &&... args) {
    HilbertCurveEncodeKernel_<order>(std::forward(args)...);
}

template<typename CoordType>
__device__ __inline__ uint8_t get_coord_bit(CoordType coord, int bit);

template<>
__device__ __inline__ uint8_t get_coord_bit<uint3>(const uint3 coord, const int bit) {
    return get_bit(coord.x, bit) + (get_bit(coord.y, bit) << 1) + (get_bit(coord.z, bit) << 2);
}

template<>
__device__ __inline__ uint8_t get_coord_bit<uint4>(const uint4 coord, const int bit) {
    return get_bit(coord.y, bit) + (get_bit(coord.z, bit) << 1) + (get_bit(coord.w, bit) << 2);

}

#endif

/***
 *
 * @tparam order
 * @tparam CoordType
 * @param num_coords
 * @param coords
 * @param codes
 * @note
 * references: \n
 * 1. http://pdebuyl.be/blog/2015/hilbert-curve.html \n
 * 2. https://lutanho.net/pic2html/draw_sfc.html \n
 */
template<uint32_t order>
__global__ void HilbertCurveEncodeKernel(const uint32_t num_coords,
                                         const uint4 *__restrict__ coords,
                                         uint64_t *__restrict__ codes) {
    const uint32_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_id >= num_coords) { return; }
    const uint4 coord = coords[thread_id];

    uint64_t code = coord.x;

#ifdef USE_MAPPING_METHOD
    /**
    * these effort have been try to adopted to address LUT memory bound. \n
    * 1. global memory. \n
    * 2. shared memory and use mat[12][32] to avoid bank conflict. \n
    * 3. local memory. \n
    * 4. tex1D. \n
    * 5. __constant__ is not suitable in this case since different threads in a warp do not access the same address. \n
    */
    uint8_t state = 0, bit_code;
#pragma  unroll
    for (int i = order - 1; i >= 0; --i) {
        bit_code = (get_bit(coord.y, i) << 2) + (get_bit(coord.z, i) << 1) + (get_bit(coord.w, i));
        code = (code << 3) | encode_code_transfer(state, bit_code);
        state = encode_state_transfer(state, bit_code);
    }
#else
    uint8_t entry = 0, direction = 2, label;
#pragma  unroll
    for (int i = order - 1; i >= 0; --i) {
        label = inverse_graycode(transformation(get_coord_bit(coord, i), entry, direction));
        entry = entry ^ (rotate_left(entry_point(label), direction + 1u));
        direction = (direction + direction_point(label) + 1u) % 3u;
        code = (code << 3) | label;
    }
#endif
    codes[thread_id] = code;
}


void HilbertCurveEncodeWrapper(at::Tensor &coords,
                               at::Tensor &codes,
                               int64_t order) {
    if (order > 20) {
        printf("launch sfc kernel with the order higher than 20\n");
    }
    const int64_t num_coords = coords.size(0);

    const auto src_grid = BLOCKS1D(num_coords);
    const auto block = THREADS();
    cudaStream_t stream = nullptr;
    UNROLL_SWITCH(order, 20, (HilbertCurveEncodeKernel, num_coords,
            (uint4 *) coords.data_ptr(),
            (uint64_t *) codes.data_ptr()))
}
/*

template<uint32_t order>
__global__ void HilbertCurveDecodeKernel(const uint32_t num_coords,
                                         const uint64_t *__restrict__ codes,
                                         uint3 *__restrict__ coords) {
    const uint32_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_id >= num_coords) { return; }
    const uint64_t code = codes[thread_id];

    uint3 coord{0, 0, 0};

#ifdef USE_MAPPING_METHOD
    uint8_t state = 0, order_code, bit_code;
#pragma unroll
    for (int i = order - 1; i >= 0; --i) {
        order_code = (code >> (i * 3u)) & 7u;
        bit_code = decode_code_transfer(state, order_code);
        coord.x |= ((bit_code >> 2) & 1u) << i;
        coord.y |= ((bit_code >> 1) & 1u) << i;
        coord.z |= (bit_code & 1u) << i;
        state = decode_state_transfer(state, order_code);
    }
#else
    uint8_t entry = 0, direction = 2;
    uint8_t xb, yb, zb, label, w;
#pragma unroll
    for (int i = order - 1; i >= 0; --i) {
        xb = get_bit(code, i * 3u);
        yb = get_bit(code, i * 3u + 1u);
        zb = get_bit(code, i * 3u + 2u);
        w = xb + (yb << 1) + (zb << 2);
        label = inverse_transformation(gray_code(w), entry, direction);
        coord.x += get_bit(label, 0u) << i;
        coord.y += get_bit(label, 1u) << i;
        coord.z += get_bit(label, 2u) << i;
        entry = entry ^ (rotate_left(entry_point(w), direction + 1u));
        direction = (direction + direction_point(w) + 1u) % 3u;
    }
#endif
    coords[thread_id] = coord;
}

void HilbertCurveDecodeWrapper(at::Tensor &codes,
                               at::Tensor &coords,
                               int64_t order) {
    if (order > 20) {
        printf("launch sfc kernel with the order higher than 20\n");
    }
    const int64_t num_coords = coords.size(0);

    const auto src_grid = BLOCKS1D(num_coords);
    const auto block = THREADS();
    cudaStream_t stream = nullptr;
    UNROLL_SWITCH(order, 20, (HilbertCurveDecodeKernel, num_coords,
            (uint64_t *) codes.data_ptr(),
            (uint3 *) coords.data_ptr()))
}

void ArgSortHilbertWrapper(at::Tensor &coords,
                           at::Tensor &key,
                           at::Tensor &val,
                           at::Tensor &key_out,
                           at::Tensor &val_out,
                           at::Tensor &bs_offsets,
                           int64_t order) {
    HilbertCurveEncodeWrapper(coords, key, order);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    auto d_keys_in = (uint64_t *) key.data_ptr();
    auto d_values_in = (uint64_t *) val.data_ptr();
    auto d_keys_out = (uint64_t *) key_out.data_ptr();
    auto d_values_out = (uint64_t *) val_out.data_ptr();
    auto d_offsets = (uint64_t *) bs_offsets.data_ptr();
    const int num_items = coords.size(0);
    const int num_segments = bs_offsets.size(0) - 1;
    cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                             d_keys_in, d_keys_out, d_values_in, d_values_out,
                                             num_items, num_segments, d_offsets, d_offsets + 1, 0, order * 3);
    auto temp_data = coords.new_empty({static_cast<long>(temp_storage_bytes)}, torch::ScalarType::Char);
    // Allocate temporary storage
    d_temp_storage = temp_data.data_ptr();
    // radix sort is slow.
    cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                             d_keys_in, d_keys_out, d_values_in, d_values_out,
                                             num_items, num_segments, d_offsets, d_offsets + 1, 0, order * 3);
}

 */

__global__ void indices_padding_kernel(const uint num_type,
                                       const uint n_index,
                                       const uint *__restrict__ batch_end,
                                       const ulong *__restrict__ indices,
                                       const uint n_pindex,
                                       const uint *__restrict__ padded_batch_end,
                                       ulong *__restrict__ padded_indices) {
    const uint bid = blockIdx.x / num_type;
    const uint dim = blockIdx.x % num_type;
    indices += dim * n_index;
    padded_indices += dim * n_pindex;

    const uint begin = bid ? batch_end[bid - 1] : 0;
    const uint num = batch_end[bid] - begin;
    const uint padded_begin = bid ? padded_batch_end[bid - 1] : 0;
    const uint padded_num = padded_batch_end[bid] - padded_begin;

    const float scale = (float) num / (float) padded_num;
    for (uint i = threadIdx.x; i < padded_num; i += THREADS_IN_BLOCK) {
        padded_indices[padded_begin + i] = indices[begin + (uint) (roundf((float) i * scale))];
    }
}

__global__ void indices_mapping_kernel(const uint ntype,
                                       const uint npindex,
                                       const ulong *__restrict__ pindices,
                                       ulong *__restrict__ mindices,
                                       ulong *__restrict__ temp) {

    const uint trg_dim = blockIdx.x;
    const uint src_dim = trg_dim ? trg_dim - 1 : ntype - 1;

    const auto *src_indices = pindices + src_dim * npindex;
    const auto *trg_indices = pindices + trg_dim * npindex;
    auto *mapping = mindices + src_dim * npindex;
    auto *this_temp = temp + src_dim * npindex;
    for (uint i = threadIdx.x; i < npindex; i += THREADS_IN_BLOCK) {
        this_temp[src_indices[i]] = i;
    }
    __syncthreads();
    for (uint i = threadIdx.x; i < npindex; i += THREADS_IN_BLOCK) {
        mapping[i] = this_temp[trg_indices[i]];
    }
}

std::vector<torch::Tensor>
IndicesGroupingBatchWrapper(const at::Tensor &batch_end, const at::Tensor &indices,
                            const at::Tensor &padded_batch_end, at::Tensor &padded_indices) {
    const uint num_batch = batch_end.size(0);
    const uint num_type = indices.size(0);
    const uint num_index = indices.size(1);
    const uint num_padded_index = padded_indices.size(1);

    indices_padding_kernel<<<num_batch * num_type, THREADS()>>>(
            num_type,
            num_index, (uint *) batch_end.data_ptr(), (ulong *) indices.data_ptr(),
            num_padded_index, (uint *) padded_batch_end.data_ptr(), (ulong *) padded_indices.data_ptr()
    );

    auto temp = indices.new_empty({num_type, num_padded_index}, torch::ScalarType::Long);
    indices_mapping_kernel<<<num_type, THREADS()>>>(
            num_type, num_padded_index,
            (ulong *) padded_indices.data_ptr(),
            ((ulong *) padded_indices.data_ptr()) + num_type * num_padded_index,
            (ulong *) temp.data_ptr()
    );
    return {temp};
}