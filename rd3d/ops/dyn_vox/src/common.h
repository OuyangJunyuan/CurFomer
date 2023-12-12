#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>


#define THREAD_SIZE 256
#define BLOCKS1D(M) dim3(((M)+THREAD_SIZE-1)/THREAD_SIZE)
#define BLOCKS2D(M, B) dim3((((M)+THREAD_SIZE-1)/THREAD_SIZE),B)
#define THREADS() dim3(THREAD_SIZE)


constexpr uint32_t kEmpty = 0xffffffff;

template<typename T>
constexpr auto AsTorchType() {
    static_assert(sizeof(T) == 0, "unsupported");
    return at::ScalarType::Undefined;
}

template<>
constexpr auto AsTorchType<bool>() {
    return at::ScalarType::Bool;
}

template<>
constexpr auto AsTorchType<float>() {
    return at::ScalarType::Float;
}

template<>
constexpr auto AsTorchType<double>() {
    return at::ScalarType::Double;
}

template<>
constexpr auto AsTorchType<char>() {
    return at::ScalarType::Char;
}

template<>
constexpr auto AsTorchType<unsigned char>() {
    return at::ScalarType::Byte;
}

template<>
constexpr auto AsTorchType<short>() {
    return at::ScalarType::Short;
}

template<>
constexpr auto AsTorchType<int>() {
    return at::ScalarType::Int;
}

template<>
constexpr auto AsTorchType<long>() {
    return at::ScalarType::Long;
}

template<typename T>
constexpr auto TorchType = AsTorchType<T>();

__device__ __forceinline__
float3 operator+(const float3 a, const float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__
float3 operator-(const float3 a, const float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__
float3 operator/(const float3 a, const float3 b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ __forceinline__
float3 operator/(const float3 a, const float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ __forceinline__
uint32_t coord_hash_32(const int x, const int y, const int z) {
    uint32_t hash = 2166136261;
    hash ^= (uint32_t) (x + 10000);
    hash *= 16777619;
    hash ^= (uint32_t) (y + 10000);
    hash *= 16777619;
    hash ^= (uint32_t) (z + 10000);
    hash *= 16777619;
    return hash;
}

__device__ __forceinline__
uint32_t coord_hash_32(const int b, const int x, const int y, const int z) {
    uint32_t hash = 2166136261;
    hash ^= (uint32_t) (b + 10000);
    hash *= 16777619;
    hash ^= (uint32_t) (x + 10000);
    hash *= 16777619;
    hash ^= (uint32_t) (y + 10000);
    hash *= 16777619;
    hash ^= (uint32_t) (z + 10000);
    hash *= 16777619;
    return hash;
}



template<typename val_t>
struct HashTable {
    using key_t = uint;

    explicit HashTable(uint num_elems) : size(suitable_size(num_elems)), smax(size - 1) {};

    inline static uint suitable_size(uint num_elems, uint min_size = 2048) {
        auto table_size = std::max(2 * num_elems, min_size);
        return 1u << (uint) (ceilf(log2f((float) table_size)));
    }

    inline void from_blob(void *ptr) {
        data = (KeyValue *) ptr;
    }

    inline uint num_bytes() const {
        return size * sizeof(KeyValue);
    }


    __device__ __inline__
    uint32_t coord_hash_32(const uint x, const uint y, const uint z) const {
        uint32_t hash = 2166136261;
        hash ^= (uint32_t) x;
        hash *= 16777619;
        hash ^= (uint32_t) y;
        hash *= 16777619;
        hash ^= (uint32_t) z;
        hash *= 16777619;
        return hash & smax;
    }


    template<typename F1>
    __device__ __inline__ key_t insert(const key_t key, F1 f1) {
        key_t slot = key;
        while (true) {
            const key_t old = atomicCAS(&data[slot].key, kEmpty, key);
            if (old == kEmpty) {
                f1(data[slot].val);
                return slot;
            }
            if (old == key) {
                return slot;
            }
            slot = (slot + 1) & smax;
        }
    }

    // __device__ __inline__ val_t lookup(const key_t key) {
    //     uint32_t slot = key & smax;
    //     while (true) {
    //         const uint32_t old = atomicCAS(keys + slot, kEmpty, key);
    //         if (old == kEmpty) {
    //             return kEmpty;
    //         }
    //         if (old == key) {
    //             return vals[slot];
    //         }
    //         slot = (slot + 1) & smax;
    //     }
    // }

    struct KeyValue {
        key_t key;
        val_t val;
    };

    KeyValue *data{nullptr};
    const uint size{0};
    const uint smax{0};
};
