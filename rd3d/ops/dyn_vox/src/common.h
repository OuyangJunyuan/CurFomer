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


inline auto get_table_size(int64_t nums, int64_t min = 2048) {
    auto table_size = nums * 2 > min ? nums * 2 : min;
    table_size = 2 << ((int64_t) ceilf((logf(static_cast<float>(table_size)) / logf(2.0f))) - 1);
    return table_size;
}

template<typename KeyType, typename ValType>
struct HashTable {
    HashTable(KeyType *key_ptr, ValType *val_ptr, uint table_size) :
            keys(key_ptr), vals(val_ptr), size(table_size), smax(table_size - 1) {};

    template<typename F1, typename F2>
    __device__ __inline__ void insert(const KeyType key, F1 f1, F2 f2) {
        uint32_t slot = key & smax;
        while (true) {
            const uint32_t old = atomicCAS(keys + slot, kEmpty, key);
            if (old == kEmpty) {
                f1(vals[slot]);
                return;
            }
            if (old == key) {
                f2(vals[slot]);
                return;
            }
            slot = (slot + 1) & smax;
        }
    }

    template<typename F1>
    __device__ __inline__ void insert(const KeyType key, F1 f1) {
        uint32_t slot = key & smax;
        while (true) {
            const uint32_t old = atomicCAS(keys + slot, kEmpty, key);
            if (old == kEmpty) {
                f1(vals[slot]);
                return;
            }
            slot = (slot + 1) & smax;
        }
    }

    __device__ __inline__ ValType lookup(const KeyType key) {
        uint32_t slot = key & smax;
        while (true) {
            const uint32_t old = atomicCAS(keys + slot, kEmpty, key);
            if (old == kEmpty) {
                return kEmpty;
            }
            if (old == key) {
                return vals[slot];
            }
            slot = (slot + 1) & smax;
        }
    }

    KeyType *const keys;
    ValType *const vals;
    const uint size, smax;
};
