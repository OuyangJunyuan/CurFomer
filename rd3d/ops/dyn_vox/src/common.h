#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>


#define THREAD_SIZE 256


#define BLOCKS1D(M) dim3(((M)+THREAD_SIZE-1)/THREAD_SIZE)
#define THREADS() dim3(THREAD_SIZE)


__device__
inline static uint32_t voxel_hash(const uint4 coor) {
    uint32_t hash = 2166136261;
    hash ^= coor.x;
    hash *= 16777619;
    hash ^= coor.y;
    hash *= 16777619;
    hash ^= coor.z;
    hash *= 16777619;
    hash *= coor.w;
    hash *= 16777619;
    return hash;
}

inline static uint suitable_size(uint num_elems, uint min_size = 2048) {
    auto table_size = std::max(2 * num_elems, min_size);
    return 1u << (uint) (ceilf(log2f((float) table_size)));
}


template<typename val_t>
struct HashTable {
    using hash_t = uint;
    constexpr static hash_t EMPTY_HASH = std::numeric_limits<hash_t>::max();


    template<typename F1>
    __device__ __inline__ void insert(const hash_t hash, F1 f1) {
        hash_t slot = hash & (size - 1);
        while (true) {
            const hash_t exist_hash = atomicCAS(&table[slot].hash, EMPTY_HASH, hash);
            if (exist_hash == EMPTY_HASH) {
                f1(table[slot].val);
                return;
            }
            if (exist_hash == hash) {
                return;
            }
            slot = (slot + 1) & (size - 1);
        }
    }

    __device__ __inline__ val_t lookup(const hash_t hash) {
        hash_t slot = hash & (size - 1);
        while (true) {
            const hash_t exist_hash = table[slot].hash;
            if (exist_hash == hash) {
                return table[slot].val;
            }
            if (exist_hash == EMPTY_HASH) {
                return EMPTY_HASH;
            }
            slot = (slot + 1) & (size - 1);
        }
    }

    inline static uint suitable_size(uint num_elems, uint min_size = 2048) {
        auto table_size = std::max(2 * num_elems, min_size);
        return 1u << (uint) (ceilf(log2f((float) table_size)));
    }

    inline void from_blob(void *ptr) { table = (KeyValue *) ptr; }

    explicit HashTable(uint num_elems) :
            size(suitable_size(num_elems)), bytes(sizeof(KeyValue) * size) {};

    struct KeyValue {
        hash_t hash;
        val_t val;
    } *table{nullptr};

    const uint size{0};
    const uint bytes{0};
};
