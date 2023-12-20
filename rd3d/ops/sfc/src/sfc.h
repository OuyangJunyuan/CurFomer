//
// Created by nrsl on 23-11-2.
//

#ifndef SIMPLECONCURRENTGPUHASHTABLE_SFC_H
#define SIMPLECONCURRENTGPUHASHTABLE_SFC_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>


#define USE_MAPPING_METHOD
#define THREADS_IN_BLOCK 256

#define BLOCKS1D(M) dim3(((M)+THREADS_IN_BLOCK-1)/THREADS_IN_BLOCK)
#define BLOCKS2D(M, B) dim3((((M)+THREADS_IN_BLOCK-1)/THREADS_IN_BLOCK),B)
#define THREADS() dim3(THREADS_IN_BLOCK)

#define GET_FUNC(f, ...) f
#define GET_ARGS(_, ...) __VA_ARGS__
#define SWITCH_CASE_I(z, n, args) case n: {GET_FUNC args <n##u><<<src_grid, block, 0, stream>>>(GET_ARGS args) ; break;}
#define UNROLL_SWITCH(k, n, args) switch (k) {BOOST_PP_REPEAT(n,SWITCH_CASE_I,args) \
                                              default: GET_FUNC args <n##u><<<src_grid,block,0,stream>>>(GET_ARGS args);}


#endif //SIMPLECONCURRENTGPUHASHTABLE_SFC_H
