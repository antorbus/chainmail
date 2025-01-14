#include "../include/ops.h"
#include "../include/tensor.h"

void b_op_add_forward(kernel_tensor *kr, kernel_tensor *k0, kernel_tensor *k1){
    KERNEL_TENSOR_5D_LOOP_START(kr){
        size_t offset_k0 = KERNEL_TENSOR_GET_OFFSET(k0);
        size_t offset_k1 = KERNEL_TENSOR_GET_OFFSET(k1);
        size_t offset_kr = KERNEL_TENSOR_GET_OFFSET(kr);
        kr->array[offset_kr] = k0->array[offset_k0] + k1->array[offset_k1];
    }
}

kernel_tensor * b_op_add_backward(  kernel_tensor *kr, 
                                    kernel_tensor *k0, 
                                    kernel_tensor *k1, 
                                    kernel_tensor *seed,
                                    size_t idx){
    //TODO: Could do something like only create it for right nodes to save mallocs
    (void) kr; (void) k0; (void) k1; (void) idx;
    kernel_tensor *next_seed = empty_kernel_tensor_like(seed); 
    memcpy(next_seed->array, seed->array, seed->length * sizeof(lemur_float));
    return next_seed;
}

void b_op_mul_forward(kernel_tensor *kr, kernel_tensor *k0, kernel_tensor *k1){
    KERNEL_TENSOR_5D_LOOP_START(kr){
        size_t offset_k0 = KERNEL_TENSOR_GET_OFFSET(k0);
        size_t offset_k1 = KERNEL_TENSOR_GET_OFFSET(k1);
        size_t offset_kr = KERNEL_TENSOR_GET_OFFSET(kr);
        kr->array[offset_kr] = k0->array[offset_k0] * k1->array[offset_k1];
    }
}

kernel_tensor * b_op_mul_backward(  kernel_tensor *kr, 
                                    kernel_tensor *k0, 
                                    kernel_tensor *k1, 
                                    kernel_tensor *seed,
                                    size_t idx){
    (void) kr;
    kernel_tensor *next_seed = empty_contiguous_kernel_tensor_like(seed);
    kernel_tensor *k = (idx == 0) ? k1 : k0;
    KERNEL_TENSOR_5D_LOOP_START(next_seed){
        size_t offset_next_seed = KERNEL_TENSOR_GET_OFFSET(next_seed);
        size_t offset_seed = KERNEL_TENSOR_GET_OFFSET(seed);
        size_t offset_k = KERNEL_TENSOR_GET_OFFSET(k);
        next_seed->array[offset_next_seed] = seed->array[offset_seed] * k->array[offset_k];
    }
    return next_seed;
}
