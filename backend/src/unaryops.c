#include "../include/ops.h"
#include "../include/tensor.h"

FORWARD_FUNC_DEF(u_op_relu_forward){
    (void) k1;
    KERNEL_TENSOR_5D_LOOP_START(kr){
        size_t offset_k0 = KERNEL_TENSOR_GET_OFFSET(k0);
        size_t offset_kr = KERNEL_TENSOR_GET_OFFSET(kr);
        lemur_float v = k0->array[offset_k0];
        kr->array[offset_kr] = (v > 0) ? v : 0.0;
    }
}

BACKWARD_FUNC_DEF(u_op_relu_backward){
    (void) k0; (void) k1; (void) idx;
    kernel_tensor *next_seed = empty_kernel_tensor_like(seed);
    KERNEL_TENSOR_5D_LOOP_START(next_seed){
        size_t offset_next_seed = KERNEL_TENSOR_GET_OFFSET(next_seed);
        size_t offset_seed = KERNEL_TENSOR_GET_OFFSET(seed);
        size_t offset_kr = KERNEL_TENSOR_GET_OFFSET(kr);
        lemur_float val= (kr->array[offset_kr] == 0.0) ? 0.0 : 1.0;  
        next_seed->array[offset_next_seed] = seed->array[offset_seed] * val;
    }
    return next_seed;
}
