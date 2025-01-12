#include "../include/ops.h"
#include "../include/tensor.h"


tensor * unary_forward(int func, tensor * t0, bool retain_grad){
    kernel_tensor *k = empty_contiguous_kernel_tensor_like(t0->k);
    forward_func_table[func](k, t0->k, NULL);
    expression *comes_from = expression_from(func, t0, NULL);
    bool requires_grad = false;
    if (t0->requires_grad == true){
        requires_grad = true;
    }
    kernel_tensor *grad;
    if (retain_grad == false){
        grad = NULL;
    } else {
       grad = empty_contiguous_kernel_tensor_like(k);
       memset_kernel_tensor(grad, 0.0);
    }
    tensor *t = tensor_from(k, comes_from, requires_grad, grad);
    return t;
}

void u_op_relu_forward   ( kernel_tensor *kr, kernel_tensor *k0, kernel_tensor *k1){
    (void) k1;
    KERNEL_TENSOR_5D_LOOP_START(kr){
        size_t offset_k0 = KERNEL_TENSOR_GET_OFFSET(k0);
        size_t offset_kr = KERNEL_TENSOR_GET_OFFSET(kr);
        lemur_float v = k0->array[offset_k0];
        kr->array[offset_kr] = (v > 0) ? v : 0.0;
    }
}

kernel_tensor * u_op_relu_backward (kernel_tensor *kr, 
                                    kernel_tensor *k0, 
                                    kernel_tensor *k1, 
                                    kernel_tensor *seed,
                                    size_t idx){
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
