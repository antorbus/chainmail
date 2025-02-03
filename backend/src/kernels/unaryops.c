#include "../../include/tensor.h"

FORWARD_FUNC_DEF(u_op_exp_forward){
    (void) k1;
    UNARY_CONTIGUOUS_ELEMENTWISE_OP_SIMD(kr, k0, expf);
}

BACKWARD_FUNC_DEF(u_op_exp_backward){
    (void) k0; (void) k1; (void) idx;
    BINARY_CONTIGUOUS_ELEMENTWISE_OP_SIMD(seed, seed, kr, _mul);
    return seed;
}

FORWARD_FUNC_DEF(u_op_pow_forward){
    lemur_float x = k1->array[0];
    if (kr->length > 1<<17) {                                                 
        #pragma omp parallel for simd                                     
          for (size_t _i = 0; _i < kr->length; _i++) {                       
            kr->array[_i] = powf(k0->array[_i], x);   
          }                                                                    
    } else {                                                                     
        #pragma omp simd                                                  
        for (size_t _i = 0; _i < kr->length; _i++) {                       
            kr->array[_i] = powf(k0->array[_i], x);   
        }                                                                        
    }  
}

BACKWARD_FUNC_DEF(u_op_pow_backward){
    (void) kr; (void) idx;
    lemur_float x = k1->array[0];
    if (kr->length > 1<<17) {                                                 
        #pragma omp parallel for simd                                     
          for (size_t _i = 0; _i < kr->length; _i++) {
            lemur_float val = x * powf(k0->array[_i], x-1.0);                       
            seed->array[_i] *= val;
          }                                                                    
    } else {                                                                     
        #pragma omp simd                                                  
         for (size_t _i = 0; _i < kr->length; _i++) {
            lemur_float val = x * powf(k0->array[_i], x-1.0);                       
            seed->array[_i] *= val;
          }                                                                          
    }  
    return seed;
}

FORWARD_FUNC_DEF(u_op_relu_forward){
    (void) k1;
    UNARY_CONTIGUOUS_ELEMENTWISE_OP_SIMD(kr, k0, _relu);
}

BACKWARD_FUNC_DEF(u_op_relu_backward){
    (void) k0; (void) k1; (void) idx;
    KERNEL_TENSOR_5D_LOOP_START(seed){
        size_t offset_seed = KERNEL_TENSOR_GET_OFFSET(seed);
        size_t offset_kr = KERNEL_TENSOR_GET_OFFSET(kr);
        lemur_float val= (kr->array[offset_kr] == 0.0) ? 0.0 : 1.0;  
        seed->array[offset_seed] *= val;
    }
    return seed;
}

FORWARD_FUNC_DEF(u_op_sigmoid_forward){
    (void) k1;
    UNARY_CONTIGUOUS_ELEMENTWISE_OP_SIMD(kr, k0, _sigmoid);
}

BACKWARD_FUNC_DEF(u_op_sigmoid_backward){
    (void) k1; (void) k0; (void) idx;
    UNARY_CONTIGUOUS_ELEMENTWISE_OP_SIMD(seed, kr, _sigmoid_grad);
    return seed;
}

FORWARD_FUNC_DEF(u_op_log_forward){
    (void) k1;
    UNARY_CONTIGUOUS_ELEMENTWISE_OP_SIMD(kr, k0, logf);
}

BACKWARD_FUNC_DEF(u_op_log_backward) {
    (void) idx; (void) k1; (void) kr;
    b_op_division_forward(seed, seed, k0);
    return seed;
}

FORWARD_FUNC_DEF(u_op_neg_forward){
    (void) k1;
    KERNEL_TENSOR_5D_LOOP_START(kr){
        size_t offset_k0 = KERNEL_TENSOR_GET_OFFSET(k0);
        size_t offset_kr = KERNEL_TENSOR_GET_OFFSET(kr);
        kr->array[offset_kr] = -1.0 * k0->array[offset_k0];
    }
}

BACKWARD_FUNC_DEF(u_op_neg_backward){
    (void) k1; (void) k0; (void) kr; (void) idx;
    KERNEL_TENSOR_5D_LOOP_START(seed){
        size_t offset_seed = KERNEL_TENSOR_GET_OFFSET(seed);
        seed->array[offset_seed] *= -1.0;
    }
    return seed;
}

FORWARD_FUNC_DEF(u_op_sqrt_forward){
    (void) k1;
    UNARY_CONTIGUOUS_ELEMENTWISE_OP_SIMD(kr, k0, sqrtf);
}

BACKWARD_FUNC_DEF(u_op_sqrt_backward){
    (void) k1; (void) k0; (void) kr; (void) idx;
    KERNEL_TENSOR_5D_LOOP_START(seed){
        size_t offset_seed = KERNEL_TENSOR_GET_OFFSET(seed);
        size_t offset_kr = KERNEL_TENSOR_GET_OFFSET(kr);
        lemur_float sqrt_val = kr->array[offset_kr];
        seed->array[offset_seed] *= 1.0 / (2 * sqrt_val);
    }
    return seed;
}

FORWARD_FUNC_DEF(u_op_abs_forward){
    (void) k1;
    UNARY_CONTIGUOUS_ELEMENTWISE_OP_SIMD(kr, k0, fabsf);
}

BACKWARD_FUNC_DEF(u_op_abs_backward){
    (void) k1; (void) idx;
    KERNEL_TENSOR_5D_LOOP_START(seed){
        size_t offset_seed = KERNEL_TENSOR_GET_OFFSET(seed);
        size_t offset_k0 = KERNEL_TENSOR_GET_OFFSET(k0);
        size_t offset_kr = KERNEL_TENSOR_GET_OFFSET(kr);
        seed->array[offset_seed] *= k0->array[offset_k0] / kr->array[offset_kr];
    }
    return seed;
}

FORWARD_FUNC_DEF(u_op_sign_forward){
    (void) k1;
    KERNEL_TENSOR_5D_LOOP_START(kr){
        size_t offset_k0 = KERNEL_TENSOR_GET_OFFSET(k0);
        size_t offset_kr = KERNEL_TENSOR_GET_OFFSET(kr);
        lemur_float v = k0->array[offset_k0];
        kr->array[offset_kr] = (v > 0) - (v < 0);
    }
}

BACKWARD_FUNC_DEF(u_op_sign_backward) {
    (void) k1; (void) k0; (void) kr; (void) idx;
    KERNEL_TENSOR_5D_LOOP_START(seed){
        size_t offset_seed = KERNEL_TENSOR_GET_OFFSET(seed);
        seed->array[offset_seed] *= 0;
    }
    return seed;
}

FORWARD_FUNC_DEF(u_op_reciprocal_forward){
    (void) k1;
    KERNEL_TENSOR_5D_LOOP_START(kr){
        size_t offset_k0 = KERNEL_TENSOR_GET_OFFSET(k0);
        size_t offset_kr = KERNEL_TENSOR_GET_OFFSET(kr);
        kr->array[offset_kr] = 1.0 / k0->array[offset_k0];
    }
}

BACKWARD_FUNC_DEF(u_op_reciprocal_backward){
    (void) k1; (void) k0; (void) idx;
    KERNEL_TENSOR_5D_LOOP_START(seed){
        size_t offset_seed = KERNEL_TENSOR_GET_OFFSET(seed);
        size_t offset_kr = KERNEL_TENSOR_GET_OFFSET(kr);
        seed->array[offset_seed] *= -1.0 * (kr->array[offset_kr] * kr->array[offset_kr]);
    }
    return seed;
}
