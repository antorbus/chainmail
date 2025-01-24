#include "../../include/ops.h"
#include "../../include/tensor.h"

FORWARD_FUNC_DEF(u_op_exp_forward){
    (void) k1;
    KERNEL_TENSOR_5D_LOOP_START(kr){
        size_t offset_k0 = KERNEL_TENSOR_GET_OFFSET(k0);
        size_t offset_kr = KERNEL_TENSOR_GET_OFFSET(kr);
        kr->array[offset_kr] = expf(k0->array[offset_k0]);
    }
}

BACKWARD_FUNC_DEF(u_op_exp_backward){
    (void) k0; (void) k1; (void) idx;
    KERNEL_TENSOR_5D_LOOP_START(seed){
        size_t offset_seed = KERNEL_TENSOR_GET_OFFSET(seed);
        size_t offset_kr = KERNEL_TENSOR_GET_OFFSET(kr);
        seed->array[offset_seed] *= kr->array[offset_kr];
    }
    return seed;
}


FORWARD_FUNC_DEF(u_op_pow_forward){
    lemur_float x = k1->array[0];
    KERNEL_TENSOR_5D_LOOP_START(kr){
        size_t offset_k0 = KERNEL_TENSOR_GET_OFFSET(k0);
        size_t offset_kr = KERNEL_TENSOR_GET_OFFSET(kr);
        kr->array[offset_kr] = pow(k0->array[offset_k0], x);
    }
}

BACKWARD_FUNC_DEF(u_op_pow_backward){
    (void) kr; (void) idx;
    lemur_float x = k1->array[0];
    KERNEL_TENSOR_5D_LOOP_START(seed){
        size_t offset_seed = KERNEL_TENSOR_GET_OFFSET(seed);
        size_t offset_k0 = KERNEL_TENSOR_GET_OFFSET(k0);
        lemur_float val = x * pow(k0->array[offset_k0], x-1);
        seed->array[offset_seed] *= val;
    }
    return seed;
}

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
    KERNEL_TENSOR_5D_LOOP_START(kr){
        size_t offset_k0 = KERNEL_TENSOR_GET_OFFSET(k0);
        size_t offset_kr = KERNEL_TENSOR_GET_OFFSET(kr);
        kr->array[offset_kr] = 1.0 / (1.0 + expf(-1.0 * k0->array[offset_k0]));
    }
}

BACKWARD_FUNC_DEF(u_op_sigmoid_backward){
    (void) k1; (void) k0; (void) idx;
    KERNEL_TENSOR_5D_LOOP_START(seed){
        size_t offset_seed = KERNEL_TENSOR_GET_OFFSET(seed);
        size_t offset_kr = KERNEL_TENSOR_GET_OFFSET(kr);
        lemur_float sigmoid_val = kr->array[offset_kr];
        seed->array[offset_seed] *= sigmoid_val * (1.0 - sigmoid_val);
    }
    return seed;
}
