#include "../include/ops.h"
#include "../include/tensor.h"

// todo add checks for this

// view takes in contiguous tensors and outputs contiguous tensors only
FORWARD_FUNC_DEF(s_op_view_forward){
    (void) k0;
    for (size_t i = 0; i < 5; i++){
        kr->shape[i] = (size_t) k1->array[i];
    }
    set_contiguous_stride(kr);
}

BACKWARD_FUNC_DEF(s_op_view_backward){
    (void) k1; (void) kr; (void) idx;
    for (size_t i = 0; i < 5; i++){
        seed->shape[i] = (size_t) k0->shape[i];
    }
    set_contiguous_stride(seed);
    return seed;
}

FORWARD_FUNC_DEF(s_op_expand_forward){
    (void) kr;  (void) k0;  (void) k1;

}

//expand (the dimension that was expanded get reduced with a sum!) so next_seed has same shape
BACKWARD_FUNC_DEF(s_op_expand_backward){
    (void) kr;  (void) k0;  (void) k1;  (void) seed;   (void) idx;
    return NULL;
}

//permute literally permutes strides
FORWARD_FUNC_DEF(s_op_permute_forward){
    size_t temp_shape[5];
    uint64_t temp_stride[5];
    memcpy(temp_shape, k0->shape, 5 * sizeof(size_t));
    memcpy(temp_stride, k0->stride, 5 * sizeof(uint64_t));
    for (size_t i = 0; i < 5; i++){
        size_t idx = (size_t) k1->array[i];
        kr->shape[idx] = temp_shape[i];
        kr->stride[idx] = temp_stride[i];
    }
}

BACKWARD_FUNC_DEF(s_op_permute_backward){
    (void) kr;  (void) k1;  (void) idx;
    for (size_t i = 0; i < 5; i++){
        size_t idx = (size_t) k1->array[i];
        seed->shape[i] = k0->shape[idx];
        seed->stride[i] = k0->stride[idx];
    }
    return seed;
}

