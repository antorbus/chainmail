#include "../include/ops.h"
#include "../include/tensor.h"

// todo add checks for this

// view takes in contiguous tensors and outputs contiguous tensors only
FORWARD_FUNC_DEF(s_op_view_forward){
    //view only works for contiguous tensors
    (void) k0;
    for (size_t i = 0; i < 5; i++){
        kr->shape[i] = (size_t) k1->array[i];
    }
    set_contiguous_stride(kr);
}

BACKWARD_FUNC_DEF(s_op_view_backward){
    (void) k1; (void) kr; (void) idx;
    inplace_contiguous_kernel_tensor(seed); 
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
    for (size_t i = 0; i < 5; i++){
        size_t idx = (size_t) k1->array[i];
        kr->shape[idx] = k0->shape[i];
        kr->stride[idx] = k0->stride[i];
    }
}

BACKWARD_FUNC_DEF(s_op_permute_backward){
    (void) k0; (void) idx; (void) kr;
    size_t temp_shape[5];
    memcpy(temp_shape, seed->shape, 5*sizeof(size_t));
    int64_t temp_stride[5];
    memcpy(temp_stride, seed->stride, 5*sizeof(int64_t));
    for (size_t i = 0; i < 5; i++){
        size_t idx = (size_t) k1->array[i];
        seed->shape[idx] = temp_shape[i];
        seed->stride[idx] = temp_stride[i];
    }
    inplace_contiguous_kernel_tensor(seed); 
    return seed;
}

