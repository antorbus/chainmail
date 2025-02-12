#include "../../include/tensor.h"

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
    for (size_t i = 0; i < 5; i++){
        seed->shape[i] = (size_t) k0->shape[i];
    }
    set_contiguous_stride(seed);
    return seed;
}

FORWARD_FUNC_DEF(s_op_expand_forward){
    (void) k0;  
    for (size_t i = 0; i < 5; i++){
        if ((kr->shape[i] == 1) && (1 != (size_t) k1->array[i])){
            kr->stride[i] = 0;
        }
        kr->shape[i] = (size_t) k1->array[i];
    } 
}

//expand (the dimension that was expanded get reduced with a sum!) so next_seed has same shape
BACKWARD_FUNC_DEF(s_op_expand_backward){
    (void) k1; (void) idx;
    
    lemur_float dim_arr[5];
    kernel_tensor dims;
    dims.array = dim_arr;
    dims.length = 5;
    dims.shallow = false;

    dims.shape[0] = 1;
    dims.shape[1] = 1;
    dims.shape[2] = 1;
    dims.shape[3] = 1;
    dims.shape[4] = 5;

    set_contiguous_stride(&dims);

    for (size_t i = 0; i < 5; i++){
        if (k0->shape[i] == kr->shape[i]){
            dim_arr[i] = 1;
        } else {
            dim_arr[i] = 0;
        }
    } 
    kernel_tensor *next_seed = empty_contiguous_kernel_tensor(k0->shape);
    forward_func_table[OP_SUM](next_seed, seed, &dims);
    return next_seed;
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
        size_t _idx = (size_t) k1->array[i];
        seed->shape[i] = temp_shape[_idx];
        seed->stride[i] = temp_stride[_idx];
    }
    return seed;
}

