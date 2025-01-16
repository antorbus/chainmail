#include "../include/ops.h"
#include "../include/tensor.h"

FORWARD_FUNC_DEF(s_op_view_forward){
    memcpy(kr->shape, k1->shape, 5 * sizeof(size_t));
    set_contiguous_stride(kr);
}

BACKWARD_FUNC_DEF(s_op_view_backward){

    return NULL;
}