#include "../include/ops.h"
#include "../include/tensor.h"

FORWARD_FUNC_DEF(r_op_sum_forward){
    (void) kr; (void) k0; (void) k1;
   return;
}

BACKWARD_FUNC_DEF(r_op_sum_backward){
    (void) kr; (void) k0; (void) k1; (void) idx; (void) seed;
    return NULL;
}