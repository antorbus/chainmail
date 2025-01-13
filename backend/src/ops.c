#include "../include/ops.h"
#include "../include/tensor.h"
#include "../include/interface.h"

forward_func forward_func_table[] = {
    //binary ops
    [OP_ADD] = b_op_add_forward,
    [OP_MUL] = b_op_mul_forward,

    //unary ops
    [OP_RELU] = u_op_relu_forward,

    //reduce ops

    //shape ops
};

backward_func backward_func_table[] = {
    //binary ops
    [OP_ADD] = b_op_add_backward,
    [OP_MUL] = b_op_mul_backward,

    //unary ops
    [OP_RELU] = u_op_relu_backward,

    //reduce ops

    //shape ops
};

void kernel_backward(tensor *tr, kernel_tensor *seed){
    //TODO: This can probably be multithread 
    tensor *t0 = tr->comes_from->t0;
    tensor *t1 = tr->comes_from->t1;
    kernel_tensor *kr = tr->k;
    kernel_tensor *k0 = t0->k;
    kernel_tensor *k1 = (t1 != NULL) ? t1->k : NULL;
    int func = tr->comes_from->backward_func;

    kernel_tensor *next_seed0 = backward_func_table[func](kr, k0, k1, seed, 0); 
    kernel_tensor *next_seed1;
    if (t1 != NULL){
        next_seed1 = backward_func_table[func](kr, k0, k1, seed, 1);
    }

    free_kernel_tensor(seed);

    derive(t0, next_seed0);
    if (t1 != NULL){
        derive(t1, next_seed1);
    }

}