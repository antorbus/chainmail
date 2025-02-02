#include "../include/ops.h"
#include "../include/tensor.h"
#include "../include/interface.h"

tensor * kernel_forward(int func, tensor * t0, tensor * t1, bool retain_grad){

    kernel_tensor *k;
    bool requires_grad = false;
    kernel_tensor *grad = NULL;

    if (is_contiguous(t0->k) == false){
        fprintf(stderr, "Error: Attempted to operate on non-contiguous tensor t0.\n");
        return NULL;
    }
    if (t1 != NULL){
        if (is_contiguous(t1->k) == false){
            fprintf(stderr, "Error: Attempted to operate on non-contiguous tensor t1.\n");
        return NULL;
    }
    }

    switch (type_table[func]){

        case TYPE_BINARY:
            if (are_shapes_equal( t0->k->shape, t1->k->shape) != true){
                fprintf(stderr, "Error: Shapes of tensors t0 and t1 are not equal.\n");
                return NULL;
            }
            if ((t0->requires_grad == true) || (t1->requires_grad == true)){
                    requires_grad = true;
            }
            k = empty_contiguous_kernel_tensor_like(t0->k);
            if (retain_grad == true){
                grad = empty_contiguous_kernel_tensor_like(k);
                memset_kernel_tensor(grad, 0.0);
            }
            forward_func_table[func](k, t0->k, t1->k);
            break;

        case TYPE_UNARY:
            if (t0->requires_grad == true){
                    requires_grad = true;
            }
            k = empty_contiguous_kernel_tensor_like(t0->k);
            if (retain_grad == true){
                grad = empty_contiguous_kernel_tensor_like(k);
                memset_kernel_tensor(grad, 0.0);
            }
            if (t1 == NULL){ 
                forward_func_table[func](k, t0->k, NULL);  
            } else {
                forward_func_table[func](k, t0->k, t1->k); //power is not REALLY unary (but sort of)
            }

            break;

        case TYPE_REDUCE: 
            // t1 will be tensor of 
            // length 5 which shape (1,1,1,1,5) 
            // and 1 the dimensions 
            // affected and 0 at the ones that stay put
            if (t1->k->length != 5){
                fprintf(stderr, "Error: Shapes of t1 (a.k.a dims) is not 5.\n");
                return NULL;
            }
            if (t0->requires_grad == true){
                requires_grad = true;
            }
            size_t reduced_shape[5];
            set_reduced_shape(reduced_shape, t0->k->shape, t1->k->array);
            k = empty_contiguous_kernel_tensor(reduced_shape);
            if (retain_grad == true){
                grad = empty_contiguous_kernel_tensor_like(k);
                memset_kernel_tensor(grad, 0.0);
            }
            forward_func_table[func](k, t0->k, t1->k);
            break;

        case TYPE_SHAPE:
            if (t0->requires_grad == true){
                    requires_grad = true;
            }
            
            k = kernel_tensor_shallow_copy(t0->k); 
            
            if (retain_grad == true){
                fprintf(stderr, "Error: Shape operations cannot retain grad as they could point to parent's memory, call deepcopy instead.\n");
                return NULL;
            }
            forward_func_table[func](k, t0->k, t1->k);
            if (func != OP_VIEW){   //view is the only one that keep the parent's memory
                kernel_tensor *k_temp = contiguous_deepcopy_kernel_tensor(k);
                free_kernel_tensor(k); //will not free parent's array because k is a shallow copy
                                       //look at free_kernel_tensor
                k = k_temp;
            }
            break;

        case TYPE_MATMUL:
            fprintf(stderr, "Error: matmul operation type not implemented.\n");
            return NULL;
        
        default:
            fprintf(stderr, "Error: unknown operation type not in type table.\n");
            return NULL;
    }

    if (is_contiguous(k) == false){
        fprintf(stderr, "%s returned non-contiguous kernel tesnsor in forward\n", get_op_name(func));
        return NULL;
    }

    expression *comes_from = expression_from(func, t0, t1);
    tensor *t = tensor_from(k, comes_from, requires_grad, grad);

    return t;
}


void kernel_backward(tensor *tr, kernel_tensor *seed){
    //TODO: This can probably be multithread 
    tensor *t0 = tr->comes_from->t0;
    tensor *t1 = tr->comes_from->t1;
    kernel_tensor *kr = tr->k;
    kernel_tensor *k0 = t0->k;
    kernel_tensor *k1 = (t1 != NULL) ? t1->k : NULL;

    int func = tr->comes_from->backward_func;

    if (is_contiguous(seed) == false){
        fprintf(stderr, "seed (kernel_backward %s call) is non-contiguous, aborting backwards\n", get_op_name(func));
        return;
    }

    kernel_tensor *next_seed1 = NULL;
    if (type_table[func] == TYPE_BINARY){ 
        kernel_tensor *deepcopy_seed = contiguous_deepcopy_kernel_tensor(seed);
        //binary backward should return deepcopy_seed, therefore this is safe and there are no leaks
        next_seed1 = backward_func_table[func](kr, k0, k1, deepcopy_seed, 1);
    }
    //TODO fix this
    //next_seed0 must be calculated AFTER next_seed1 since next_seed0 could be seed itself 
    //making the calculation of next_seed1 incorrect 
    kernel_tensor *next_seed0 = backward_func_table[func](kr, k0, k1, seed, 0); 
    inplace_contiguous_kernel_tensor(next_seed0);

    if (next_seed1 == seed){
        fprintf(stderr, "next_seed1 cannot be seed\n");
        return;
    }

    if (next_seed0 != seed){ //some ops do not make a new gradient for next_seed0. one is always made for next_seed1
        free_kernel_tensor(seed); 
    } 

    //derive(t0, next_seed0);
    if (t0->grad != NULL){
        b_op_add_forward(t0->grad, t0->grad, next_seed0);
    }
    if (t0->comes_from != NULL){
        kernel_backward(t0, next_seed0);
    } else {
        free_kernel_tensor(next_seed0); //frees leaf gradients
    }

    if (type_table[func] == TYPE_BINARY){
        if (next_seed1 == NULL){
            fprintf(stderr, "binary backwards kernel returns null seed1\n");
            return;
        }

        //derive(t1, next_seed1);
        if (t1->grad != NULL){
            b_op_add_forward(t1->grad, t1->grad, next_seed1);
        }
        if (t1->comes_from != NULL){
            kernel_backward(t1, next_seed1);
        } else {
            free_kernel_tensor(next_seed1); //frees leaf gradients
        }
        
    }

}


forward_func forward_func_table[] = {
    //binary ops
    [OP_ADD] = b_op_add_forward,
    [OP_SUB] = b_op_sub_forward,
    [OP_MUL] = b_op_mul_forward,
    [OP_DIVISION] = b_op_division_forward,

    //unary ops
    [OP_EXP] = u_op_exp_forward,
    [OP_POW] = u_op_pow_forward,
    [OP_RELU] = u_op_relu_forward,
    [OP_SIGMOID] = u_op_sigmoid_forward,
    [OP_LOG] = u_op_log_forward,
    [OP_NEG] = u_op_neg_forward,
    [OP_SQRT] = u_op_sqrt_forward,
    [OP_ABS] = u_op_abs_forward,
    [OP_SIGN] = u_op_sign_forward,
    [OP_RECIPROCAL] = u_op_reciprocal_forward,

    //reduce ops
    [OP_SUM] = r_op_sum_forward,

    //shape ops
    [OP_VIEW] = s_op_view_forward,
    [OP_EXPAND] = s_op_expand_forward,
    [OP_PERMUTE] = s_op_permute_forward,

    //matmul ops
    [OP_MATMUL] = m_op_mm_forward,
    [OP_BATCH_MATMUL] = m_op_bmm_forward,
};

backward_func backward_func_table[] = {
    //binary ops
    [OP_ADD] = b_op_add_backward,
    [OP_SUB] = b_op_sub_backward,
    [OP_MUL] = b_op_mul_backward,
    [OP_DIVISION] = b_op_division_backward,

    //unary ops
    [OP_EXP] = u_op_exp_backward,
    [OP_POW] = u_op_pow_backward,
    [OP_RELU] = u_op_relu_backward,
    [OP_SIGMOID] = u_op_sigmoid_backward,
    [OP_LOG] = u_op_log_backward,
    [OP_NEG] = u_op_neg_backward,
    [OP_SQRT] = u_op_sqrt_backward,
    [OP_ABS] = u_op_abs_backward,
    [OP_SIGN] = u_op_sign_backward,
    [OP_RECIPROCAL] = u_op_reciprocal_backward,

    //reduce ops
    [OP_SUM] = r_op_sum_backward,

    //shape ops
    [OP_VIEW] = s_op_view_backward,
    [OP_EXPAND] = s_op_expand_backward,
    [OP_PERMUTE] = s_op_permute_backward,

    //matmul ops
    [OP_MATMUL] = m_op_mm_backward,
    [OP_BATCH_MATMUL] = m_op_bmm_backward,
};

int type_table[] = { //TODO ADD TO DOCS
    //binary ops
    [OP_ADD] = TYPE_BINARY,
    [OP_SUB] = TYPE_BINARY,
    [OP_MUL] = TYPE_BINARY,
    [OP_DIVISION] = TYPE_BINARY,

    //unary ops
    [OP_EXP] = TYPE_UNARY,
    [OP_POW] = TYPE_UNARY,  // (implemented as unary op, but takes two tensor inputs)
    [OP_RELU] = TYPE_UNARY,
    [OP_SIGMOID] = TYPE_UNARY,
    [OP_LOG] = TYPE_UNARY,
    [OP_NEG] = TYPE_UNARY,
    [OP_SQRT] = TYPE_UNARY,
    [OP_ABS] = TYPE_UNARY,
    [OP_SIGN] = TYPE_UNARY,
    [OP_RECIPROCAL] = TYPE_UNARY,

    //reduce ops
    [OP_SUM] = TYPE_REDUCE, 

    //shape ops
    [OP_VIEW] = TYPE_SHAPE, 
    [OP_EXPAND] = TYPE_SHAPE,
    [OP_PERMUTE] = TYPE_SHAPE,

    //matmul ops
    [OP_MATMUL] = TYPE_MATMUL,
    [OP_BATCH_MATMUL] = TYPE_MATMUL,
};