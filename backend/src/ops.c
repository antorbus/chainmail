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
    [OP_SUM] = r_op_sum_forward,

    //shape ops
    [OP_VIEW] = s_op_view_forward,
};

backward_func backward_func_table[] = {
    //binary ops
    [OP_ADD] = b_op_add_backward,
    [OP_MUL] = b_op_mul_backward,

    //unary ops
    [OP_RELU] = u_op_relu_backward,

    //reduce ops
    [OP_SUM] = r_op_sum_backward,

    //shape ops
    [OP_VIEW] = s_op_view_backward,
};

int type_table[] = { //TODO ADD TO DOCS
    //binary ops
    [OP_ADD] = TYPE_BINARY,
    [OP_MUL] = TYPE_BINARY,

    //unary ops
    [OP_RELU] = TYPE_UNARY,

    //reduce ops
    [OP_SUM] = TYPE_REDUCE, 

    //shape ops
    [OP_VIEW] = TYPE_SHAPE, 
};

tensor * kernel_forward(int func, tensor * t0, tensor * t1, bool retain_grad){

    kernel_tensor *k;
    bool requires_grad = false;
    kernel_tensor *grad = NULL;

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
            t1 = NULL;
            if (t0->requires_grad == true){
                    requires_grad = true;
            }
            k = empty_contiguous_kernel_tensor_like(t0->k);
            if (retain_grad == true){
                grad = empty_contiguous_kernel_tensor_like(k);
                memset_kernel_tensor(grad, 0.0);
            }
            forward_func_table[func](k, t0->k, t1->k);
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
            set_reduced_shape(reduced_shape, t0->k->shape, t1->k->shape);
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
                fprintf(stderr, "Error: Shape operations cannot retain grad as they point to parent's memory, call deepcopy instead.\n");
                return NULL;
            }
            forward_func_table[func](k, t0->k, t1->k);
            break;
        
        default:
            fprintf(stderr, "Error: unknown operation type not in type table.\n");
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

    kernel_tensor *next_seed0 = backward_func_table[func](kr, k0, k1, seed, 0); 
    kernel_tensor *next_seed1;
    if (type_table[func] == TYPE_BINARY){ 
        next_seed1 = backward_func_table[func](kr, k0, k1, seed, 1);
    }

    if (next_seed0 != seed){ //shape ops do not make a new gradient
        free_kernel_tensor(seed); 
    } else {
        if (type_table[func] == TYPE_SHAPE){
            fprintf(stderr, "backwards kernel returns seed for shape op\n");
            return NULL;
        }
    }

    derive(t0, next_seed0);
    if (type_table[func] == TYPE_BINARY){
        derive(t1, next_seed1);
    }

}