#include "../include/interface.h"

//TODO stubs
char* get_op_name(int op_id) {
    if(op_id >= 0 && op_id < TOTAL_OPS) {
        return op_map[op_id];
    }
    return "unknown_op";
}

char* op_map[TOTAL_OPS] ={
//binary ops
[OP_ADD] = "add",
[OP_SUB] = "sub",
[OP_MUL] = "mul",
[OP_DIVISION] = "div",
//unary ops
[OP_EXP] = "exp",
[OP_POW] = "pow",
[OP_RELU] = "relu",
[OP_SIGMOID] = "sigmoid",
[OP_LOG] = "log",
[OP_NEG] = "neg",
[OP_SQRT] = "sqrt",
[OP_ABS] = "abs",
[OP_SIGN] = "sign",
[OP_RECIPROCAL] = "reciprocal",
//reduce ops
[OP_SUM] = "sum",
//shape ops
[OP_VIEW] = "view",
[OP_EXPAND] = "expand",
[OP_PERMUTE] = "permute",
//matmul
[OP_BATCH_MATMUL] = "bmm",
[OP_BROADCAST_MATMUL] = "bcmm",
};


//binary ops

DOUBLE_INPUT_FUNC_DEF(add){
    return kernel_forward(OP_ADD, t0, t1, retain_grad);  
}

DOUBLE_INPUT_FUNC_DEF(sub){
    return kernel_forward(OP_SUB, t0, t1, retain_grad);  
}

DOUBLE_INPUT_FUNC_DEF(mul){
    return kernel_forward(OP_MUL, t0, t1, retain_grad);  
}

DOUBLE_INPUT_FUNC_DEF(division){
    return kernel_forward(OP_DIVISION, t0, t1, retain_grad);  
}

//unary ops

SINGLE_INPUT_FUNC_DEF(exponential){
    return kernel_forward(OP_EXP, t0, NULL, retain_grad);
}

DOUBLE_INPUT_FUNC_DEF(power){ 
    if (is_tensor_scalar(t1) == false){
        fprintf(stderr, "Error: Exponent of tensor must be a scalar.\n");
        return NULL;
    }
    return kernel_forward(OP_POW, t0, t1, retain_grad);
}

SINGLE_INPUT_FUNC_DEF(relu){
    return kernel_forward(OP_RELU, t0, NULL, retain_grad);  
}

SINGLE_INPUT_FUNC_DEF(sigmoid){
    return kernel_forward(OP_SIGMOID, t0, NULL, retain_grad);  
}

SINGLE_INPUT_FUNC_DEF(logarithm){
    return kernel_forward(OP_LOG, t0, NULL, retain_grad);  
}

SINGLE_INPUT_FUNC_DEF(neg){
    return kernel_forward(OP_NEG, t0, NULL, retain_grad);  
}

SINGLE_INPUT_FUNC_DEF(square_root){
    return kernel_forward(OP_SQRT, t0, NULL, retain_grad);  
}

SINGLE_INPUT_FUNC_DEF(absolute){
    return kernel_forward(OP_ABS, t0, NULL, retain_grad);  
}

SINGLE_INPUT_FUNC_DEF(sign){
    return kernel_forward(OP_SIGN, t0, NULL, retain_grad);  
}

SINGLE_INPUT_FUNC_DEF(reciprocal){
    return kernel_forward(OP_RECIPROCAL, t0, NULL, retain_grad);  
}

//reduce ops

DOUBLE_INPUT_FUNC_DEF(sum){
    return kernel_forward(OP_SUM, t0, t1, retain_grad); 
}

//shape ops

//todo add checks, like view positive dimensions, length is the same, etc
DOUBLE_INPUT_FUNC_DEF(view){
    (void) retain_grad;
    if(is_contiguous(t0->k) == false){
        fprintf(stderr, "Error: View can only be perfomed on contiguous tensors.\n");
        return NULL;
    }
    if (t1->k->length != 5){
        fprintf(stderr, "Error: View dimensions must be 5.\n");
        return NULL;
    }
    return kernel_forward(OP_VIEW, t0, t1, false); 
}
DOUBLE_INPUT_FUNC_DEF(expand){
    (void) retain_grad;
    if (t1->k->length != 5){
        fprintf(stderr, "Error: Expand dimensions must be 5.\n");
        return NULL;
    }
    for (size_t i = 0; i<5; i++){
        if((t0->k->shape[i] != 1) && ((size_t) t1->k->array[i] != t0->k->shape[i])){
            fprintf(stderr, "Error: Can only expand singleton dimensions.\n");
            return NULL;
        }
    }
    return kernel_forward(OP_EXPAND, t0, t1, false); 
}

DOUBLE_INPUT_FUNC_DEF(permute){
    (void) retain_grad;
    if (t1->k->length != 5){
        fprintf(stderr, "Error: Permute dimensions must be 5.\n");
        return NULL;
    }
    int valid_perm_counter = 0;
    bool perm_tracker[5] = {false, false, false, false, false};
    for (size_t i = 0; i<5; i++){
        size_t perm_idx = (size_t) t1->k->array[i];
        if(( perm_idx > 4) || (perm_idx < 0)){
            break;
        }
        else{
            if (perm_tracker[perm_idx] == false){
                perm_tracker[perm_idx] = true;
            } else{
                break;
            }
            valid_perm_counter++;
        }
    }
    if (valid_perm_counter != 5){
        fprintf(stderr, "Error: Invalid permutation.\n");
        return NULL;
    }
    return kernel_forward(OP_PERMUTE, t0, t1, false); 
}

//matmul
//mxn nxk --> nxk
DOUBLE_INPUT_FUNC_DEF(bmm){
    if ((t0->k->shape[0] != t1->k->shape[0]) ||
        (t0->k->shape[1] != t1->k->shape[1]) ||
        (t0->k->shape[2] != t1->k->shape[2]) ||
        (t0->k->shape[4] != t1->k->shape[3])){
        fprintf(stderr, "Error: Invalid input shapes.\n");
        return NULL;
    }
    return kernel_forward(OP_BATCH_MATMUL, t0, t1, retain_grad);
}

DOUBLE_INPUT_FUNC_DEF(bcmm){
    if ((1 != t1->k->shape[0]) ||
        (1 != t1->k->shape[1]) ||
        (1 != t1->k->shape[2]) ||
        (t0->k->shape[4] != t1->k->shape[3])){
        fprintf(stderr, "Error: Invalid input shapes.\n");
        return NULL;
    }
    return kernel_forward(OP_BROADCAST_MATMUL, t0, t1, retain_grad);
}