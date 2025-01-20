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
[OP_MUL] = "mul",
[OP_DIVISION] = "div",
//unary ops
[OP_RELU] = "relu",
[OP_SIGMOID] = "sigmoid",
//reduce ops
[OP_SUM] = "sum",
//shape ops
[OP_VIEW] = "view",
[OP_EXPAND]= "expand",
[OP_PERMUTE]= "permute",
};

//binary ops

BINARY_FUNC_DEF(mul, OP_MUL){
    return kernel_forward(OP_MUL, t0, t1, retain_grad);  
}

BINARY_FUNC_DEF(add, OP_ADD){
    return kernel_forward(OP_ADD, t0, t1, retain_grad);  
}

BINARY_FUNC_DEF(division, OP_DIVISION){
    return kernel_forward(OP_DIVISION, t0, t1, retain_grad);  
}

//unary ops

UNARY_FUNC_DEF(relu, OP_RELU){
    return kernel_forward(OP_RELU, t0, NULL, retain_grad);  
}

UNARY_FUNC_DEF(sigmoid, OP_SIGMOID){
    return kernel_forward(OP_SIGMOID, t0, NULL, retain_grad);  
}

//reduce ops

REDUCE_FUNC_DEF(sum, OP_SUM){
    return kernel_forward(OP_SUM, t0, dim_data, retain_grad); 
}

//shape ops

//todo add checks, like view positive dimensions, length is the same, etc
SHAPE_FUNC_DEF(view, OP_VIEW){
    if(is_contiguous(t0->k) == false){
        fprintf(stderr, "Error: View can only be perfomed on contiguous tensors.\n");
        return NULL;
        
    }
    if (dim_data->k->length != 5){
        fprintf(stderr, "Error: View dimensions must be 5.\n");
        return NULL;
    }
    return kernel_forward(OP_VIEW, t0, dim_data, false); 
}
SHAPE_FUNC_DEF(expand, OP_EXPAND){
    if (dim_data->k->length != 5){
        fprintf(stderr, "Error: Expand dimensions must be 5.\n");
        return NULL;
    }
    for (size_t i = 0; i<5; i++){
        if((t0->k->shape[i] != 1) && ((size_t) dim_data->k->array[i] != t0->k->shape[i])){
            fprintf(stderr, "Error: Can only expand singleton dimensions.\n");
            return NULL;
        }
    }
    return kernel_forward(OP_EXPAND, t0, dim_data, false); 
}

SHAPE_FUNC_DEF(permute, OP_PERMUTE){

    if (dim_data->k->length != 5){
        fprintf(stderr, "Error: Permute dimensions must be 5.\n");
        return NULL;
    }
    int valid_perm_counter = 0;
    bool perm_tracker[5] = {false, false, false, false, false};
    for (size_t i = 0; i<5; i++){
        size_t perm_idx = (size_t) dim_data->k->array[i];
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
    return kernel_forward(OP_PERMUTE, t0, dim_data, false); 
}