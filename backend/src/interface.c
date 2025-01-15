#include "../include/interface.h"

//TODO stubs

//binary ops

BINARY_FUNC_DEF(mul, OP_MUL){
    return kernel_forward(OP_MUL, t0, t1, retain_grad);  
}

BINARY_FUNC_DEF(add, OP_ADD){
    return kernel_forward(OP_ADD, t0, t1, retain_grad);  
}

//unary ops

UNARY_FUNC_DEF(relu, OP_RELU){
    return kernel_forward(OP_RELU, t0, NULL, retain_grad);  
}

//reduce ops

REDUCE_FUNC_DEF(sum, OP_SUM){
    return kernel_forward(OP_SUM, t0, dims, retain_grad); 
}

//shape ops

SHAPE_FUNC_DEF(view, OP_VIEW){
    if(is_contiguous(t0->k) == false){
        fprintf(stderr, "Error: View can only be perfomed on contiguous tensors, use reshape instead.\n");
        return NULL;
        
    }
    return kernel_forward(OP_VIEW, t0, dims, retain_grad); 
}