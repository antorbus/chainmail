#ifndef interface_H
#define interface_H

#include "tensor.h"

//helper macros

#define BINARY_FUNC_DEF(name, op)            \
    tensor * name(tensor *t0, tensor *t1, bool retain_grad) 

#define UNARY_FUNC_DEF(name, op)            \
    tensor * name(tensor *t0, bool retain_grad)

#define REDUCE_FUNC_DEF(name, op)            \
    tensor * name(tensor *t0, tensor *dims, bool retain_grad)

//external functions

#ifdef __cplusplus
extern "C" {
#endif

void derive(tensor * t, kernel_tensor * seed);

void backwards(tensor * t);

void free_tensor(tensor *t);

tensor * empty_tensor(size_t shape[5], bool retain_grad);

//binary ops

BINARY_FUNC_DEF(mul, OP_MUL);
BINARY_FUNC_DEF(add, OP_ADD);

//unary ops

UNARY_FUNC_DEF(relu, OP_RELU);

//reduce ops

REDUCE_FUNC_DEF(sum, OP_SUM);

//shape ops

#ifdef __cplusplus
}
#endif

#endif 
