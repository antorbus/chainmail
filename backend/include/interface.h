#ifndef interface_H
#define interface_H

#include "tensor.h"

//helper macros

#define BINARY_FUNC_DEF(name, op)            \
    tensor * name(tensor *t0, tensor *t1, bool retain_grad) 

#define UNARY_FUNC_DEF(name, op)            \
    tensor * name(tensor *t0, bool retain_grad)

#define REDUCE_FUNC_DEF(name, op)            \
    tensor * name(tensor *t0, tensor *dim_data, bool retain_grad)

#define SHAPE_FUNC_DEF(name, op)            \
    tensor * name(tensor *t0, tensor *dim_data)

//external functions

#ifdef __cplusplus
extern "C" {
#endif

void derive(tensor * t, kernel_tensor * seed);

void backwards(tensor * t);

void free_tensor(tensor *t);

tensor * empty_tensor(size_t shape[5], bool retain_grad);
void memset_kernel_tensor(kernel_tensor * k, lemur_float val);

extern char* op_map[TOTAL_OPS];
char* get_op_name(int op_id);

//binary ops
BINARY_FUNC_DEF(mul, OP_MUL);
BINARY_FUNC_DEF(add, OP_ADD);
BINARY_FUNC_DEF(division, OP_DIVISION);

//unary ops
UNARY_FUNC_DEF(relu, OP_RELU);
UNARY_FUNC_DEF(sigmoid, OP_SIGMOID);
UNARY_FUNC_DEF(exponential, OP_EXP);
UNARY_FUNC_DEF(logarithm, OP_LOG);
UNARY_FUNC_DEF(neg, OP_NEG);
UNARY_FUNC_DEF(square_root, OP_SQRT);
UNARY_FUNC_DEF(absolute, OP_ABS);
UNARY_FUNC_DEF(sign, OP_SIGN);
UNARY_FUNC_DEF(reciprocal, OP_RECIPROCAL);

//reduce ops
REDUCE_FUNC_DEF(sum, OP_SUM);

//shape ops
SHAPE_FUNC_DEF(view, OP_VIEW);
SHAPE_FUNC_DEF(expand, OP_EXPAND);
SHAPE_FUNC_DEF(permute, OP_PERMUTE);

#ifdef __cplusplus
}
#endif

#endif 