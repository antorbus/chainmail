#ifndef OPS_H
#define OPS_H

#include <stddef.h>
#include <stdbool.h>
#include <string.h>


#if defined(__FLT32_MANT_DIG__) 
    typedef _Float32 lemur_float;
#else
    typedef float lemur_float;
#endif


//forward declarations
struct tensor;
struct kernel_tensor;
typedef struct tensor tensor;
typedef struct kernel_tensor kernel_tensor;

typedef void (*forward_func)(   kernel_tensor *kr, 
                                kernel_tensor *k0, 
                                kernel_tensor *k1);

typedef kernel_tensor * (*backward_func)(   kernel_tensor *kr, 
                                            kernel_tensor *k0, 
                                            kernel_tensor *k1, 
                                            kernel_tensor *seed,
                                            size_t idx);



tensor * kernel_forward(int func, tensor * t0, tensor * t1, bool retain_grad);
void kernel_backward(tensor *tr, kernel_tensor *seed);

#define FORWARD_FUNC_DEF(name)            \
    void name(kernel_tensor *kr, kernel_tensor *k0, kernel_tensor *k1)

#define BACKWARD_FUNC_DEF(name)            \
    kernel_tensor * name (kernel_tensor *kr, \
                          kernel_tensor *k0, \
                          kernel_tensor *k1, \
                          kernel_tensor *seed,\
                          size_t idx) 
//binary ops
FORWARD_FUNC_DEF(b_op_add_forward);
BACKWARD_FUNC_DEF(b_op_add_backward);

FORWARD_FUNC_DEF(b_op_mul_forward);
BACKWARD_FUNC_DEF(b_op_mul_backward);

FORWARD_FUNC_DEF(b_op_division_forward);
BACKWARD_FUNC_DEF(b_op_division_backward);

//unary ops
FORWARD_FUNC_DEF(u_op_exp_forward);
BACKWARD_FUNC_DEF(u_op_exp_backward);

FORWARD_FUNC_DEF(u_op_pow_forward);
BACKWARD_FUNC_DEF(u_op_pow_backward);

FORWARD_FUNC_DEF(u_op_relu_forward);
BACKWARD_FUNC_DEF(u_op_relu_backward);

FORWARD_FUNC_DEF(u_op_sigmoid_forward);
BACKWARD_FUNC_DEF(u_op_sigmoid_backward);

//reduce ops
FORWARD_FUNC_DEF(r_op_sum_forward);
BACKWARD_FUNC_DEF(r_op_sum_backward);

//shape ops
FORWARD_FUNC_DEF(s_op_view_forward);
BACKWARD_FUNC_DEF(s_op_view_backward);

FORWARD_FUNC_DEF(s_op_expand_forward);
BACKWARD_FUNC_DEF(s_op_expand_backward);

FORWARD_FUNC_DEF(s_op_permute_forward);
BACKWARD_FUNC_DEF(s_op_permute_backward);

//function tables entries

enum {
    TYPE_BINARY = 0,
    TYPE_UNARY,
    TYPE_REDUCE,
    TYPE_SHAPE,
};

enum OPS {
  //binary ops
  OP_ADD = 0,
  OP_MUL,
  OP_DIVISION,
  //unary ops
  OP_EXP,
  OP_POW,
  OP_RELU,
  OP_SIGMOID,
  //reduce ops
  OP_SUM,
  //shape ops
  OP_VIEW,
  OP_EXPAND,
  OP_PERMUTE,
  //
  TOTAL_OPS,
};

extern int type_table[TOTAL_OPS];
extern forward_func forward_func_table[TOTAL_OPS];
extern backward_func backward_func_table[TOTAL_OPS];

//helper macros

#define KERNEL_TENSOR_5D_LOOP_START(k)                            \
  for (size_t d0 = 0; d0 < (k)->shape[0]; d0++)            \
    for (size_t d1 = 0; d1 < (k)->shape[1]; d1++)          \
      for (size_t d2 = 0; d2 < (k)->shape[2]; d2++)        \
        for (size_t d3 = 0; d3 < (k)->shape[3]; d3++)      \
          for (size_t d4 = 0; d4 < (k)->shape[4]; d4++)


#define KERNEL_TENSOR_GET_OFFSET(k) \
    ( d0*(k)->stride[0] +    \
      d1*(k)->stride[1] +    \
      d2*(k)->stride[2] +    \
      d3*(k)->stride[3] +    \
      d4*(k)->stride[4] )



#endif 
