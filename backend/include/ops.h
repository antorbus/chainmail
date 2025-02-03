#ifndef OPS_H
#define OPS_H

#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>


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

FORWARD_FUNC_DEF(b_op_sub_forward);
BACKWARD_FUNC_DEF(b_op_sub_backward);

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

FORWARD_FUNC_DEF(u_op_exp_forward);
BACKWARD_FUNC_DEF(u_op_exp_backward);

FORWARD_FUNC_DEF(u_op_log_forward);
BACKWARD_FUNC_DEF(u_op_log_backward);

FORWARD_FUNC_DEF(u_op_neg_forward);
BACKWARD_FUNC_DEF(u_op_neg_backward);

FORWARD_FUNC_DEF(u_op_sqrt_forward);
BACKWARD_FUNC_DEF(u_op_sqrt_backward);

FORWARD_FUNC_DEF(u_op_abs_forward);
BACKWARD_FUNC_DEF(u_op_abs_backward);

FORWARD_FUNC_DEF(u_op_sign_forward);
BACKWARD_FUNC_DEF(u_op_sign_backward);

FORWARD_FUNC_DEF(u_op_reciprocal_forward);
BACKWARD_FUNC_DEF(u_op_reciprocal_backward);

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

//matmul ops
FORWARD_FUNC_DEF(m_op_bmm_forward);
BACKWARD_FUNC_DEF(m_op_bmm_backward);

FORWARD_FUNC_DEF(m_op_bcmm_forward);
BACKWARD_FUNC_DEF(m_op_bcmm_backward);

//function tables entries

enum {
    TYPE_BINARY = 0,
    TYPE_UNARY,
    TYPE_REDUCE,
    TYPE_SHAPE,
    TYPE_MATMUL,
};

enum OPS {
  //binary ops
  OP_ADD = 0,
  OP_SUB,
  OP_MUL,
  OP_DIVISION,
  //unary ops
  OP_POW,
  OP_RELU,
  OP_SIGMOID,
  OP_EXP,
  OP_LOG,
  OP_NEG,
  OP_SQRT,
  OP_ABS,
  OP_SIGN,
  OP_RECIPROCAL,
  //reduce ops
  OP_SUM,
  //shape ops
  OP_VIEW,
  OP_EXPAND,
  OP_PERMUTE,
  //matmul ops
  OP_BROADCAST_MATMUL,
  OP_BATCH_MATMUL,
  //
  TOTAL_OPS,
};

extern int type_table[TOTAL_OPS];
extern forward_func forward_func_table[TOTAL_OPS];
extern backward_func backward_func_table[TOTAL_OPS];

//helper macros

#define KERNEL_TENSOR_5D_LOOP_START(k)                     \
  _Pragma("omp parallel for collapse(5)")                  \
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

#define BINARY_CONTIGUOUS_ELEMENTWISE_OP_SIMD(kr, k0, k1, operation) \
do {                                                                             \
    if ((kr)->length > 1<<17) {                                                 \
        _Pragma("omp parallel for simd")                                       \
          for (size_t _i = 0; _i < (kr)->length; _i++) {                       \
              (kr)->array[_i] = operation((k0)->array[_i], (k1)->array[_i]);   \
          }                                                                    \
    } else {                                                                     \
        _Pragma("omp simd")                                                      \
        for (size_t _i = 0; _i < (kr)->length; _i++) {                           \
            (kr)->array[_i] = operation((k0)->array[_i], (k1)->array[_i]);       \
        }                                                                        \
    }                                                                            \
} while(0)


#define UNARY_CONTIGUOUS_ELEMENTWISE_OP_SIMD(kr, k0, operation) \
do {                                                                             \
    if ((kr)->length > 1<<17) {                                                 \
        _Pragma("omp parallel for simd")                                       \
          for (size_t _i = 0; _i < (kr)->length; _i++) {                       \
              (kr)->array[_i] = operation((k0)->array[_i]);   \
          }                                                                    \
    } else {                                                                     \
        _Pragma("omp simd")                                                      \
        for (size_t _i = 0; _i < (kr)->length; _i++) {                           \
            (kr)->array[_i] = operation((k0)->array[_i]);       \
        }                                                                        \
    }                                                                            \
} while(0)

#define _add(a, b) a + b
#define _mul(a, b) a * b
#define _sub(a, b) a - b
#define _div(a, b) a / b
#define _neg(a) -1.0 * a
#define _relu(v) ((v) > 0.0) ? (v) : 0.0
#define _sigmoid(x) 1.0 / (1.0 + expf(-1.0 * x))
#define _sigmoid_grad(s) s * (1.0 - s)

#endif 