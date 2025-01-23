#ifndef TENSOR_H
#define TENSOR_H

#include "ops.h"

typedef struct kernel_tensor {
    lemur_float *array; 
    size_t length; 
    size_t shape[5]; 
    int64_t stride[5];
    bool computed; 
    bool shallow;
} kernel_tensor;


typedef struct expression expression; //ignore: forward declaration

typedef struct tensor {
    kernel_tensor *k;
    expression *comes_from; 
    bool requires_grad;
    kernel_tensor *grad;
} tensor;

typedef struct expression {
    tensor *t0;
    tensor *t1;
    int backward_func;
} expression;

void init_seed(unsigned int seed);

void backward(tensor * t);

expression * expression_from(int func, tensor *t0, tensor *t1);

tensor * empty_tensor(size_t shape[5], bool retain_grad);
void memset_kernel_tensor(kernel_tensor * k, lemur_float val);
void free_kernel_tensor(kernel_tensor *k);
void free_tensor(tensor *t);

kernel_tensor * empty_contiguous_kernel_tensor(size_t shape[5]);
kernel_tensor * empty_contiguous_kernel_tensor_like(kernel_tensor *k);
kernel_tensor * empty_kernel_tensor_like(kernel_tensor *k);
kernel_tensor * empty_kernel_tensor(size_t shape[5]);
tensor * tensor_from(kernel_tensor *k, expression *comes_from, bool requires_grad, kernel_tensor* grad);
kernel_tensor * kernel_tensor_shallow_copy(kernel_tensor *k);
tensor * dim_tensor_from(size_t shape[5]);
void inplace_contiguous_kernel_tensor(kernel_tensor *k); 
kernel_tensor * contiguous_deepcopy_kernel_tensor(kernel_tensor *k); 
void init_random_uniform_kernel_tensor(kernel_tensor * k, lemur_float min, lemur_float max);
void init_random_normal_kernel_tensor(kernel_tensor * k, lemur_float mean, lemur_float std);

bool are_shapes_equal(size_t shape0[5], size_t shape1[5]);
void set_reduced_shape(size_t reduced_shape[5], size_t original_shape[5], lemur_float dims[5]);
bool is_contiguous(kernel_tensor *k);
void set_contiguous_stride(kernel_tensor * k);
bool is_tensor_scalar(tensor* t);

void print_kernel_tensor(kernel_tensor *k);
void print_expression(expression *e);
void print_tensor(tensor *t);

#endif 
