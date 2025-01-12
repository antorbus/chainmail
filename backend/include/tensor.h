#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stdlib.h>
#include "ops.h"

typedef struct kernel_tensor {
    lemur_float *array; 
    size_t length; 
    size_t shape[5]; 
    int64_t stride[5];
    bool computed; 
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

void derive(tensor * t, kernel_tensor * seed);
void backwards(tensor * t);

expression * expression_from(int func, tensor *t0, tensor *t1);

void memset_kernel_tensor(kernel_tensor * k, lemur_float val);
void free_kernel_tensor(kernel_tensor *k);
void free_tensor(tensor *t);
kernel_tensor * empty_contiguous_kernel_tensor_like(kernel_tensor *k);
kernel_tensor * empty_kernel_tensor_like(kernel_tensor *k);
kernel_tensor * empty_kernel_tensor(size_t shape[5]);
tensor * empty_tensor(size_t shape[5], bool retain_grad);
tensor * tensor_from(kernel_tensor *k, expression *comes_from, bool requires_grad, kernel_tensor* grad);


void print_kernel_tensor(kernel_tensor *k);
void print_expression(expression *e);
void print_tensor(tensor *t);


#endif 
