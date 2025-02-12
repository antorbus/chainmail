#include "ops.h"

// representation of any trainable tensor (weights, biases, ..)
typedef struct parameter { 
    tensor *tensor_ptr;
    kernel_tensor **optim_data;
    size_t num_data;
} parameter;

parameter * create_parameter(tensor* tensor_ptr, size_t num_data);
void free_parameter(parameter* param);
void update_param_data(parameter *param, size_t index, kernel_tensor *new_val);