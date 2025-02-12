#include "../include/tensor.h"
#include "../include/interface.h"
#include "../include/parameter.h"

parameter * create_parameter (tensor *tensor_ptr, size_t num_data) {
    if (!tensor_ptr) {
        return NULL;
    }

    parameter* param = (parameter*)malloc(sizeof(parameter));
    if (!param) {
        perror("Failed to allocate parameter");
        return NULL;
    }

    param->optim_data = (kernel_tensor **)malloc(num_data * sizeof(kernel_tensor *));
    param->tensor_ptr = tensor_ptr;
    param->num_data = num_data;

    for (size_t i = 0; i < num_data; i++) {
        param->optim_data[i] = empty_contiguous_kernel_tensor(tensor_ptr->k->shape);
    }
    return param;
}

void free_parameter(parameter *param) {

    if (param->optim_data) {
        for (size_t i = 0; i < param->num_data; i++) {
            if (param->optim_data[i]) {
                free_kernel_tensor(&(param->optim_data[i]));
            }
        }
        free(param->optim_data);
    }
    free(param);
}

void update_param_data(parameter *param, size_t index, kernel_tensor *new_val) {
    if (!param || index >= param->num_data) return;

    free_kernel_tensor(&(param->optim_data[index]));
    param->optim_data[index] = new_val;
}