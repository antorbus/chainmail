#include "../include/ops.h"
#include "../include/tensor.h"


//#define REDUCE_KERNEL_TENSOR_5D_LOOP_START(k, dims)   //TODO 

FORWARD_FUNC_DEF(r_op_sum_forward){
    for (size_t d0 = 0; d0 < kr->shape[0]; d0++){           
        for (size_t d1 = 0; d1 < kr->shape[1]; d1++){         
            for (size_t d2 = 0; d2 < kr->shape[2]; d2++){       
                for (size_t d3 = 0; d3 < kr->shape[3]; d3++){     
                    for (size_t d4 = 0; d4 < kr->shape[4]; d4++){

                    }
                }
            }
        }
    }


    for (size_t d = 0; d < 5; d++){
        if (k1->array[d] != 0){
            //reduce
        } else{
            //do not reduce
        }
    }
    
   return;
}

BACKWARD_FUNC_DEF(r_op_sum_backward){
    (void) kr; (void) idx; 
    kernel_tensor *next_seed = empty_kernel_tensor_like(seed); 
    memcpy(next_seed->array, seed->array, seed->length * sizeof(lemur_float));
    for (size_t i = 0; i < 5; i++) {
        if (k1->array[i] !=0 ){
            next_seed->shape[i] = k0->shape[i];
            next_seed->stride[i] = 0;
        }
    } 
    return next_seed;
}