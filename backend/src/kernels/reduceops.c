#include "../../include/ops.h"
#include "../../include/tensor.h"


FORWARD_FUNC_DEF(r_op_sum_forward){

   memset_kernel_tensor(kr, 0.0);

   size_t rd0, rd1, rd2, rd3, rd4;
   size_t is_rd0 = (size_t) k1->array[0];
   size_t is_rd1 = (size_t) k1->array[1];
   size_t is_rd2 = (size_t) k1->array[2];
   size_t is_rd3 = (size_t) k1->array[3];
   size_t is_rd4 = (size_t) k1->array[4];

   KERNEL_TENSOR_5D_LOOP_START(k0){

        size_t offset_k0 = KERNEL_TENSOR_GET_OFFSET(k0);
    
        rd0 = d0 * is_rd0;
        rd1 = d1 * is_rd1;
        rd2 = d2 * is_rd2;
        rd3 = d3 * is_rd3;
        rd4 = d4 * is_rd4;
        size_t offset_kr = rd0*(kr)->stride[0] + rd1*(kr)->stride[1] 
                         + rd2*(kr)->stride[2] + rd3*(kr)->stride[3] 
                         + rd4*(kr)->stride[4];

        kr->array[offset_kr] += k0->array[offset_k0];
   }
    
   return;
}

BACKWARD_FUNC_DEF(r_op_sum_backward){
    (void) kr; (void) idx; 
    for (size_t i = 0; i < 5; i++) {
        if ( (size_t) k1->array[i] == 0 ){ //todo this is an expand to the size of k0
            seed->stride[i] = 0;
            seed->shape[i] = k0->shape[i];
        }
    } 
    return seed;
}