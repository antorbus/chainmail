#include <stdio.h>
#include <stdbool.h>
#include "backend/include/tensor.h"


int main(){


    // t0 * t1 + t2 


    size_t shape[5] = {1,1,1,1,1};
    
    tensor *t0 = empty_tensor(shape, true);
    t0->k->array[0] = 2;

    tensor *t1 = empty_tensor(shape, true);
    t1->k->array[0] = 12.0;

    tensor *t2 = empty_tensor(shape, true);
    t2->k->array[0] = -15.0;

    tensor *t3 = binary_forward(OP_MUL, t0, t1, false);
    tensor *t4 = binary_forward(OP_ADD, t3, t2, false);
    
    backwards(t4); 

    print_tensor(t0);
    print_tensor(t1);
    print_tensor(t2);
    print_tensor(t4);

    free_tensor(t0);
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(t3);
    free_tensor(t4);

    return 0;
}