#include <stdio.h>
#include <stdbool.h>
#include "../backend/include/interface.h"


int main(){


    // t0 * t1 + relu(t2) 


    size_t shape[5] = {1,1,1,1,1};
    
    tensor *t0 = empty_tensor(shape, true);
    t0->k->array[0] = 1;

    tensor *t1 = empty_tensor(shape, true);
    t1->k->array[0] = 12.0;

    tensor *t2 = empty_tensor(shape, true);
    t2->k->array[0] = -15.0;

    tensor *t3 = mul(t0, t1, false); 
    tensor *t4 = relu(t2, false);
    tensor *tf = add(t3, t4, false);
    
    backwards(tf); 

    print_tensor(t0);
    printf("\n");
    print_tensor(t1);
    printf("\n");
    print_tensor(t2);
    printf("\n");
    print_tensor(tf);

    free_tensor(t0);
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(t3);
    free_tensor(t4);
    free_tensor(tf);

    return 0;
}