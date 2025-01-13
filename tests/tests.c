#include <stdio.h>
#include <stdbool.h>
#include "../backend/include/interface.h"


int main(){

    // x = tensor([1.0], requires_grad=True)
    // y = tensor([4.0], requires_grad=True)

    // z = x + y  
    // w = z * x  
    // v = w.relu() 

    // v.backward() 

    size_t shape[5] = {1,1,1,1,1};
    
    tensor *x = empty_tensor(shape, true);
    x->k->array[0] = 1.0;

    tensor *y = empty_tensor(shape, true);
    y->k->array[0] = 4;

    tensor *z = add(x, y, false); 
    tensor *w = mul(z, x, false);
    tensor *v = relu(w, false);
    
    backwards(v); 

    print_tensor(y);
    printf("\n");
    

    free_tensor(x);
    free_tensor(y);
    free_tensor(w);
    free_tensor(z);
    free_tensor(v);

    return 0;
}