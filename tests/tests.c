#include <stdio.h>
#include <stdbool.h>
#include "../backend/include/interface.h"

#define RESET "\033[0m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
typedef int (*test_func)(void);

int test_basic_add_mul(){

    int errorval = 0;

    // relu(x * x + x*y)
    size_t shape[5] = {1,1,1,1,1};
    tensor *x = empty_tensor(shape, true);
    x->k->array[0] = 1.0;
    tensor *y = empty_tensor(shape, true);
    y->k->array[0] = 4.0;

    tensor *z = add(x, y, true); 
    tensor *w = mul(z, x, true);
    tensor *v = relu(w, true);

    backward(v); 
    
    if (v->grad->array[0] != 1.0) errorval+= 1<<0;
    if (w->grad->array[0] != 1.0) errorval+= 1<<1;
    if (z->grad->array[0] != 1.0) errorval+= 1<<2;
    if (x->grad->array[0] != 6.0) errorval+= 1<<3;
    if (y->grad->array[0] != 1.0) errorval+= 1<<4;
    if (v->k->array[0] != 5.0) errorval+= 1<<5;

    free_tensor(x);
    free_tensor(y);
    free_tensor(w);
    free_tensor(z);
    free_tensor(v);

    return errorval;
}

int test_add(){
    int errorval = -1;
    return errorval;
}

int test_div(){
    int errorval = -1;
    return errorval;
}

int test_sum(){
    int errorval = -1;
    return errorval;
}

int test_sigmoid(){
    int errorval = -1;
    return errorval;
}

int test_view(){
    int errorval = -1;
    return errorval;
}

int test_expand_sum(){
    int errorval = 0;

    size_t shape[5] = {1,1,1,1,4};
    size_t shape_dim[5] = {1,1,1,1,5};
    tensor *dim_e = empty_tensor(shape_dim, false);
    dim_e->k->array[0] = 2.0;
    dim_e->k->array[1] = 1.0;
    dim_e->k->array[2] = 1.0;
    dim_e->k->array[3] = 2.0;
    dim_e->k->array[4] = 4.0;
    tensor *dim_s = empty_tensor(shape_dim, false);
    dim_s->k->array[0] = 0.0;
    dim_s->k->array[1] = 0.0;
    dim_s->k->array[2] = 0.0;
    dim_s->k->array[3] = 0.0;
    dim_s->k->array[4] = 0.0;

    tensor *a = empty_tensor(shape, true);
    tensor *b = empty_tensor(shape, true);
    linspace_kernel_tensor(a->k, 0.0, 4.0-1.0);
    linspace_kernel_tensor(b->k, -4.0, 0.0-1.0);
    tensor *c = mul(a, b, false);
    tensor *d = expand(c, dim_e, false);
    tensor *e = sum(d, dim_s, false);

    int num_backward = 100;
    for (int i =0; i < num_backward; i++){
        backward(e);
    }
   
    if (a->grad->array[0] != -16.0 * num_backward) errorval+= 1<<0;
    if (a->grad->array[1] != -12.0 * num_backward) errorval+= 1<<1;
    if (a->grad->array[2] != -8.0 * num_backward) errorval+= 1<<2;
    if (a->grad->array[3] != -4.0 * num_backward) errorval+= 1<<3;
    if (b->grad->array[0] != 0.0 * num_backward) errorval+= 1<<4;
    if (b->grad->array[1] != 4.0 * num_backward) errorval+= 1<<5;
    if (b->grad->array[2] != 8.0 * num_backward) errorval+= 1<<6;
    if (b->grad->array[3] != 12.0 * num_backward) errorval+= 1<<7;

    if (errorval != 0){
        print_tensor(a);
        print_tensor(b);
    }

    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
    free_tensor(d);
    free_tensor(e);
    free_tensor(dim_e);
    free_tensor(dim_s);

    return errorval;
}

int test_permute(){
    int errorval = -1;
    return errorval;
}

test_func tests[] = {
    test_basic_add_mul,
    test_add,
    test_div,
    test_sum,
    test_sigmoid,
    test_view,
    test_expand_sum,
    test_permute,

};

int num_passed_tests = 0;
int num_not_implemented = 0;

void run_tests(test_func f, int i){
    int passed = f();
    if (passed == 0) {
        num_passed_tests++;
        printf(GREEN "Test %d passed\n" RESET, i);
    } else if (passed == -1) {
        printf(YELLOW "Test %d not implemented\n" RESET, i);
        num_not_implemented++;
    } else {
        printf(RED "Test %d failed with code %d\n" RESET, i, passed);
    }
}

int main(){
    printf("\nRunning LightLemur test suite\n\n");
    int total_tests = sizeof(tests)/sizeof(test_func);
    for (int i = 0; i < total_tests; i++){
        run_tests(tests[i], i);
    }
    
    printf("\nTotal passed tests: %d/%d (with %d not implemented)\n\n", num_passed_tests, total_tests, num_not_implemented);
    return 0;
}