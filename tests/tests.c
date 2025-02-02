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
    
    if (v->grad->array[0] != 1.0) errorval+= 1;
    if (w->grad->array[0] != 1.0) errorval+= 2;
    if (z->grad->array[0] != 1.0) errorval+= 4;
    if (x->grad->array[0] != 6.0) errorval+= 8;
    if (y->grad->array[0] != 1.0) errorval+= 16;
    if (v->k->array[0] != 5.0) errorval+= 32;

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

int test_expand(){
    int errorval = -1;
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
    test_expand,
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