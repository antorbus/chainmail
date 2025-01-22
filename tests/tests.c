#include <stdio.h>
#include <stdbool.h>
#include "../backend/include/interface.h"

#define RESET "\033[0m"
#define RED "\033[31m"
#define GREEN "\033[32m"

typedef int (*test_func)(void);

int test_basic(){
    size_t shape[5] = {1,1,1,1,1};
    tensor *x = empty_tensor(shape, true);
    x->k->array[0] = 1.0;
    tensor *y = empty_tensor(shape, true);
    y->k->array[0] = 4;
    tensor *z = add(x, y, true); 
    tensor *w = mul(z, x, true);
    tensor *v = relu(w, true);
    backwards(v); 
    free_tensor(x);
    free_tensor(y);
    free_tensor(w);
    free_tensor(z);
    free_tensor(v);
    return 1;
}

test_func tests[] = {
    test_basic,
    test_basic,
    test_basic,
    test_basic,
    test_basic,
};

int num_passed_tests = 0;

void run_tests(test_func f, int i){
    printf("Running test %d...\n", i);
    int passed = f();
    if (passed) {
        num_passed_tests++;
        printf(GREEN "Test %d passed\n" RESET, i);
    } else {
        printf(RED "Test %d failed\n" RESET, i);
    }
}

int main(){
    printf("\nRunning LightLemur test suite\n\n");
    int total_tests = sizeof(tests)/sizeof(test_func);
    for (int i = 0; i < total_tests; i++){
        run_tests(tests[i], i);
    }
    
    printf("\nTotal passed tests: %d/%d\n\n", num_passed_tests, total_tests);
    return 0;
}