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
    tensor *x = empty_tensor(shape, true, true);
    x->k->array[0] = 1.0;
    tensor *y = empty_tensor(shape, true, true);
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

int test_free(){
    int errorval = 0;
    size_t shape_dim[5] = {1,1,1,1,5};
    tensor *dim_s = empty_tensor(shape_dim, false, false);
    dim_s->k->array[0] = 0.0;
    dim_s->k->array[1] = 0.0;
    dim_s->k->array[2] = 0.0;
    dim_s->k->array[3] = 0.0;
    dim_s->k->array[4] = 0.0;
    for (size_t i = 0; i < 128; i++){
        tensor *x = empty_tensor((size_t[5]){16,16,16,16,16}, true, true);
        tensor *y = sum(x, dim_s, true);
        backward(y);
        if (y->grad->array[0] != 1.0){
            errorval = 1;
        }
        if (x->grad->array[0] != 1.0){
            errorval = 2;
        }
        free_tensor(y);
        free_tensor(x);
    }
    free_tensor(dim_s);
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

int test_sum_simple(){
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

int test_relu(){
    int errorval = 0;
    size_t size = 1<<19;
    size_t shape[5] = {1,1,1,1,size};
    size_t shape_dim[5] = {1,1,1,1,5};
    tensor *dim_s = empty_tensor(shape_dim, false, false);
    dim_s->k->array[0] = 0.0;
    dim_s->k->array[1] = 0.0;
    dim_s->k->array[2] = 0.0;
    dim_s->k->array[3] = 0.0;
    dim_s->k->array[4] = 0.0;
    tensor *a = empty_tensor(shape, true, true);
    linspace_kernel_tensor(a->k, -1.0, 1.0);
    tensor *ar = relu(a, false);
    tensor *c = sum(ar, dim_s, false);

    size_t num_backward = 100;
    for (size_t i =0; i < num_backward; i++){
        backward(c);
    }

    for (size_t i =0; i < size/2; i++){
        if (a->grad->array[i] != 0.0){
            errorval = 1;
        }
    }

    for (size_t i =size/2; i < size; i++){
        if (a->grad->array[i] != (lemur_float) num_backward){
            errorval = 1;
        }
    }

    free_tensor(dim_s);
    free_tensor(a);
    free_tensor(ar);
    free_tensor(c);

    return errorval;
}

int test_expand_sum(){
    int errorval = 0;

    size_t shape[5] = {1,1,1,1,4};
    size_t shape_dim[5] = {1,1,1,1,5};
    tensor *dim_e = empty_tensor(shape_dim, false, false);
    dim_e->k->array[0] = 2.0;
    dim_e->k->array[1] = 1.0;
    dim_e->k->array[2] = 1.0;
    dim_e->k->array[3] = 2.0;
    dim_e->k->array[4] = 4.0;
    tensor *dim_s = empty_tensor(shape_dim, false, false);
    dim_s->k->array[0] = 0.0;
    dim_s->k->array[1] = 0.0;
    dim_s->k->array[2] = 0.0;
    dim_s->k->array[3] = 0.0;
    dim_s->k->array[4] = 0.0;

    tensor *a = empty_tensor(shape, true, true);
    tensor *b = empty_tensor(shape, true, true);
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
    int errorval = 0;

    size_t shape_view_perm[5] = {1,1,1,1,5};
    tensor *dim = empty_tensor(shape_view_perm, false, false);
    dim->k->array[0] = 2;
    dim->k->array[1] = 2;
    dim->k->array[2] = 2;
    dim->k->array[3] = 2;
    dim->k->array[4] = 2;

    tensor *dim_perm0 = empty_tensor(shape_view_perm, false, false);
    dim_perm0->k->array[0] = 0;
    dim_perm0->k->array[1] = 1;
    dim_perm0->k->array[2] = 2;
    dim_perm0->k->array[3] = 4;
    dim_perm0->k->array[4] = 3;


    tensor *dim_perm1 = empty_tensor(shape_view_perm, false, false);
    dim_perm1->k->array[0] = 4;
    dim_perm1->k->array[1] = 3;
    dim_perm1->k->array[2] = 2;
    dim_perm1->k->array[3] = 1;
    dim_perm1->k->array[4] = 0;

    tensor *dim_reduce = empty_tensor(shape_view_perm, false, false);
    dim_reduce->k->array[0] = 0;
    dim_reduce->k->array[1] = 0;
    dim_reduce->k->array[2] = 0;
    dim_reduce->k->array[3] = 0;
    dim_reduce->k->array[4] = 0;

    size_t shape[5] = {1,1,1,1,32};
    tensor *a = empty_tensor(shape, true, true);
    linspace_kernel_tensor(a->k, 1.0, 32.0);
    tensor *a_v0 = view(a, dim, false);
    tensor *a_v0_perm = permute(a_v0, dim_perm0, false);
    tensor *a_v1 = view(a, dim, false);
    tensor *c = mul(a_v0_perm, a_v1, false);
    tensor *c_perm = permute(c, dim_perm1, false);
    tensor *c_perm_sum = sum(c_perm, dim_reduce, false);
    
    backward(c_perm_sum);

    lemur_float correct_grad[32] = {2.,  6.,  4.,  8., 10., 14., 12., 16., 18., 22., 20., 24., 26., 30.,
        28., 32., 34., 38., 36., 40., 42., 46., 44., 48., 50., 54., 52., 56.,
        58., 62., 60., 64.};
    
    for (size_t i = 0; i<32; i++){
        if (correct_grad[i] != a->grad->array[i]){
            errorval++;
        }
    }

    free_tensor(dim);
    free_tensor(dim_perm0);
    free_tensor(dim_perm1);
    free_tensor(dim_reduce);
    free_tensor(a);
    free_tensor(a_v0);
    free_tensor(a_v0_perm);
    free_tensor(a_v1);
    free_tensor(c);
    free_tensor(c_perm);
    free_tensor(c_perm_sum);

    return errorval;
}

test_func tests[] = {
    test_basic_add_mul,
    test_add,
    test_div,
    test_sum_simple,
    test_sum,
    test_sigmoid,
    test_free,
    test_view,
    test_expand_sum,
    test_permute,
    test_relu,

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