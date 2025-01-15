#include "../include/tensor.h"
#include "../include/interface.h"


void derive(tensor *t, kernel_tensor *seed){
    if (t->grad != NULL){
        b_op_add_forward(t->grad, t->grad, seed);
    }
    if (t->comes_from != NULL){
        kernel_backward(t, seed);
    } else {
        free_kernel_tensor(seed);
    }
    
}

void set_contiguous_stride(kernel_tensor * k){
    k->stride[4] = 1;
    for (int i = 3; i >= 0; i--) {
        k->stride[i] = k->stride[i + 1] * (int64_t) k->shape[i + 1];
    }
}

kernel_tensor * create_seed_kernel_tensor(){
    kernel_tensor *seed = (kernel_tensor *) malloc(sizeof(kernel_tensor));
    seed->array = (lemur_float *) malloc(sizeof(lemur_float));
    seed->length = 1;
    for (size_t i = 0; i < 5; i++){
        seed->shape[i] = 1;
    }
    set_contiguous_stride(seed);
    seed->computed = true;
    seed->array[0] = 1.0;
    return seed;
}

bool is_tensor_scalar(tensor *t){
    if ((t->k->shape[0] == 1) && 
        (t->k->shape[1] == 1) && 
        (t->k->shape[2] == 1) && 
        (t->k->shape[3] == 1) && 
        (t->k->shape[4] == 1) ){
            return true;
        }
    return false;
}

void backwards(tensor * t){     //TODO: change name???
    if (is_tensor_scalar(t) == true){
        kernel_tensor *seed = create_seed_kernel_tensor();
        derive(t, seed);
    } else{
        fprintf(stderr, "backwards can only be called on a leaf tensor\n");
    }
}

void free_kernel_tensor(kernel_tensor *k){
    free(k->array);
    free(k);
}

void free_tensor(tensor *t){
    free_kernel_tensor(t->k);
    if (t->grad != NULL){
        free_kernel_tensor(t->grad);
    }
    if (t->comes_from != NULL){
        free(t->comes_from);
    }
    free(t);
}

size_t get_alleged_length(size_t shape[5]){
    size_t l = 1;    
    for (size_t i = 0; i < 5; i++){
            l *= shape[i];
        }
    return l;
} 

kernel_tensor * empty_contiguous_kernel_tensor(size_t shape[5]){
    kernel_tensor *k = (kernel_tensor *)malloc(sizeof(kernel_tensor));
    memcpy(k->shape, shape, 5 * sizeof(size_t));
    k->length = get_alleged_length(shape);
    k->array = (lemur_float *)malloc(k->length * sizeof(lemur_float));
    set_contiguous_stride(k);
    k->computed = false;
    return k;
}

kernel_tensor * empty_contiguous_kernel_tensor_like(kernel_tensor *k){
    kernel_tensor *k1 = empty_contiguous_kernel_tensor(k->shape);
    return k1;
}

kernel_tensor * empty_kernel_tensor_like(kernel_tensor *k){
    kernel_tensor *k1 = (kernel_tensor *)malloc(sizeof(kernel_tensor));
    k1->array = (lemur_float *)malloc(k->length*sizeof(lemur_float));
    k1->length = k->length;
    memcpy(k1->shape, k->shape, 5 * sizeof(size_t));
    memcpy(k1->stride, k->stride, 5 * sizeof(size_t));
    return k1;
}

void memset_kernel_tensor(kernel_tensor * k, lemur_float val){
    memset(k->array, *(int*)&val, k->length * sizeof(lemur_float)); 
} 

tensor * empty_tensor(size_t shape[5], bool retain_grad){
    tensor *t = (tensor *)malloc(sizeof(tensor));
    t->comes_from = NULL;
    t->requires_grad = true;
    t->grad = NULL;
    if (retain_grad){
        t->grad = empty_contiguous_kernel_tensor(shape);
        memset_kernel_tensor(t->grad, 0.0);
    }
    t->k = empty_contiguous_kernel_tensor(shape);
    return t;
}

tensor * tensor_from(kernel_tensor *k, expression *comes_from, bool requires_grad, kernel_tensor* grad){
    tensor *t = (tensor *)malloc(sizeof(tensor));
    t->comes_from = comes_from;
    t->requires_grad = requires_grad;
    t->grad = grad;
    t->k = k;
    return t;
}

expression * expression_from(int func, tensor *t0, tensor *t1){
    expression *e = (expression *)malloc(sizeof(expression));
    e->t0 = t0;
    e->t1 = t1;
    e->backward_func = func;
    return e;
}

void print_kernel_tensor(kernel_tensor *k){
if (!k) {
        printf("  [NULL kernel_tensor]\n");
        return;
    }
    
    printf("  kernel_tensor: %p\n", k);
    printf("    length     = %zu\n", k->length);
    printf("    shape      = [");
    for (int i = 0; i < 5; i++) {
        printf("%zu", k->shape[i]);
        if (i < 4) printf(", ");
    }
    printf("]\n");
    
    printf("    stride     = [");
    for (int i = 0; i < 5; i++) {
        printf("%lld", k->stride[i]);
        if (i < 4) printf(", ");
    }
    printf("]\n");
    
    //TODO make is_contiguous
    //printf("    contiguous = %s\n", k->contiguous ? "true" : "false");
    printf("    computed   = %s\n", k->computed   ? "true" : "false");

    // Print all elements in a 5D loop
    // Be cautious, this can print a LOT of lines if shapes are large!
    printf("    data:\n");
    printf("tensor([");
    for (size_t d0 = 0; d0 < k->shape[0]; d0++) {
        printf("[");
        for (size_t d1 = 0; d1 < k->shape[1]; d1++) {
            printf("[");
            for (size_t d2 = 0; d2 < k->shape[2]; d2++) {
                printf("[");
                for (size_t d3 = 0; d3 < k->shape[3]; d3++) {
                    printf("[");
                    for (size_t d4 = 0; d4 < k->shape[4]; d4++) {
                        size_t idx = KERNEL_TENSOR_GET_OFFSET(k);
                        printf("%.4f", k->array[idx]);
                        if (d4 < k->shape[4] - 1) {
                            printf(", ");
                        }
                    }
                    printf("]");
                    if (d3 < k->shape[3] - 1) {
                        printf(",");
                    } else {
                        printf("");
                    }
                }
                printf("]");
                if (d2 < k->shape[2] - 1) {
                    printf(",");
                } else {
                    printf("");
                }
            }
            printf("]");
            if (d1 < k->shape[1] - 1) {
                printf(",");
            } else {
                printf("");
            }
        }
        printf("]");
        if (d0 < k->shape[0] - 1) {
            printf(",");
        } else {
            printf("");
        }
    }
    printf("])\n");

                
}

void print_expression(expression *e){
    if (!e) {
        printf("  [NULL expression]\n");
        return;
    }
    
    printf("  expression:\n");
    printf("    backward_func = %d\n", e->backward_func);
    
    printf("    t0:");
    if (e->t0) printf("      %p\n", e->t0); 
    else
        printf("      [NULL tensor]\n");
    
    printf("    t1:");
    if (e->t1) printf("      %p\n", e->t1);
    else
        printf("      [NULL tensor]\n");
    
   
}

void print_tensor(tensor *t){
    if (!t) {
        printf("[NULL tensor]\n");
        return;
    }
    
    printf("tensor: %p\n", t);
    printf("  requires_grad = %s\n", t->requires_grad ? "true" : "false");
    
    printf("  comes_from:\n");
    print_expression(t->comes_from);
    
    printf("  k:\n");
    print_kernel_tensor(t->k);
    
    printf("  grad:\n");
    print_kernel_tensor(t->grad);
}

bool are_shapes_equal(size_t shape0[5], size_t shape1[5]) {
    for (size_t i = 0; i < 5; i++) {
        if (shape0[i] != shape1[i]) {
            return false; 
        }
    }
    return true; 
}

void set_reduced_shape(size_t reduced_shape[5], size_t original_shape[5], size_t dims[5]) {
    for (size_t i = 0; i < 5; i++) {
        reduced_shape[i] = (dims[i] != 0) ? original_shape[i] : 1;
    }
}

bool is_contiguous(kernel_tensor *k) {
    size_t total_elems = get_alleged_length(k->shape);

    if (k->length != total_elems) {
        return false;
    }

    size_t expected_stride = 1;  
    for (int d = 4; d >= 0; d--) {
        if (k->stride[d] != (int64_t)expected_stride) {
            return false;
        }
        expected_stride *= k->shape[d];
    }

    return true;  
}
