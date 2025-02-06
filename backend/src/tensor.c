#include "../include/tensor.h"
#include "../include/interface.h"




void set_contiguous_stride(kernel_tensor * k){
    k->stride[4] = 1;
    for (int i = 3; i >= 0; i--) {
        k->stride[i] = k->stride[i + 1] * (int64_t) k->shape[i + 1];
    }
}

lemur_float * lemur_alloc(size_t length){
    size_t size_in_bytes = length*sizeof(lemur_float);
    size_t alignment = (size_in_bytes > 1024) ? 64 : 16;  
    size_t aligned_size = (size_in_bytes + alignment - 1) & ~(alignment - 1);
    lemur_float * arr;
    arr = (lemur_float *)aligned_alloc(alignment, aligned_size);
    if (arr == NULL){
        perror("aligned_alloc failed");
    }
    return arr;
}


kernel_tensor * create_seed_kernel_tensor(){
    kernel_tensor *seed = (kernel_tensor *) malloc(sizeof(kernel_tensor));
    seed->array = lemur_alloc(1);
    seed->length = 1;
    for (size_t i = 0; i < 5; i++){
        seed->shape[i] = 1;
    }
    set_contiguous_stride(seed);
    seed->computed = true;
    seed->array[0] = 1.0;
    seed->shallow = false;
    return seed;
}

bool is_tensor_scalar(tensor *t){
    if (t == NULL) return false;
    if ((t->k->shape[0] == 1) && 
        (t->k->shape[1] == 1) && 
        (t->k->shape[2] == 1) && 
        (t->k->shape[3] == 1) && 
        (t->k->shape[4] == 1) ){
            return true;
        }
    return false;
}

void backward(tensor * t){     
    if (is_tensor_scalar(t) == true){
        if (t->requires_grad == false){
            fprintf(stderr, "backward can only be called on a tensors that require grad\n");
            return;
        }
        kernel_tensor *seed = create_seed_kernel_tensor();
        //derive(t, seed);
        if (t->grad != NULL){
            b_op_add_forward(t->grad, t->grad, seed);
        }
        if (t->comes_from != NULL){
            kernel_backward(t, seed);
        } else {
            free_kernel_tensor(seed); //frees leaf gradients
        }
    } else{
        fprintf(stderr, "backwards can only be called on a leaf (scalar) tensors\n");
    }
}

void free_kernel_tensor(kernel_tensor *k){
    if (k != NULL){
        if ((k->array != NULL) && (k->shallow == false)){
            free(k->array);
        }
        free(k);
    }
}

void free_tensor(tensor *t){
    if (t->k != NULL){
        free_kernel_tensor(t->k);
    }
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
    k->array = lemur_alloc(k->length);
    set_contiguous_stride(k);
    k->computed = false;
    k->shallow = false;
    return k;
}

kernel_tensor * empty_contiguous_kernel_tensor_like(kernel_tensor *k){
    kernel_tensor *k1 = empty_contiguous_kernel_tensor(k->shape);
    return k1;
}

kernel_tensor * empty_kernel_tensor_like(kernel_tensor *k){
    kernel_tensor *k1 = (kernel_tensor *)malloc(sizeof(kernel_tensor));
    k1->array = lemur_alloc(k->length);
    k1->length = k->length;
    memcpy(k1->shape, k->shape, 5 * sizeof(size_t));
    memcpy(k1->stride, k->stride, 5 * sizeof(size_t));
    k1->shallow = false;
    return k1;
}

kernel_tensor * kernel_tensor_shallow_copy(kernel_tensor *k){
    kernel_tensor *k1 = (kernel_tensor *)malloc(sizeof(kernel_tensor));
    k1->array = k->array;
    k1->length = k->length;
    k1->shallow = true;
    memcpy(k1->shape, k->shape, 5 * sizeof(size_t));
    memcpy(k1->stride, k->stride, 5 * sizeof(size_t));
    return k1;
}

void memset_kernel_tensor(kernel_tensor * k, lemur_float val){
    if (k == NULL){
        perror("Error: tried to memset NULL kernel tensor");
        return;
    }
    if (val = 0.0){
        memset(k->array, 0, k->length * sizeof(lemur_float));
        return;
    }
    #pragma omp parallel for simd
    for (size_t i = 0; i < k->length; i++){
        k->array[i] = val;
    }
} 

tensor * empty_tensor(size_t shape[5], bool requires_grad, bool retains_grad){
    tensor *t = (tensor *)malloc(sizeof(tensor));
    t->comes_from = NULL;
    t->requires_grad = requires_grad;
    t->grad = NULL;
    if (retains_grad){
        if (requires_grad == false){
            fprintf(stderr, "Error. Requires_grad must be true if retains_grad is true.");
            free_tensor(t);
            return NULL;
        }
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

kernel_tensor * dim_kernel_tensor_from(size_t shape[5]){
    kernel_tensor *k = (kernel_tensor *)malloc(sizeof(kernel_tensor));
    memcpy(k->shape, shape, 5 * sizeof(size_t));
    k->length = 0;
    k->array = NULL;
    for (size_t i =0; i<5; i++){
        k->stride[i] = 0;
    }
    k->computed = false;
    return k;
}

tensor * dim_tensor_from(size_t shape[5]){
    kernel_tensor *k = dim_kernel_tensor_from(shape);
    return tensor_from(k, NULL, false, NULL);
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
        printf("%ld", (long) k->stride[i]);
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

void set_reduced_shape(size_t reduced_shape[5], size_t original_shape[5], lemur_float dims[5]) {
    for (size_t i = 0; i < 5; i++) {
        size_t d = (size_t)dims[i];
        reduced_shape[i] = (d == 1) ? original_shape[i] : 1;
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

void inplace_contiguous_kernel_tensor(kernel_tensor *k){
    if (k == NULL){
        fprintf(stderr, "Error: attempting to call inplace_contiguous_kernel_tensor on NULL kernel tensor");
        return;
    }
    if (is_contiguous(k)){
        return;
    }
    lemur_float* prev_array = k->array;
    int64_t prev_stride[5];
    memcpy(prev_stride, k->stride, 5 * sizeof(int64_t));
    set_contiguous_stride(k);
    k->length = get_alleged_length(k->shape);
    k->array = lemur_alloc(k->length);
    
    KERNEL_TENSOR_5D_LOOP_START(k){
        size_t offset_k = KERNEL_TENSOR_GET_OFFSET(k);
        size_t offset_prev_k = d0*prev_stride[0] + d1*prev_stride[1] 
                             + d2*prev_stride[2] + d3*prev_stride[3] + d4*prev_stride[4];
        k->array[offset_k] = prev_array[offset_prev_k];
    }
    if (k->shallow == false){
        free(prev_array);
    }
    k->shallow = false;
}

kernel_tensor * contiguous_deepcopy_kernel_tensor(kernel_tensor *k){
    kernel_tensor *kc = empty_contiguous_kernel_tensor_like(k);
    if (is_contiguous(k) == false){
        KERNEL_TENSOR_5D_LOOP_START(kc){
            size_t offset_kc = KERNEL_TENSOR_GET_OFFSET(kc);
            size_t offset_k= KERNEL_TENSOR_GET_OFFSET(k);
            kc->array[offset_kc] =  k->array[offset_k];
        }
    } else {
        memcpy(kc->array, k->array, k->length * sizeof(lemur_float));
    }
    return kc;
}


bool is_initialize_random = false;
unsigned int _seed = 0;
void init_seed(unsigned int seed){
    if (seed == 0){
            _seed = time(NULL);
        } else {
            _seed = seed;
        }
    srand(_seed);
    is_initialize_random = true;
    printf("seed = %u\n", _seed);
}

void init_random() {
    if (!is_initialize_random){
        init_seed(0);
    }
}
void random_uniform_kernel_tensor(kernel_tensor *k, lemur_float min, lemur_float max) {
    #pragma omp parallel for
    for (size_t i = 0; i < k->length; i++) {
        k->array[i] = min + (lemur_float)rand() / (lemur_float)RAND_MAX * (max - min);
    }
}

void random_normal_kernel_tensor(kernel_tensor *k, lemur_float mean, lemur_float std) {
    #pragma omp parallel for
    for (size_t i = 0; i < k->length; i++) {
        lemur_float u1 = (lemur_float)rand() / (lemur_float)RAND_MAX;
        lemur_float u2 = (lemur_float)rand() / (lemur_float)RAND_MAX;
        k->array[i] = mean + std * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    }
}


void linspace_kernel_tensor(kernel_tensor *k, lemur_float start, lemur_float end){
    if (k->length == 1){
        k->array[0] = start;
        return;
    }
    lemur_float step_size = (end - start) / (k->length - 1);
    for (size_t i = 0; i < k->length; i++){
        k->array[i] = start + i * step_size;
    }
}