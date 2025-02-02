#include "../../include/tensor.h"

#define _add(a, b) a + b
#define _mul(a, b) a * b
#define _sub(a, b) a - b
#define _div(a, b) a / b
#define _neg(a) -1.0 * a

FORWARD_FUNC_DEF(b_op_add_forward){
    BINARY_CONTIGUOUS_ELEMENTWISE_OP_SIMD(kr, k0, k1, _add);
}

BACKWARD_FUNC_DEF(b_op_add_backward){
    (void) kr; (void) k0; (void) k1; (void) idx;
    return seed;
}

FORWARD_FUNC_DEF(b_op_mul_forward){
    BINARY_CONTIGUOUS_ELEMENTWISE_OP_SIMD(kr, k0, k1, _mul);
}

BACKWARD_FUNC_DEF(b_op_mul_backward){
    (void) kr;
    kernel_tensor *k;
    if (idx == 0){
        k = k1;
    }
    else{
        k = k0;
    }
    BINARY_CONTIGUOUS_ELEMENTWISE_OP_SIMD(seed, seed, k, _mul);
    return seed;
}

FORWARD_FUNC_DEF(b_op_sub_forward){
    BINARY_CONTIGUOUS_ELEMENTWISE_OP_SIMD(kr, k0, k1, _sub);
}

BACKWARD_FUNC_DEF(b_op_sub_backward){
    (void) kr; (void) k0; (void) k1;
    if (idx == 0){
        //nothing
    }
    else{
       UNARY_CONTIGUOUS_ELEMENTWISE_OP_SIMD(seed, seed, _neg);
    }
    return seed;
}


FORWARD_FUNC_DEF(b_op_division_forward){
    BINARY_CONTIGUOUS_ELEMENTWISE_OP_SIMD(kr, k0, k1, _div);
}

BACKWARD_FUNC_DEF(b_op_division_backward){
    (void) k0;
    if (idx == 0) {
        BINARY_CONTIGUOUS_ELEMENTWISE_OP_SIMD(seed, seed, k1, _div);
    }
    else {
        #pragma omp parallel for simd
        for (size_t _i = 0; _i < kr->length; _i++) {                     
          seed->array[_i] = -1.0 * seed->array[_i] * (kr->array[_i] / (k1->array[_i]));
        }
    }
    return seed;
}