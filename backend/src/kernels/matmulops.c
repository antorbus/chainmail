#include "../../include/tensor.h"

#define TILE_SIZE 64
#define MIN(a, b) ((a) < (b) ? (a) : (b))

//A --> i k
//B --> k j
//A @ B = C --> i j 
//grad wrt A --> C @ B^T
//grad wrt B --> A^T @ C

//TODO: NEED TWO more kernels with transpose A and transpose B
//make this by adding FLAGS to forward/backward func def (also stored in graph)
//or just make more kernels (prob better KISS)

FORWARD_FUNC_DEF(m_op_bmm_forward){
    size_t i = kr->shape[3];
    size_t j = kr->shape[4];
    size_t k = k0->shape[4];
    size_t bs = kr->shape[0] * kr->shape[1] * kr->shape[2];
   
    // bs x (i x k) @ bs x (k x j) --> bs x (i x j)
    lemur_float (*A)[i][k] = (lemur_float (*)[i][k]) k0->array;
    lemur_float (*B)[k][j] = (lemur_float (*)[k][j]) k1->array;
    lemur_float (*C)[i][j] = (lemur_float (*)[i][j]) kr->array;

    //do ikj for better locality 
    #pragma omp parallel for 
    for (size_t _b = 0; _b < bs; _b++){

        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (size_t _i = 0; _i < i; _i += TILE_SIZE){
            for (size_t _k = 0; _k < k; _k += TILE_SIZE){
                for (size_t _j = 0; _j < j; _j += TILE_SIZE){
                    
                    for (size_t ii = _i; ii < MIN(_i + TILE_SIZE, i); ii++) {
                        for (size_t kk = _k; kk < MIN(_k + TILE_SIZE, k); kk++) {
                            #pragma omp simd
                            for (size_t jj = _j; jj < MIN(_j + TILE_SIZE, j); jj++) {
                                C[_b][ii][jj] += A[_b][ii][kk] * B[_b][kk][jj];
                            }
                        }
                    }

                }
            }
        }
    }

}

BACKWARD_FUNC_DEF(m_op_bmm_backward){
    (void) kr; (void) k0; (void) k1; (void) seed; (void) idx;
    return NULL;
}

FORWARD_FUNC_DEF(m_op_bcmm_forward){
    (void) kr; (void) k0; (void) k1;

}

BACKWARD_FUNC_DEF(m_op_bcmm_backward){
    (void) kr; (void) k0; (void) k1; (void) seed; (void) idx;
    return NULL;
}

FORWARD_FUNC_DEF(m_op_bmm_fast_forward){
    (void) kr; (void) k0; (void) k1;

}

BACKWARD_FUNC_DEF(m_op_bmm_fast_backward){
    (void) kr; (void) k0; (void) k1; (void) seed; (void) idx;
    return NULL;
}

FORWARD_FUNC_DEF(m_op_bcmm_fast_forward){
    (void) kr; (void) k0; (void) k1;

}

BACKWARD_FUNC_DEF(m_op_bcmm_fast_backward){
    (void) kr; (void) k0; (void) k1; (void) seed; (void) idx;
    return NULL;
}
