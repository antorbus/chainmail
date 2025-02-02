#include "../../include/tensor.h"


FORWARD_FUNC_DEF(m_op_mm_forward){
    (void) kr; (void) k0; (void) k1;
}

BACKWARD_FUNC_DEF(m_op_mm_backward){
    (void) kr; (void) k0; (void) k1; (void) seed; (void) idx;
    return NULL;
}

FORWARD_FUNC_DEF(m_op_bmm_forward){
    (void) kr; (void) k0; (void) k1;

}

BACKWARD_FUNC_DEF(m_op_bmm_backward){
    (void) kr; (void) k0; (void) k1; (void) seed; (void) idx;
    return NULL;
}