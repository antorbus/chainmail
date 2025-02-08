#include "../../include/tensor.h"


FORWARD_FUNC_DEF(m_op_bmm_forward){
    (void) kr; (void) k0; (void) k1;
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
