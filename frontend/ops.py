from frontend.ptensor import *

def isclose(a, b, rtol=1e-05, atol=1e-08):
    _ptr = lib.isclose(a._ptr, b._ptr, rtol, atol)
    return LemurTensor(_ptr=_ptr, _parents=(a, b))