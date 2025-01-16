import ctypes
import os
import platform
from typing import Optional

SYSTEM = platform.system()
if SYSTEM == "Darwin":
    LIB_FILE = "liblightlemur.dylib"  
elif SYSTEM == "Windows":
    LIB_FILE = "liblightlemur.dll"   
else:
    LIB_FILE = "liblightlemur.so"    

lib_path = os.path.join(
    os.path.dirname(__file__),  
    "..",                      
    LIB_FILE
)

lib = ctypes.CDLL(lib_path)

class lemur_float(ctypes.c_float): #TODO
    pass

class KernelTensor(ctypes.Structure):
    _fields_ = [
        ("array",    ctypes.POINTER(lemur_float)),  
        ("length",   ctypes.c_size_t),
        ("shape",    ctypes.c_size_t * 5),            
        ("stride",   ctypes.c_int64  * 5),            
        ("computed", ctypes.c_bool),
    ]

class Tensor(ctypes.Structure):
    pass 

class Expression(ctypes.Structure):
    _fields_ = [
        ("t0",    ctypes.POINTER(Tensor)),  
        ("t1",    ctypes.POINTER(Tensor)),
        ("backward_func", ctypes.c_int),            
    ] 

class Tensor(ctypes.Structure):
    _fields_ = [
        ("k",             ctypes.POINTER(KernelTensor)), 
        ("comes_from",    ctypes.POINTER(Expression)),    
        ("requires_grad", ctypes.c_bool),
        ("grad",          ctypes.POINTER(KernelTensor)),  
    ]

#from interface.h

# tensor* empty_tensor(size_t shape[5], bool retain_grad);
lib.empty_tensor.argtypes = [(ctypes.c_size_t * 5), ctypes.c_bool]
lib.empty_tensor.restype  = ctypes.POINTER(Tensor)

# void free_tensor(tensor* t);
lib.free_tensor.argtypes = [ctypes.POINTER(Tensor)]
lib.free_tensor.restype  = None

# void backwards(tensor* t);
lib.backwards.argtypes = [ctypes.POINTER(Tensor)]
lib.backwards.restype  = None

# tensor* mul(tensor* t0, tensor* t1, bool b);
lib.mul.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_bool]
lib.mul.restype  = ctypes.POINTER(Tensor)

# tensor* add(tensor* t0, tensor* t1, bool b);
lib.add.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_bool]
lib.add.restype  = ctypes.POINTER(Tensor)

# tensor* relu(tensor* t0, bool b);
lib.relu.argtypes = [ctypes.POINTER(Tensor), ctypes.c_bool]
lib.relu.restype  = ctypes.POINTER(Tensor)


#print

def _format_kernel_tensor(k_ptr):
    if not k_ptr:
        return "  [NULL kernel_tensor]\n"

    k = k_ptr.contents
    lines = []
    lines.append(f"  kernel_tensor: 0x{ctypes.addressof(k):x}\n")
    lines.append(f"    length     = {k.length}\n")

    shape = [k.shape[i] for i in range(5)]
    lines.append("    shape      = [" + ", ".join(str(s) for s in shape) + "]\n")

    stride = [k.stride[i] for i in range(5)]
    lines.append("    stride     = [" + ", ".join(str(s) for s in stride) + "]\n")

    lines.append(f"    computed   = {str(k.computed).lower()}\n")

    data_str = []
    data_str.append("tensor([")
    arr = k.array  
    for d0 in range(shape[0]):
        data_str.append("[")
        for d1 in range(shape[1]):
            data_str.append("[")
            for d2 in range(shape[2]):
                data_str.append("[")
                for d3 in range(shape[3]):
                    data_str.append("[")
                    for d4 in range(shape[4]):
                        idx = (d0 * stride[0] +
                               d1 * stride[1] +
                               d2 * stride[2] +
                               d3 * stride[3] +
                               d4 * stride[4])
                        val = float(arr[idx].value)  # a lemur_float
                        data_str.append(f"{val:.4f}")
                        if d4 < shape[4] - 1:
                            data_str.append(", ")
                    data_str.append("]")
                    if d3 < shape[3] - 1:
                        data_str.append(",")
                data_str.append("]")
                if d2 < shape[2] - 1:
                    data_str.append(",")
            data_str.append("]")
            if d1 < shape[1] - 1:
                data_str.append(",")
        data_str.append("]")
        if d0 < shape[0] - 1:
            data_str.append(",")
    data_str.append("])")

    data_str = "".join(data_str)

    lines.append("    data:\n")
    lines.append(f"      {data_str}\n")

    return "".join(lines)


def _format_expression(e_ptr):
    if not e_ptr:
        return "  [NULL expression]\n"
    e = e_ptr.contents
    lines = []
    lines.append("  expression:\n")
    lines.append(f"    backward_func = {e.backward_func}\n")

    if e.t0:
        lines.append(f"    t0:      0x{ctypes.addressof(e.t0.contents):x}\n")
    else:
        lines.append("    t0:      [NULL tensor]\n")

    if e.t1:
        lines.append(f"    t1:      0x{ctypes.addressof(e.t1.contents):x}\n")
    else:
        lines.append("    t1:      [NULL tensor]\n")

    return "".join(lines)


def _tensor_repr(t_ptr):
    """
    Builds the multi-line string replicating print_tensor().
    """
    if not t_ptr:
        return "[NULL tensor]"

    t = t_ptr.contents
    lines = []
    lines.append(f"tensor: 0x{ctypes.addressof(t):x}\n")
    lines.append(f"  requires_grad = {str(t.requires_grad).lower()}\n")

    #lines.append("  comes_from:\n")
    #lines.append(_format_expression(t.comes_from))

    lines.append("  k:\n")
    lines.append(_format_kernel_tensor(t.k))

    lines.append("  grad:\n")
    lines.append(_format_kernel_tensor(t.grad))

    return "".join(lines)


class LemurTensor:
    __slots__ = ("_ptr", "_parents")
    #TODO make note that _parents is needed so that when doing w = w.relu() or similar, GC doesnt mess us up

    def __init__(self, 
             shape: Optional[list[int]] = None, 
             requires_grad: Optional[bool] = False, 
             _ptr = None, 
             _parents = None):
        
        if _ptr is not None:
            self._ptr = _ptr
            self._parents = _parents or ()
        else:
            if shape is None:
                shape = (1,)
            c_shape = (ctypes.c_size_t * 5)(*([1]*5))
            for i, dim in enumerate(shape):
                c_shape[i] = dim

            t_ptr = lib.empty_tensor(c_shape, requires_grad)
            if not t_ptr:
                raise RuntimeError("empty_tensor returned NULL.")
            self._ptr = t_ptr

    def __del__(self):
        if getattr(self, "_ptr", None) is not None:
            lib.free_tensor(self._ptr)
            self._ptr = None

    def backward(self):
        lib.backwards(self._ptr)

    def relu(self):
        c_result = lib.relu(self._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,))

    def __add__(self, other):
        if not isinstance(other, LemurTensor):
            raise TypeError("Can't add LemurTensor with non-LemurTensor.")
        c_result = lib.add(self._ptr, other._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self, other))

    def __mul__(self, other):
        if not isinstance(other, LemurTensor):
            raise TypeError("Can't multiply LemurTensor with non-LemurTensor.")
        c_result = lib.mul(self._ptr, other._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self, other))

    def __repr__(self):
        return _tensor_repr(self._ptr)



def tensor(data, requires_grad=True):
    """
    Creates a new LemurTensor with shape=(1,1,1,1, len(data)).
    """
    arr_len = len(data)
    shape = (1, 1, 1, 1, arr_len)
    t = LemurTensor(shape=shape, requires_grad=requires_grad)

    k_ptr = t._ptr.contents.k
    c_arr = k_ptr.contents.array 

    for i, val in enumerate(data):
        c_arr[i] = val 

    return t