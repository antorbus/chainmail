from typing import Optional
import ctypes
from frontend.bindings import lib
import frontend.reprutils as reprutils



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

    @property
    def grad(self):
        print(reprutils._format_kernel_tensor(self._ptr.contents.grad))

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
    
    def __truediv__(self, other):
        if not isinstance(other, LemurTensor):
            raise TypeError("Can't divide LemurTensor with non-LemurTensor.")
        c_result = lib.division(self._ptr, other._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self, other))
    
    def sum(self, other):
        if not isinstance(other, LemurTensor):
            raise TypeError("Can't sum LemurTensor with non-LemurTensor dim.")
        c_result = lib.sum(self._ptr, other._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,other))
    
    def view(self, other):
        if not isinstance(other, LemurTensor):
            raise TypeError("Can't view LemurTensor with non-LemurTensor.")
        c_result = lib.view(self._ptr, other._ptr)
        return LemurTensor(_ptr=c_result, _parents=(self,other))
    
    def expand(self, other):
        if not isinstance(other, LemurTensor):
            raise TypeError("Can't expand LemurTensor with non-LemurTensor.")
        c_result = lib.expand(self._ptr, other._ptr)
        return LemurTensor(_ptr=c_result, _parents=(self, other))
    
    def permute(self, other):
        if not isinstance(other, LemurTensor):
            raise TypeError("Can't permute LemurTensor with non-LemurTensor.")
        c_result = lib.permute(self._ptr, other._ptr)
        return LemurTensor(_ptr=c_result, _parents=(self, other))
    
    def __repr__(self):
        return reprutils._tensor_repr(self._ptr)
    
    @property
    def graph(self):
        print(reprutils.plot_tensor_graph_parents(self))

    def sigmoid(self):
        c_result = lib.sigmoid(self._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,))

def empty(shape, requires_grad=False):
    t = LemurTensor(shape=shape, requires_grad=requires_grad)
    return t

def _infer_shape(data):
    if not isinstance(data, list):
        return []
    
    if len(data) == 0:
        return [0]
    
    first_sub_shape = _infer_shape(data[0])

    top_shape = [len(data)] + first_sub_shape
    
    for sub in data[1:]:
        sub_shape = _infer_shape(sub)
        if sub_shape != first_sub_shape:
            raise ValueError("Inconsistent dimensions encountered in nested list.")
    return top_shape

def _flatten_data(data):
    if not isinstance(data, list):
        return [data]
    flat = []
    for sub in data:
        flat.extend(_flatten_data(sub))
    return flat

def tensor(data, requires_grad=False):
    
    inferred_shape = _infer_shape(data)  # e.g. [2, 3, 4]
    if len(inferred_shape) > 5:
        raise ValueError("Data has more than 5 dimensions, which is not supported.")
    
    pad_length = 5 - len(inferred_shape)
    final_shape = [1] * pad_length + inferred_shape 

    t = empty(shape=final_shape, requires_grad=requires_grad)

    flat_data = _flatten_data(data)

    k_ptr = t._ptr.contents.k
    c_arr = k_ptr.contents.array

    if len(flat_data) != (final_shape[0] * 
                          final_shape[1] * 
                          final_shape[2] *
                          final_shape[3] *
                          final_shape[4]):
        raise ValueError("Number of elements in `data` does not match the tensor's shape.")
    
    for i, val in enumerate(flat_data):
        c_arr[i] = val

    return t

def full(shape, fill_value, requires_grad=False):
    t = empty(shape, requires_grad=requires_grad)
    lib.memset_kernel_tensor(t._ptr.contents.k, ctypes.c_float(fill_value))
    return t

def arange(end, start=0, step=1, requires_grad=False):
    if step == 0:
        raise ValueError("Step must not be zero.")

    steps = int((end - 1 - start) / step + 1)

    return linspace(start, start + (steps - 1) * step, steps, requires_grad=requires_grad)

def linspace(start, end, steps, requires_grad=False):
    if steps <= 0:
        raise ValueError("Steps must be a positive integer.")

    if steps == 1:
        data = [start]
    else:
        step_size = (end - start) / (steps - 1)
        data = [start + i * step_size for i in range(steps)]
    return tensor(data, requires_grad=requires_grad)

def zeros(shape, requires_grad=False):
    return full(shape, 0.0, requires_grad=requires_grad) 

def ones(shape, requires_grad=False):
    return full(shape, 1.0, requires_grad=requires_grad)
