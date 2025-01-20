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
        return LemurTensor(_ptr=c_result, _parents=(self,))
    
    def view(self, other):
        if not isinstance(other, LemurTensor):
            raise TypeError("Can't view LemurTensor with non-LemurTensor.")
        c_result = lib.view(self._ptr, other._ptr)
        return LemurTensor(_ptr=c_result, _parents=(self,))
    
    def expand(self, other):
        if not isinstance(other, LemurTensor):
            raise TypeError("Can't expand LemurTensor with non-LemurTensor.")
        c_result = lib.expand(self._ptr, other._ptr)
        return LemurTensor(_ptr=c_result, _parents=(self,))
    
    def permute(self, other):
        if not isinstance(other, LemurTensor):
            raise TypeError("Can't permute LemurTensor with non-LemurTensor.")
        c_result = lib.permute(self._ptr, other._ptr)
        return LemurTensor(_ptr=c_result, _parents=(self,))
    
    def __repr__(self):
        return reprutils._tensor_repr(self._ptr)
    
    def graph(self):
        return reprutils.plot_tensor_graph_parents(self)

    def sigmoid(self):
        c_result = lib.sigmoid(self._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,))

def tensor(data, shape = None, requires_grad=False):
    """
    Creates a new LemurTensor with shape=(1,1,1,1, len(data)).
    """
    
    _shape = shape
    if not shape:
        _shape = (1, 1, 1, 1, len(data))
    t = LemurTensor(shape=_shape, requires_grad=requires_grad)

    k_ptr = t._ptr.contents.k
    c_arr = k_ptr.contents.array 

    for i, val in enumerate(data):
        c_arr[i] = val 

    return t