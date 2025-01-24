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

    @staticmethod
    def _convert_to_tensor(obj):
        if isinstance(obj, (tuple, list)):
            obj = tensor(obj)
        if not isinstance(obj, LemurTensor):
            raise TypeError("Input must be a LemurTensor, tuple, or list.")
        return obj
    
    def _process_args(self, *args):
        if len(args) == 1 and isinstance(args[0], (tuple, list, LemurTensor)):
            return self._convert_to_tensor(args[0])
        return self._convert_to_tensor(list(args))
    
    def backward(self):
        lib.backward(self._ptr)

    def stride(self):
        return [self._ptr.contents.k.contents.stride[i] for i in range(5)]
    
    def is_shallow(self):
        return self._ptr.contents.k.contents.shallow

    def is_contiguous(self):
        return lib.is_contiguous(self._ptr.contents.k)
    
    @property
    def memory_length(self):
        return self._ptr.contents.k.contents.length
    
    @property
    def shape(self):
        return [self._ptr.contents.k.contents.shape[i] for i in range(5)]

    def numel(self):
        shape = self.shape
        return shape[0] * shape[1] * shape[2] * shape[3] * shape[4]
    
    def ndimension(self):  
        return 5
    
    def flatten(self, dim=4):
        total_elements = self.numel()
        view_dim = [1,1,1,1,1]
        view_dim[dim] = total_elements
        return self.view(view_dim)

    @property
    def grad(self):
        return reprutils._format_kernel_tensor(self._ptr.contents.grad)

    def relu(self):
        c_result = lib.relu(self._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,))

    def __add__(self, other):
        if not isinstance(other, LemurTensor):
            raise TypeError("Can't add LemurTensor with non-LemurTensor.")
        c_result = lib.add(self._ptr, other._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self, other))
    
    def __sub__(self, other):
        if not isinstance(other, LemurTensor):
            raise TypeError("Can't subtract LemurTensor with non-LemurTensor.")
        c_result = lib.sub(self._ptr, other._ptr, False)
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
    
    def exp(self):
        c_result = lib.exponential(self._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,))
    
    def __pow__(self, other):
        if not isinstance(other, LemurTensor):
            if isinstance(other, float) or isinstance(other, int):
                other = tensor([float(other)])
            else:
                raise TypeError("Can't take LemurTensor to non-float or non-LemurTensor exponent.")
        c_result = lib.power(self._ptr, other._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self, other))
    
    def sum(self, *args):
        if not args: 
            dims = [0,0,0,0,0]
        else:
            dims = [1,1,1,1,1]
        for d in args:
            dims[d] = 0
        other = self._convert_to_tensor(dims)
        c_result = lib.sum(self._ptr, other._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,other))
    
    def view(self, *args):
        other = self._process_args(*args)
        c_result = lib.view(self._ptr, other._ptr)
        return LemurTensor(_ptr=c_result, _parents=(self, other))

    def expand(self, *args):
        other = self._process_args(*args)
        c_result = lib.expand(self._ptr, other._ptr)
        return LemurTensor(_ptr=c_result, _parents=(self, other))

    def permute(self, *args):
        other = self._process_args(*args)
        c_result = lib.permute(self._ptr, other._ptr)
        return LemurTensor(_ptr=c_result, _parents=(self, other))
    
    def __repr__(self):
        return reprutils._tensor_repr(self._ptr)
    
    @property
    def graph(self):
        return reprutils.plot_tensor_graph_parents(self)

    def sigmoid(self):
        c_result = lib.sigmoid(self._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,))
    
    def compile(self):
        lib.compile(self._ptr)

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
