from frontend.ptensor import *
### Tensor Creation ###

def full(shape : tuple[int, int, int, int, int], 
         fill_value : lemur_float, 
         requires_grad : bool = False) -> LemurTensor:
    
    t = empty(shape, requires_grad=requires_grad)
    lib.memset_kernel_tensor(t._ptr.contents.k, ctypes.c_float(fill_value))
    return t

def arange(end : lemur_float, 
           start : lemur_float = 0.0, 
           step : int = 1, 
           requires_grad : bool = False) -> LemurTensor:
    
    if step == 0:
        raise ValueError("Step must not be zero.")
    steps = int((end - 1 - start) / step + 1)
    return linspace(start, start + (steps - 1) * step, steps, requires_grad=requires_grad)

def linspace(start : lemur_float, 
             end : lemur_float, 
             steps : int, 
             requires_grad : bool = False) -> LemurTensor:
    
    if steps <= 0:
        raise ValueError("Steps must be a positive integer.")
    t = empty((1,1,1,1,steps), requires_grad=requires_grad)
    lib.linspace_kernel_tensor(t._ptr.contents.k, ctypes.c_float(start), ctypes.c_float(end))
    return t

def zeros(shape : tuple[int, int, int, int, int], 
          requires_grad : bool = False) -> LemurTensor:
    
    return full(shape, 0.0, requires_grad=requires_grad) 

def ones(shape : tuple[int, int, int, int, int], 
         requires_grad : bool = False) -> LemurTensor:
    
    return full(shape, 1.0, requires_grad=requires_grad)

### Tensor Creation ###

def init_seed(seed : int) -> None: #TODO should this be moved?
    lib.init_seed(ctypes.c_uint(seed))

def rand(shape : tuple[int, int, int, int, int], 
         low : lemur_float = 0.0, 
         high : lemur_float = 1.0, 
         requires_grad : bool = False) -> LemurTensor:
    
    t = empty(shape, requires_grad=requires_grad)
    lib.random_uniform_kernel_tensor(t._ptr.contents.k, ctypes.c_float(low), ctypes.c_float(high))
    return t

def randn(shape : tuple[int, int, int, int, int], 
          mean : lemur_float = 0.0, 
          std : lemur_float = 1.0, 
          requires_grad : bool = False) -> LemurTensor:
    
    t = empty(shape, requires_grad=requires_grad)
    lib.random_normal_kernel_tensor(t._ptr.contents.k, ctypes.c_float(mean), ctypes.c_float(std))
    return t

    