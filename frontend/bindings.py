import ctypes
import os
import platform


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
        ("shallow",       ctypes.c_bool),  
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

lib.init_random_uniform_kernel_tensor.argtypes = [ctypes.POINTER(KernelTensor), ctypes.c_float, ctypes.c_float] #lemur_float
lib.init_random_uniform_kernel_tensor.restype = None 

lib.memset_kernel_tensor.argtypes = [ctypes.POINTER(KernelTensor), ctypes.c_float] #lemur_float
lib.memset_kernel_tensor.restype = None 

lib.get_op_name.argtypes = [ctypes.c_int]
lib.get_op_name.restype  = ctypes.c_char_p

# tensor* empty_tensor(size_t shape[5], bool retain_grad);
lib.empty_tensor.argtypes = [(ctypes.c_size_t * 5), ctypes.c_bool]
lib.empty_tensor.restype  = ctypes.POINTER(Tensor)

# void free_tensor(tensor* t);
lib.free_tensor.argtypes = [ctypes.POINTER(Tensor)]
lib.free_tensor.restype  = None

# void backwards(tensor* t);
lib.backwards.argtypes = [ctypes.POINTER(Tensor)]
lib.backwards.restype  = None

# tensor* mul(tensor* t0, tensor* t1, bool retain_grad);
lib.mul.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_bool]
lib.mul.restype  = ctypes.POINTER(Tensor)

# tensor* division(tensor* t0, tensor* t1, bool b);
lib.division.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_bool]
lib.division.restype  = ctypes.POINTER(Tensor)

# tensor* add(tensor* t0, tensor* t1, bool retain_grad);
lib.add.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_bool]
lib.add.restype  = ctypes.POINTER(Tensor)

# tensor* relu(tensor* t0, bool retain_grad);
lib.relu.argtypes = [ctypes.POINTER(Tensor), ctypes.c_bool]
lib.relu.restype  = ctypes.POINTER(Tensor)

# tensor* exponential(tensor* t0, bool retain_grad);
lib.exponential.argtypes = [ctypes.POINTER(Tensor), ctypes.c_bool]
lib.exponential.restype  = ctypes.POINTER(Tensor)

# tensor* power(tensor* t0, tensor* t1, bool retain_grad);
lib.power.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_bool]
lib.power.restype  = ctypes.POINTER(Tensor)

# tensor* sigmoid(tensor* t0, bool retain_grad);
lib.sigmoid.argtypes = [ctypes.POINTER(Tensor), ctypes.c_bool]
lib.sigmoid.restype  = ctypes.POINTER(Tensor)

#tensor * sum(tensor *t0, tensor *dim_data, bool retain_grad)
lib.sum.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_bool]
lib.sum.restype  = ctypes.POINTER(Tensor)

#tensor * view(tensor *t0, tensor *dim_data)
lib.view.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
lib.view.restype  = ctypes.POINTER(Tensor)

#tensor * expand(tensor *t0, tensor *dim_data)
lib.expand.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
lib.expand.restype  = ctypes.POINTER(Tensor)

#tensor * permute(tensor *t0, tensor *dim_data)
lib.permute.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
lib.permute.restype  = ctypes.POINTER(Tensor)


