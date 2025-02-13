from typing import Optional
import ctypes
from frontend.bindings import lib, lemur_float
import frontend.reprutils as reprutils
from frontend.ptensor import LemurTensor, empty
from frontend.tensor_creation import rand

class Parameter:

    def __init__(self, tensor: LemurTensor):
        self.tensor = tensor
        self._ptr = lib.create_parameter(tensor._ptr)

        if not self._ptr:
            raise RuntimeError("Failed to create parameter")

    def __del__(self):
        """
        free memory when garbage collected.
        """
        if getattr(self, "_ptr", None) is not None:
            lib.free_parameter(self._ptr)
            self._ptr = None
        
    def __repr__(self):
        return f"Parameter(tensor={self.tensor})"


class Module:
    __slots__ = ("name", "_train_mode", "_children", "_parent", "_parameters") 
    
    def __init__(self,
             name: Optional[str] = None,
             training: bool = True,
             _children: Optional[dict[str, "Module"]] = None,
             _parent: Optional["Module"] = None,
             _parameters: Optional[list[Parameter]] = None
    ):
        
        self._children = _children if _children is not None else {} 
        self._parent = _parent
        self._parameters = _parameters if _parameters is not None else []
        self._train_mode = training
        self.name = name if name is not None else self.__class__.__name__

    def __setattr__(self, name, value):
        """
        Automatically track parameters and submodules when assigned.
        """
        super().__setattr__(name, value)

        if isinstance(value, Parameter) and value not in self._parameters:
            self._parameters.append(value)
        elif isinstance(value, Module) and value not in self._children:
            self._children[name] = value

    def add_module(self, name: str, model: "Module"):
        """
        Add a child module dynamically.
        """
        if not isinstance(model, Module):
            raise TypeError("model must be an instance of Module.")
        self._children[name] = model

    def apply(self, fn):
        """
        Apply a function recursively to every submodule and self.
        """
        for child in self._children.values():
            child.apply(fn)
        fn(self)

    def children(self):
        """
        Return an iterator over immediate children modules.
        """
        return iter(self._children.values())
    
    def parameters(self, recurse=True):
        """
        Return an iterator over parameters.
        """
        for param in self._parameters:
            yield param
        
        if recurse:
            for child in self._children.values():
                yield from child.parameters()
    
    def train(self, mode):
        self._train_mode = mode
        for child in self._children.values():
            child._train_mode(mode)
        return self

    def eval(self):
        self._train_mode(False)

    def forward(self, x: LemurTensor): 
        """
        Defines the core computation performed by the module.
        Takes the input(s) (e.g., a tensor or a batch of data) and computes the output(s) using the module's parameters and submodules.
        Needs to be overridden in subclasses of Module for specific functionality.
        """
        raise NotImplementedError("Forward method must be implemented in the subclass.")
    
    def __call__(self, *args, **kwds):
        """
        Allows the module to be called like a function, 
        triggering hooks (if any) and the forward method.
        """
        output = self.forward(*args, **kwds)
        return output

    def __del__(self):
        """
        free memory when garbage collected.
        """
        for _, child in self._children.items():
            del child

        for param in self._parameters:
            del param

        self._children.clear()
        self._parameters.clear()
        self._parent = None

    def __repr__(self, level=0):
        """
        Represent the module hierarchy.
        """
        indent = "  " * level
        child_repr = "\n".join(child.__repr__(level + 1) for child in self._children.values())
        param_repr = ", ".join(f"Parameter({p.tensor.shape})" for p in self._parameters) if self._parameters else "No Parameters"
        children_str = f"\n{child_repr}" if self._children else ""

        return f"{indent}{self.name}(\n{indent}  Parameters: [{param_repr}]{children_str}\n{indent})"
