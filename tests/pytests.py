import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import lemur
from frontend.module import Module, Parameter
from frontend.tensor_creation import *
from frontend.ptensor import *

def shape_to_list(a: Parameter, b: LemurTensor):
    if isinstance(a, LemurTensor):
        a_shape = [int(a.shape[i]) for i in range(5)]
    else:
        a_shape = [int(a.tensor.shape[i]) for i in range(5)]
    b_shape = [int(b.shape[i]) for i in range(5)]
    return a_shape, b_shape

class TestLightLemur(unittest.TestCase):
    
    def test_basic(self):
        a = lemur.full((1,1,1,1,10), 15, requires_grad=True)
        (a + a).sum().backward()
        real_grad = lemur.full((1,1,1,1,10), 2)
        self.assertTrue((real_grad == a.grad).all(), "Grad check failed")

    def test_gradient_accumulation_basic(self):
        a = lemur.full((3, 3), 3, requires_grad=True)
        loss1 = (a + a).sum()
        loss1.backward()  
        self.assertTrue((a.grad == lemur.full((3, 3), 2)).all(), "First backward call failed")
        loss2 = (a + a + a).sum()
        loss2.backward() 
        expected_grad = lemur.full((3, 3), 2 + 3)  
        self.assertTrue((expected_grad == a.grad).all(), "Gradient accumulation failed")

    def test_retains_grad(self):
        a = lemur.randn((10,10,10,10,10))
        self.assertTrue(a.grad == None, "tensor should not have gradient")
        self.assertTrue(a.requires_grad() == False, "tensor should not have req grad being true")
        self.assertTrue(a.retain_grad() == False, "tensor should not have ret grad being true")
        a.retain_grad_(True)
        expected_grad = lemur.zeros((10,10,10,10,10))  
        self.assertTrue(a.requires_grad() == True, "tensor should not have req grad being false")
        self.assertTrue(a.retain_grad() == True, "tensor should not have ret grad being false")
        self.assertTrue((expected_grad == a.grad).all(), "Gradient is non zero")
        a.retain_grad_(False)
        self.assertTrue(a.grad == None, "tensor should not have gradient")
        self.assertTrue(a.requires_grad() == True, "tensor should not have req grad being false")
        self.assertTrue(a.retain_grad() == False, "tensor should not have ret grad being true")

    def test_sum(self):
        a = lemur.rand((1,2,3,48,512), requires_grad=True)
        c = a.sum(1,4).sum(2,3)
        c.backward()
        c.backward()
        c.backward()
        real_grad = lemur.full((1,2,3,48,512), 3)
        self.assertTrue((real_grad == a.grad).all(), "Grad check failed")

    # testing module.py
    def test_parameter_creation(self):
        """Test Parameter initialization and retrieval."""
        tensor = rand(shape=[1, 10])
        param = Parameter(tensor)
        param_shape, tensor_shape = shape_to_list(param, tensor)
        
        self.assertIsInstance(param, Parameter, "Object is not of type Parameter")
        self.assertIsInstance(param.tensor, LemurTensor, "Object is not of type LemurTensor")
        self.assertEqual(param_shape, tensor_shape, "Shape mismatch")

    def test_module_children(self):
        """Test if child modules are correctly registered."""
        class ParentModule(Module):
            def __init__(self):
                super().__init__()
                self.child = Module()

        parent = ParentModule()
        children = list(parent.children())

        self.assertEqual(len(children), 1, "Not expected number of children")
        self.assertIsInstance(children[0], Module, "Object is not of type Module")

    def test_module_parameters(self):
        """Test if parameters are correctly collected."""
        class Model(Module):
            def __init__(self):
                super().__init__()
                self.param1 = Parameter(rand(shape=[1, 10]))
                self.param2 = Parameter(rand(shape=[1, 10]))

        model = Model()
        params = list(model.parameters())

        self.assertEqual(len(params), 2, "Not expected number of parameters")
        self.assertIsInstance(params[0], Parameter, "Object is not of type Parameter")
        self.assertIsInstance(params[1], Parameter, "Object is not of type Parameter")

    def test_forward_pass(self):
        """Test the forward method on a simple module."""
        class EWMul(Module):
            def __init__(self):
                super().__init__()
                self.parameter = Parameter(rand(shape=[1, 10]))

            def forward(self, x):
                return x * self.parameter.tensor

        model = EWMul()
        input_data = rand(shape=[1, 10])
        output = model(input_data)
        expected_output = input_data * model.parameter.tensor
        output_shape, input_shape = shape_to_list(output, input_data)
        for i in range(int(output.numel())):
            expected_value = expected_output[i]
            actual_value = output[i]
            self.assertEqual(expected_value, actual_value, "Values mismatch")
            
        self.assertIsInstance(output, LemurTensor, "Object is not of type LemurTensor")
        self.assertEqual(output_shape, input_shape, "Shape mismatch")

    def test_apply_function(self):
        """Test apply() method modifies parameters."""
        def double_params(module):
            """Function to double all parameters."""
            if isinstance(module, Module):
                for param in module.parameters():
                    param.tensor = param.tensor * twos(shape=[1, 10])

        class TestModel(Module):
            def __init__(self):
                super().__init__()
                self.param = Parameter(rand(shape=[1, 10]))

        model = TestModel()
        original_values = model.param.tensor
        expected_values = original_values * twos(shape=[int(original_values.shape[i]) for i in range(5)])
        model.apply(double_params)
        updated_values = model.param.tensor
        param_shape, expected_shape = shape_to_list(updated_values, expected_values)
        for i in range(int(updated_values.numel())):
            expected_value = expected_values[i]
            actual_value = updated_values[i]
            self.assertEqual(expected_value, actual_value, "Values mismatch")
        
        self.assertEqual(updated_values.numel(), expected_values.numel(), "Size mismatch")
        self.assertEqual(param_shape, expected_shape, "Shape mismatch after apply()")

    def test_add_module(self):
        """Test adding child modules dynamically."""
        parent = Module()
        child = Module()
        parent.add_module("child", child)
        children = list(parent.children())

        self.assertEqual(len(children), 1)
        self.assertEqual(children[0], child)

    def test_missing_forward(self):
        """Test that calling forward() on base Module raises error."""
        model = Module()
        try:
            model(rand(shape=[1, 10]))
            assert False, "Expected NotImplementedError"
        except NotImplementedError as e:
            print("NotImplementedError was raised correctly:", e)

    def test_train_eval_mode(self):
        """Test switching between training and evaluation mode."""
        model = Module()
        model.train(False)
        self.assertFalse(model._train_mode)

        model.train(True)
        self.assertTrue(model._train_mode)

    def test_repr_output(self):
        """Test __repr__ method to check module hierarchy printout."""
        class TestModel(Module):
            def __init__(self):
                super().__init__()
                self.param = Parameter(rand(shape=[1, 10]))

        model = TestModel()
        output = repr(model)

        self.assertIn("TestModel", output)
        self.assertIn("Parameter", output)

if __name__ == "__main__":
    unittest.main()