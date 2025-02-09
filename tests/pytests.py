import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import lemur

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

if __name__ == "__main__":
    unittest.main()