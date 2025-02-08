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



if __name__ == "__main__":
    unittest.main()