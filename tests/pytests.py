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

    def test_add(self):
        self.skipTest("Not implemented")


if __name__ == "__main__":
    unittest.main()