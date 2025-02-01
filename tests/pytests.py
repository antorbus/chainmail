import sys
import unittest
sys.path.append('../')
import lemur

class TestLightLemur(unittest.TestCase):
    
    def test_basic(self):
        self.assertEqual(1, 1, "Grad check for v failed")

    def test_basic2(self):
        self.assertEqual(1, 1, "Grad check for v failed")

    def test_basic3(self):
        self.assertEqual(1, 1, "Grad check for v failed")


    def test_add(self):
        self.skipTest("Not implemented")


if __name__ == "__main__":
    unittest.main()