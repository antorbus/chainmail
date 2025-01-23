from frontend.ptensor import *

def MSELoss(x, y):
    return ((x-y)**2).mean()