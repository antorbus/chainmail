from frontend.bindings import LemurTensor, tensor

__version__ = "0.0.1"

def main():
    print(f"LightLemur Version: {__version__}")

    x = tensor([i for i in range(1,33)], (2,2,2,2,2),  requires_grad=True)
    s = tensor([0,0,0,0,0]) 
    y = x.sum(s)
    y.backward() 
    print(x)
    # y = tensor([4.0], requires_grad=True)

    # z = x + y  
    # w = z * x  
    # w = w.relu() 

    # w.backward() 
    # print(w)
    # print()
    # print(x)



if __name__ == "__main__":
    main()
