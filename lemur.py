from frontend.bindings import LemurTensor, tensor

__version__ = "0.0.1"

def main():
    print(f"LightLemur Version: {__version__}")

    x = tensor([1.0], requires_grad=True)
    y = tensor([4.0], requires_grad=True)

    z = x + y  
    w = z * x  
    w = w.relu() 

    w.backward() 
    print("Result from main:", w)

if __name__ == "__main__":
    main()
