from frontend.ptensor import tensor, empty, full, arange, linspace, zeros, ones


__version__ = "0.0.1"

def main():
    print(f"LightLemur Version: {__version__}")

    x = arange(32, requires_grad=True).view(tensor([1,1,2,4,4]))
    y = x.sum(tensor([0,0,0,0,0]))
    y.backward() 
    print(x)
    print(y.graph())


if __name__ == "__main__":
    main()
