from frontend.ptensor import tensor, empty, full, arange, linspace, zeros, ones, rand
from frontend.reprutils import set_verbose_print, set_sci_print, print_lemur_version

__version__ = "0.0.1"

def main():
    print_lemur_version(__version__)
    print()
    print()
    x = arange(32, requires_grad=True).view(tensor([1,1,2,4,4]))
    y = x.sum()
    y.backward() 
    print(x)
    print()
    print(y.graph)
    print()


if __name__ == "__main__":
    main()
