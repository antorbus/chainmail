# LightLemur

**LightLemur** is a lightweight tensor differentiation library with minimal dependencies.

---

## **Installation**

Clone the repository and build the library:

```bash
git clone https://github.com/antorbus/lightlemur.git
cd lightlemur
make
```

This will generate the shared library `liblightlemur.dylib` (or `.so` on Linux) in the project directory.

---

## **Usage**

### **Linking the Library**

Compile your program with the LightLemur library:

```bash
clang -o my_program my_program.c -L/path/to/lightlemur -llightlemur
```

### **Example: Basic Tensor Operations**

```c
int main(){

    // t0 * t1 + t2 

    size_t shape[5] = {1,1,1,1,1};
    
    tensor *t0 = empty_tensor(shape, true);
    t0->k->array[0] = 2;

    tensor *t1 = empty_tensor(shape, true);
    t1->k->array[0] = 12.0;

    tensor *t2 = empty_tensor(shape, true);
    t2->k->array[0] = -15.0;

    tensor *t3 = binary_forward(OP_MUL, t0, t1, false);
    tensor *t4 = binary_forward(OP_ADD, t3, t2, false);
    
    backwards(t4); 

    print_tensor(t0);
    print_tensor(t1);
    print_tensor(t2);
    print_tensor(t4);

    free_tensor(t0);
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(t3);
    free_tensor(t4);

    return 0;
}
```

Compile and run:

```bash
clang -o example example.c -L/path/to/lightlemur -llightlemur
./example
```

---

## **Contributing**

Contributions are welcome! If you'd like to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a clear description of your changes.

---

## **License**

LightLemur is licensed under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use, modify, and distribute it as needed.

---

## **Contact**

For questions, suggestions, or feedback, please open an issue on the [GitHub repository](https://github.com/antorbus/lightlemur) or contact the maintainer directly.

