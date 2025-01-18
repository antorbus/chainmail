import ctypes

LEMUR_VERBOSE = True

def _format_kernel_tensor(k_ptr, verbose=LEMUR_VERBOSE):
    if not k_ptr:
        return "[NULL kernel_tensor]\n"

    k = k_ptr.contents
    lines = []

    stride = [k.stride[i] for i in range(5)]
    shape = [k.shape[i] for i in range(5)]
    
    if verbose:
        lines.append(f"kernel_tensor @ 0x{ctypes.addressof(k):x} with length = {k.length} \n")
        
        lines.append("stride = [" + ", ".join(str(s) for s in stride) + "]\n")
        lines.append(f"computed = {str(k.computed).lower()}\n")

    

    data_str = []
    data_str.append("tensor([")
    arr = k.array  
    for d0 in range(shape[0]):
        data_str.append("[")
        for d1 in range(shape[1]):
            data_str.append("[")
            for d2 in range(shape[2]):
                data_str.append("[")
                for d3 in range(shape[3]):
                    data_str.append("[")
                    for d4 in range(shape[4]):
                        idx = (d0 * stride[0] +
                               d1 * stride[1] +
                               d2 * stride[2] +
                               d3 * stride[3] +
                               d4 * stride[4])
                        val = float(arr[idx].value)  # a lemur_float
                        data_str.append(f"{val:.4f}")
                        if d4 < shape[4] - 1:
                            data_str.append(", ")
                    data_str.append("]")
                    if d3 < shape[3] - 1:
                        data_str.append(",")
                data_str.append("]")
                if d2 < shape[2] - 1:
                    data_str.append(",")
            data_str.append("]")
            if d1 < shape[1] - 1:
                data_str.append(",")
        data_str.append("]")
        if d0 < shape[0] - 1:
            data_str.append(",")
    data_str.append("])\n")
    data_str = "".join(data_str)
    lines.append(data_str)

    lines.append("shape = [" + ", ".join(str(s) for s in shape) + "]\n")

    return "".join(lines)


def _format_expression(e_ptr, verbose=LEMUR_VERBOSE):
    if not e_ptr:
        return "[NULL expression]\n"
    e = e_ptr.contents
    lines = []
    lines.append(f"backward_func = {e.backward_func}")

    if e.t0:
        lines.append(f"    t0 @ 0x{ctypes.addressof(e.t0.contents):x}")
    else:
        lines.append("    t0 @ [NULL tensor]")

    if e.t1:
        lines.append(f"    t1 @ 0x{ctypes.addressof(e.t1.contents):x}")
    else:
        lines.append("    t1 @ [NULL tensor]")
    lines.append("\n")
    return "".join(lines)


def _tensor_repr(t_ptr, verbose=LEMUR_VERBOSE):
    if not t_ptr:
        return "[NULL tensor]"

    t = t_ptr.contents
    lines = []
    if verbose:
        lines.append(f"tensor @ 0x{ctypes.addressof(t):x}\n")
        lines.append("comes_from: ")
        lines.append(_format_expression(t.comes_from))
   
    lines.append("k:\n")
    lines.append(_format_kernel_tensor(t.k))

    if t.requires_grad:
        lines.append("grad:\n")
        lines.append(_format_kernel_tensor(t.grad))

    return "".join(lines)

