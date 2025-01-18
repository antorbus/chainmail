import ctypes

def _format_kernel_tensor(k_ptr):
    if not k_ptr:
        return "  [NULL kernel_tensor]\n"

    k = k_ptr.contents
    lines = []
    lines.append(f"  kernel_tensor: 0x{ctypes.addressof(k):x}\n")
    lines.append(f"    length     = {k.length}\n")

    shape = [k.shape[i] for i in range(5)]
    lines.append("    shape      = [" + ", ".join(str(s) for s in shape) + "]\n")

    stride = [k.stride[i] for i in range(5)]
    lines.append("    stride     = [" + ", ".join(str(s) for s in stride) + "]\n")

    lines.append(f"    computed   = {str(k.computed).lower()}\n")

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
    data_str.append("])")

    data_str = "".join(data_str)

    lines.append("    data:\n")
    lines.append(f"      {data_str}\n")

    return "".join(lines)


def _format_expression(e_ptr):
    if not e_ptr:
        return "  [NULL expression]\n"
    e = e_ptr.contents
    lines = []
    lines.append("  expression:\n")
    lines.append(f"    backward_func = {e.backward_func}\n")

    if e.t0:
        lines.append(f"    t0:      0x{ctypes.addressof(e.t0.contents):x}\n")
    else:
        lines.append("    t0:      [NULL tensor]\n")

    if e.t1:
        lines.append(f"    t1:      0x{ctypes.addressof(e.t1.contents):x}\n")
    else:
        lines.append("    t1:      [NULL tensor]\n")

    return "".join(lines)


def _tensor_repr(t_ptr):
    """
    Builds the multi-line string replicating print_tensor().
    """
    if not t_ptr:
        return "[NULL tensor]"

    t = t_ptr.contents
    lines = []
    lines.append(f"tensor: 0x{ctypes.addressof(t):x}\n")
    lines.append(f"  requires_grad = {str(t.requires_grad).lower()}\n")

    lines.append("  comes_from:\n")
    lines.append(_format_expression(t.comes_from))

    lines.append("  k:\n")
    lines.append(_format_kernel_tensor(t.k))

    lines.append("  grad:\n")
    lines.append(_format_kernel_tensor(t.grad))

    return "".join(lines)

