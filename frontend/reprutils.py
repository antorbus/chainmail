import ctypes
from frontend.bindings import lib

LEMUR_VERBOSE = False
LEMUR_SCI_PRINT = False

def set_verbose_print(value):
    global LEMUR_VERBOSE
    LEMUR_VERBOSE = value

def set_sci_print(value):
    global LEMUR_SCI_PRINT
    LEMUR_SCI_PRINT = value

def get_op_name(i):
    return lib.get_op_name(i)

def _format_kernel_tensor(k_ptr, postfix = ""):
    if not k_ptr:
        return "[NULL kernel_tensor]\n"

    k = k_ptr.contents
    lines = []
    vlines = []
    stride = [k.stride[i] for i in range(5)]
    shape = [k.shape[i] for i in range(5)]
    
    if LEMUR_VERBOSE:
        lines.append(f"kernel_tensor @ 0x{ctypes.addressof(k):x} with length = {k.length} \n")
        vlines.append(", stride=[" + ", ".join(str(s) for s in stride) + "]")
        vlines.append(f", computed={str(k.computed).lower()}")
        vlines.append(f", shallow={str(k.shallow).lower()}")

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
                        if LEMUR_SCI_PRINT:
                            data_str.append(f"{val:6.2e}")
                        else:
                            data_str.append(f"{val:6.2f}")
                        if d4 < shape[4] - 1:
                            data_str.append(", ")
                    data_str.append("]")
                    if d3 < shape[3] - 1:
                        data_str.append(",\n\t   ")
                data_str.append("]")
                if d2 < shape[2] - 1:
                    data_str.append(",\n\n\t  ")
            data_str.append("]")
            if d1 < shape[1] - 1:
                data_str.append(",\n\n\n\t ")
        data_str.append("]")
        if d0 < shape[0] - 1:
            data_str.append(",\n\n\n\n\t")

    data_str.append("], shape=[" + ", ".join(str(s) for s in shape) + "]"+ "".join(vlines) + postfix + ")")
    data_str = "".join(data_str)
    lines.append(data_str)

    return "".join(lines)


def _format_expression(e_ptr):
    if not e_ptr:
        return "[NULL expression]\n"
    e = e_ptr.contents
    lines = []
    raw_op_name = get_op_name(e.backward_func)
    op_name = raw_op_name.decode("utf-8") if raw_op_name else "unknown_op"
    lines.append(f"backward_func=[{op_name}]")

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


def _tensor_repr(t_ptr):
    if not t_ptr:
        return "[NULL tensor]"

    t = t_ptr.contents
    lines = []
    if LEMUR_VERBOSE:
        lines.append(f"tensor @ 0x{ctypes.addressof(t):x}\n")
        lines.append("comes_from: ")
        lines.append(_format_expression(t.comes_from))
    
    if t.requires_grad and t.comes_from and not LEMUR_VERBOSE:
            raw_op_name = get_op_name(t.comes_from.contents.backward_func)
            op_name = raw_op_name.decode("utf-8") if raw_op_name else "unknown_op"
            postfix = f", comes_from=[{op_name}]"
    else:
        postfix = ""
    lines.append(_format_kernel_tensor(t.k, postfix=postfix))

    if t.requires_grad and LEMUR_VERBOSE:
        lines.append("\ngrad:\n"+_format_kernel_tensor(t.grad))

    return "".join(lines)

def _short_label(t):
    ptr = getattr(t, "_ptr", None)
    addr_str = f"0x{id(t):x}"
    shape_str = ""

    if ptr:
        c_tensor = ptr.contents
        addr_str = f"0x{ctypes.addressof(c_tensor):x}"  
        if c_tensor.k:
            k_obj = c_tensor.k.contents
            dims = [k_obj.shape[i] for i in range(5)]
            non1 = [str(d) for d in dims]
            if non1:
                shape_str = f"(shape=[{', '.join(non1)}])"
    return f"LemurTensor @ {addr_str} {shape_str}"


def _build_ascii_lines(t, prefix="", is_last=True, visited=None):
    if visited is None:
        visited = set()

    lines = []
    if t is None:
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + "[NULL LemurTensor]")
        return lines

    if id(t) in visited:
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + f"[repeated] {_short_label(t)}")
        return lines
    visited.add(id(t))

    connector = "└── " if is_last else "├── "
    lines.append(prefix + connector + _short_label(t))

    child_prefix = prefix + ("    " if is_last else "│   ")

   
    op_name = None
    ptr = getattr(t, "_ptr", None)
    if ptr:
        c_tensor = ptr.contents
        expr_ptr = c_tensor.comes_from
        if expr_ptr:
            bf = expr_ptr.contents.backward_func
            raw_op_name = get_op_name(bf)  # returns a c_char_p
            op_name = raw_op_name.decode("utf-8") if raw_op_name else "unknown_op"

   
    parents = getattr(t, "_parents", [])
    if op_name:
        lines.append(child_prefix + f"└── [{op_name}]")
        op_prefix = child_prefix + "    "
        for i, p in enumerate(parents):
            parent_is_last = (i == len(parents) - 1)
            lines.extend(_build_ascii_lines(p, 
                                            prefix=op_prefix, 
                                            is_last=parent_is_last,
                                            visited=visited))
    else:
        for i, p in enumerate(parents):
            parent_is_last = (i == len(parents) - 1)
            lines.extend(_build_ascii_lines(p, 
                                            prefix=child_prefix, 
                                            is_last=parent_is_last,
                                            visited=visited))

    return lines


def plot_tensor_graph_parents(t):
    lines = _build_ascii_lines(t)
    return "self\n"+"\n".join(lines)
