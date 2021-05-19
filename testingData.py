from pynwb import NWBHDF5IO
import numpy as np
import pandas as pd
import sys
import inspect
import os

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
stack = []


def save_as_np(arr):
    file = input("Under what name should I save?")
    np.save(os.path.dirname(os.path.abspath(__file__)) + file + ".npy", arr, allow_pickle=True)


def print_file(file, stack):
    """
    Prints the last part of the file using a stack or list of instructions.
    :param file: File we want to print the attributes of
    :param stack: The stack of attributes / fields that
    :return: Nothing.
    """
    attr = file
    for i in stack:
        attr_test = getattr(attr, i, "0")
        if attr_test == "0":
            try:
                if i == "Save":
                    save_as_np(attr)
                    stack.pop()
                elif i == ":":
                    attr = attr[:]
                else:
                    attr = attr[i]
            except TypeError or KeyError:
                print(i + " is not an attribute or field!")
                stack.pop()
                return
            except KeyError:
                print(i + " is not an attribute or field!")
                stack.pop()
                return
        else:
            attr = attr_test
    print(attr)


def navigate(file):
    """
    Function that helps to navigate through the file
    :param file: The file to navigate through
    :return: Nothing.
    """
    print(file)
    print_sep()
    response = input("Where to now?")
    stack.append(response)
    while response != "0":
        if response == "-1":
            print("Going backwards")
            stack.pop()
            stack.pop()
        print_file(file, stack)
        print_sep()
        response = input("Where to now?")
        stack.append(response)


def get_size(obj, seen=None):
    """Recursively finds size of objects in bytes"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += get_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((get_size(v, seen) for v in obj.values()))
        size += sum((get_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum((get_size(i, seen) for i in obj))

    if hasattr(obj, '__slots__'):  # can have __slots__ with __dict__
        size += sum(get_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))

    return size


def print_sep():
    print("==================================================================================")


def print_list(dat):
    for i in dat:
        print(str(i) + ", ", end="")
    print()


def get_aquisition(nwbfile_in):
    electrode = nwbfile_in.electrodes
    print("-------------------------------------------------")
    print(electrode)
    print("-----------------------------------------------------")
    x_dat = electrode.get('x')[:-2]
    print(np.shape(x_dat))
    for i in x_dat:
        print(str(i) + ", ", end="")
    print()


def fuckyou(arr):
    for x in arr:
        if not x:
            print("FUCK YOU")


io = NWBHDF5IO(os.path.dirname(os.path.abspath(__file__))
               + "/sub-YutaMouse37_ses-YutaMouse37-150609_behavior+ecephys.nwb", "r")
nwbfile_in = io.read()
navigate(nwbfile_in)
