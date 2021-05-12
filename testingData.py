from pynwb import NWBHDF5IO
import numpy as np
import pandas as pd
import sys
import inspect

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
stack = []


def print_file(file):
    attr = file
    for i in stack:
        attr_test = getattr(attr, i, "0")
        if attr_test == "0":
            try:
                if i == ":":
                    attr = attr[:]
                else:
                    attr = attr[i]
            except TypeError or KeyError:
                print(i + " is not an attribute or field!")
                stack.pop()
                return
        else:
            attr = attr_test
    print(attr)


def navigate(file):
    print(file)
    print_sep()
    response = input("Where to now?")
    stack.append(response)
    while response != "0":
        if response == "-1":
            print("Going backwards")
            stack.pop()
            stack.pop()
        print_file(file)
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


io = NWBHDF5IO("/mnt/d/Users/odilo/Downloads/sub-YutaMouse37_ses-YutaMouse37-150609_behavior+ecephys.nwb", "r")
nwbfile_in = io.read()
navigate(nwbfile_in)
#print(nwbfile_in)
# test_timeseries_in = nwbfile_in.electrode_groups['4x8_Neuronexus_Z50um_X200um: 32']
# get_aquisition(nwbfile_in)
#print_sep()
#print(nwbfile_in.acquisition["ch_SsolL"].data[:])
# print(nwbfile_in.acquisition["lick_trace"].time_series["lick_trace"])
# print(nwbfile_in.acquisition)
#print_sep()
# print(get_size(nwbfile_in.acquisition["principal_whisker_b2"]))
# print(nwbfile_in.acquisition["principal_whisker_C2"].time_series["touch_offset"].data[:])
# print_sep()
# print(nwbfile_in.acquisition["CurrentClampSeries"].data[:])
# print(get_size(nwbfile_in.acquisition["CurrentClampSeries"].data[:]))
# print(nwbfile_in.acquisition["principal_whisker_b2"].time_series["pole_available"])
# print(str(nwbfile_in.electrodes[0:]))
# print_sep()
# print_list(test_timeseries_in.device)
