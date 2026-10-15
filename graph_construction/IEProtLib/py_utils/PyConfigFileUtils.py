
import numpy as np

def parse_elem_list(pStr, dtype):
    """Method to parse a list of elements in a string format.

    Args:
        pStr (string): List of float values separated by coma.
    Returns:
        (np.array): Array of floats.
    """
    elemsStrList = pStr.split(',')
    elems = []
    for curElem in elemsStrList:
        elems.append(dtype(curElem))
    return np.asarray(elems)