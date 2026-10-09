import sys
import numpy as np
from typing import Tuple
import networkx as nx


def extract_raw_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.uint8)
    return data

data = extract_raw_data("data/tiny.csv")

n = 10
i = 1
x = list(range(n))
print(x[0:0] +  x[1:9])