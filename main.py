import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from matplotlib.patches import Patch
from IPython import embed

from preprocess import get_data

stim_path = "GP_rat/stim"
res_path = "GP_rat/results"
data_path = "GP_rat/data"
rat = 1
pos = 1

data = get_data(f"{data_path}/r{rat}_p{pos}_test.nex", rat, pos)

embed()