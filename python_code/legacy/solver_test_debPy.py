import numpy as np
import numpy as np
import datetime
from stochastic_search import StochasticSearch
import shutil
import matplotlib.pyplot as plt

solver = StochasticSearch()
sol = solver.ode_int_solution()
solver.save_parameters(file_name_prefix='parameters')