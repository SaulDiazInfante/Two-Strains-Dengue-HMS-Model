import numpy as np
import datetime
from stochastic_search import StochasticSearch
import shutil
sim = StochasticSearch()
sim.__init__()
sim.number_of_samples = 30
sim.bound_error_FD = 0.4
sim.bound_error_FHD = 0.4
print '%-8s%-12s%-12s%-12s%-12s%-12s' \
      % ('i', 'R_01', 'R_02', 'R_zero', 'error_DF', 'error_DHF')
for i in np.arange(sim.number_of_samples):
    sim.ode_int_solution()
    sim.fitting_error()
    error_DF = sim.fitting_error_DF
    error_DHF = sim.fitting_error_DHF
    r01_per_week, r02_per_week, r0_per_week = sim.compute_r_zero()
    sim.update_conditions_search()
    #if sim.stop_condition:
    sim.solution_plot()
    sim.fitting_plot()
    sim.save_parameters('./OutputParameters/deb/')
    str_time = str(datetime.datetime.now())
    file_name = './plots/deb/fitting_DF_DHF_' + str_time + '.png'
    shutil.copy2('./plots/fitting_DF_DHF.png', file_name)
    file_name = './plots/deb/populations_grid_' + str_time + '.png'
    shutil.copy2('./plots/populations_grid.png', file_name)
    #print '=)'
    print '%-12d%-12f%-12f%-12f%-12f%-12f' \
          % (i, r01_per_week, r02_per_week, r0_per_week, error_DF, error_DHF)
    sim.parameters_sampling(flag_deterministic=False)