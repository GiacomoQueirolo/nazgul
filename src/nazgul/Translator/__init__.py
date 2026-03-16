
#allowed simusuites:
SimSuiteNames = ["EAGLE","COLIBRE"]
# allowed simulations
from nazgul.Translator.EAGLE import sim as sim_EAGLE

from nazgul.Translator.COLIBRE import sim as sim_COLIBRE
from nazgul.Translator.COLIBRE import subsim as subsim_COLIBRE

sims    = {"EAGLE":  sim_EAGLE,
           "COLIBRE":sim_COLIBRE}
subsims = {"EAGLE":{},
           "COLIBRE":subsim_COLIBRE}

# Define the standard simulation (for convenience)
std_simsuite  = SimSuiteNames[0]
std_sim       = sims[std_simsuite][0]

# use the following simulation only as test case
test_sim      = sims[std_simsuite][1]
# used for tutorial -linked ONLY snap 20 of test_sim
tutorial_sim  = sims[std_simsuite][2]

# Default values 
min_z        = 0.02
max_z        = 2
min_mass     = 1e12 # Sol Mass