import nazgul.configurations as conf

from nazgul.Translator.EAGLE   import sim as sim_EAGLE
from nazgul.Translator.COLIBRE import sim as sim_COLIBRE
from nazgul.Translator.COLIBRE import subsim as subsim_COLIBRE


sims    = {"EAGLE":  sim_EAGLE,
           "COLIBRE":sim_COLIBRE}
subsims = {"EAGLE":{},
           "COLIBRE":subsim_COLIBRE}

# Define the standard simulation (for convenience)
std_simsuite  = conf.std_simsuite
std_sim       = sims[std_simsuite][0]

# use the following simulation only as test case
test_sim      = sims[std_simsuite][1]
# used for tutorial -linked ONLY snap 20 of test_sim
tutorial_sim  = sims[std_simsuite][2]

# Default values - kept here fore backwards compatability but now chosen in configurations
min_z        = conf.min_z
max_z        = conf.max_z
min_mass     = conf.min_mass