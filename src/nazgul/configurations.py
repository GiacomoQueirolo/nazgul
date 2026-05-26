from pathlib import Path
import astropy.units as u


## Set the path to the nazgul directory - may be needed for hpc systems
nazgul_path = Path(__file__).parent #Path('/cosma8/data/dp004/dc-lang2/ColibreLens/EAGLE_Lensing/src/nazgul')
#Path('/Users/samlange/Code/ColibreLens/EAGLE_Lensing/src/nazgul')

## Set which lens populations to use: LSST, Euclid or DES
forecast_telescope = 'Euclid'

## Set standard simsuite
SimSuiteNames = ["EAGLE", "COLIBRE"] #allowed simusuites - do not edit
std_simsuite  = SimSuiteNames[0]


## Set Default values
#The below came from Translator/init
min_z        = 0.02
max_z        = 2
min_mass     = 1e12 # Sol Mass

#The below came from cracks_of_doom
min_thetaE    = .3*u.arcsec # minimum theta_E allowed for a lens
pixel_num     = 200 # pix for image
verbose       = True
z_source_max  = 4
scale_tE      = 10 # How many times larger than the einstein radius should the image be

