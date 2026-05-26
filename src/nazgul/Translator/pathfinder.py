# allowed simulations suites
from nazgul.configurations import SimSuiteNames

# -> change the name of this to nominator?
def translate_galname(kw_gal,simsuite,**kwargs):
    
    if simsuite=="EAGLE":
        #Note: this is unique only within the snap
        from nazgul.Translator.EAGLE.pathfinder import get_galname
        Gn  = kw_gal["Gn"]
        SGn = kw_gal["SGn"]
        galname = get_galname(Gn=Gn,SGn=SGn)
    elif simsuite=="COLIBRE":
        from nazgul.Translator.COLIBRE.pathfinder import get_galname
        soap_index = kw_gal["soap_index"]
        galname = get_galname(soap_index)
    elif simsuite=="ANL_TEST":
        from nazgul.Translator.ANL_TEST.pathfinder import get_galname
        galname = get_galname(kw_gal["theta_E"],kw_gal["n_smpl"])
    else:
        raise RuntimeError(f"The simsuite {simsuite} is not yet implemented, allowed only {SimSuiteNames}") 
    return galname