def get_galname(kw_gal,**kwargs):
    theta_E = kw_gal["theta_E"]
    n_smpl  = kw_gal["n_smpl"] 
    prof    = kw_gal["profile"]
    if prof == "SIS":
        return f"TEST_GAL_SIS_tE{theta_E}_nS{n_smpl}"
    elif prof == "SIE":
        return f"TEST_GAL_SIE_tE{theta_E}_nS{n_smpl}"