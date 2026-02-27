from Gen_PM_PLL_AMR import LoadLens
lens = LoadLens("/pbs/home/g/gqueirolo/EAGLE/sim_lens//RefL0025N0752/snap20_G8.0/test_sim_lens_AMR/G8SGn0_Npix200_PartAS.pkl")
lens.run(read_prev=False)
print("ThetaE:",lens.thetaE)
