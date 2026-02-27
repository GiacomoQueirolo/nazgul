# to run to test Gen_PM_PLL_alpha
from remade_gal import get_rnd_NG
from python_tools.get_res import LoadClass
from Gen_PM_PLL_alpha import LensPart,plot_all
from Gen_PM_PLL_alpha import kwlens_part_AS,cutoff_radius,z_source_max,pixel_num
if __name__ == "__main__":
    Gal = get_rnd_NG()
    #print("Loading specific gal for debugging")
    #Gal = LoadClass("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_23//Gn3SGn0.pkl")
    #Gal = LoadClass("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_16//Gn2SGn0.pkl") #  16.20.0 is a great one for Einstein Ring
    #Gal = LoadClass("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_18/Gn23SGn0.pkl")
    #Gal = LoadClass("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_23//Gn3SGn0.pkl") -> fails without reason

    #Gal = LoadClass("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_18//Gn5SGn0.pkl")

    # for testing we reload the lensing data 
    mod_LP = LensPart(Galaxy=Gal,kwlens_part=kwlens_part_AS,
                       cutoff_radius=cutoff_radius,z_source_max=z_source_max, 
                       pixel_num=pixel_num,reload=False,savedir_sim="test_sim_lens")
    mod_LP.run()
    plot_all(mod_LP,skip_show=True,skip_caustic=True)


