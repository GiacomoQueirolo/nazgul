# to run to test Gen_PM_PLL_alpha
from remade_gal import get_rnd_NG
from python_tools.get_res import LoadClass
from Gen_PM_PLL_AMR import LensPart,plot_all,LoadLens
from Gen_PM_PLL_AMR import kwlens_part_AS,z_source_max,pixel_num
from pyinstrument import Profiler
from Gen_PM_PLL_AMR_test_caustics import LensPart as LensPartCaustic
if __name__ == "__main__":
    Gal = get_rnd_NG()
    #print("Loading specific gal for debugging")
    #Gal = LoadClass("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_23//Gn3SGn0.pkl")
    #Gal = LoadClass("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_16//Gn2SGn0.pkl") #  16.20.0 is a great one for Einstein Ring
    #Gal = LoadClass("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_18/Gn23SGn0.pkl")
    #Gal = LoadClass("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_23//Gn3SGn0.pkl") -> fails without reason

    #Gal = LoadClass("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_18//Gn5SGn0.pkl")

    # for testing we reload the lensing data 
    profiler = Profiler()
    profiler.start()
    mod_LP = LensPart(Galaxy=Gal,kwlens_part=kwlens_part_AS,
                       z_source_max=z_source_max, 
                       pixel_num=pixel_num,reload=False,savedir_sim="test_sim_lens_AMR")
    """
    """
    #mod_LP = LoadLens("sim_lens/RefL0025N0752/snap16_G8.0//test_sim_lens_AMR/G8SGn0_Npix200_PartAS.pkl")
    mod_LP.run()
    profiler.stop()
    print(profiler.output_text(color=True,show_all=False))
    plot_all(mod_LP,skip_caustic=True)

    profiler.start()
    mod_LP_caustic = LensPartCaustic(Galaxy=Gal,kwlens_part=kwlens_part_AS,
                       z_source_max=z_source_max, 
                       pixel_num=pixel_num,reload=False,savedir_sim="test_sim_lens_AMR_caustic")
    mod_LP_caustic.run()
    profiler.stop()
    print(profiler.output_text(color=True,show_all=False))


