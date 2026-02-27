# for now this programs will plot the density contours of a given galaxy,
# inspired heavily by create_isocont
from Gen_PM_PLL import LensPart,kwlens_part_AS,cutoff_radius,z_source_max,pixel_num
from python_tools.get_res import LoadClass
from remade_gal import get_rnd_NG


def get_lens(path_gal_name=None,reload=True ,pixel_num=pixel_num):
    # to restructure in light of LoadLens()
    if path_gal_name is None:
        Gal = get_rnd_NG()
    else:
        Gal = LoadClass(path_gal_name)
    lensGal = LensPart(Galaxy=Gal,kwlens_part=kwlens_part_AS,
                       cutoff_radius=cutoff_radius,z_source_max=z_source_max, 
                       pixel_num=pixel_num,reload=reload)
    lensGal.run()
    return lensGal
import astropy.units as u 
from create_isocont import plot_dens_map_hist,plot_densWzoom

# temp : 
from Gen_PM_PLL import LensPart,PMLens,thetaE_AS_prefact,thetaE_AS,get_lens_model_AS
if __name__ == "__main__":
    
    #Lens = get_lens()
    #Gal = LoadClass(
    #Lens = get_lens(path_gal_name="/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_23/Gn3SGn0.pkl")
    Lens = get_lens(path_gal_name="/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_18/Gn5SGn0.pkl")
    
    name_proj = Lens.savedir+"/proj_mass.pdf"
    #plot_dens_map_hist(Lens.Gal,Lens.proj_index,Lens.pixel_num,cutoff_radius=Lens.cutoff_radius,namefig=name_proj)
    """
    cutoff_radius1 = 25
    name_proj_zoom = Lens.savedir+f"/proj_mass_{cutoff_radius1}kpc2.pdf"
    title_zoom = "Zoom of Mass Density of Gal: "+str(Lens.Gal.Name)
    plot_dens_map_hist(Lens.Gal,Lens.proj_index,Lens.pixel_num,namefig=name_proj_zoom,cutoff_radius=cutoff_radius1*u.kpc,title=title_zoom)
    """
    cutoff_radius2 = 5
    name_proj_zoom2 = Lens.savedir+f"/proj_mass_zoom_r{cutoff_radius2}kpc2.pdf"
    title_zoom2 = "Higher Zoom of Mass Density of Gal: "+str(Lens.Gal.Name)
    #plot_dens_map_hist(Lens.Gal,Lens.proj_index,Lens.pixel_num,namefig=name_proj_zoom2,cutoff_radius=cutoff_radius2*u.kpc,title=title_zoom2)
    plot_densWzoom(Lens.Gal,Lens.proj_index,Lens.pixel_num,namefig=name_proj_zoom2,cutoff_radius=cutoff_radius*u.kpc,cutoff_radius_zoom=cutoff_radius2*u.kpc)

    