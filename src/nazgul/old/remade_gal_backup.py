# copied from tmp_print_xyz, we want to 
# obtain a new gal class that works

# -> this should only deal with Galaxy -> projection etc should be done in another file
import os
import glob
import pickle
import numpy as np
import h5py
import astropy.units as u
import matplotlib.pyplot as plt
from decimal import Decimal

from python_tools.tools import mkdir
import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM
from get_gal_indexes import get_gals
from fnct import _count_part,_mass_part,get_gal_path
from fnct import part_data_path,std_sim,get_z_snap,prepend_str,get_snap,read_snap_header_simple

# combination btw get_rnd_gal and _get_rnd_gal
z_source_max = 4
verbose      = True
min_z        = 0.2
# max_z is implicitely =3.53
min_mass     = 5e13 # Sol Mass

def get_rnd_gal_indexes(sim=std_sim,min_mass = str(min_mass),min_z=str(min_z),
                        max_z="2",pkl_name="massive_gals.pkl",check_prev=True,save_pkl=True):
    data  = get_gals(sim=sim,min_mass=min_mass,max_z=max_z,min_z=min_z,
                     pkl_name=pkl_name,check_prev=check_prev,plot=False,save_pkl=save_pkl)
    index = np.arange(len(data["z"]))
    rnd_i = np.random.choice(index)
    kw = {}
    for k in data.keys():
        if k=="query" or k=="sim":
            kw[k] = data[k]
        else:
            kw[k] = data[k][rnd_i]
    return kw

def basic_get_radius(RAs,DECs):
    # define a "radius" (more like 1/2 of the edge-lenght) of the grid used to sample the density
    
    # We want to have the same pixelscale in the 2Dim
    # the obvioious way would give RA and DEC that might not be on the same range
    # but we create a grid with the same number of points for both
    # we rather would go as such: redefine the ranges such that the number of pixels 
    # and the ranges are the same (but there might be some empty, ie 0 density, pixels
    # for either of the two dimensions)
    ramin  = RAs.min()
    ramax  = RAs.max()
    #rangeRa = ramax - ramin
    decmin = DECs.min()
    decmax = DECs.max()
    #rangeDec = decmax-decmin
    # we have the advantage that the center is set to 0 ->
    radius =  max([0-ramin,ramax-0,0-decmin,decmax-0])
    # verify:
    assert(ramin>=-radius)
    assert(decmin>=-radius)
    assert(ramax<=radius)
    assert(decmax<=radius)
    return radius

def get_radius(RAs,DECs,sigmas=6):
    # cut-out outlier particles 
    rad_max = basic_get_radius(RAs,DECs)
    # we take 6 <sigmas> of 
    rad_min = sigmas*(np.std(RAs)+np.std(DECs))/2
    return np.min([rad_max,rad_min])

def get_z_source(cosmo,z_lens,dens_Ms_kpc2,z_source_max=z_source_max,verbose=verbose):
    # the lens has to be supercritical
    # dens>Sigma_crit = (c^2/4PiG D_d(z_lens) ) D_s(z_source)/D_ds(z_lens,z_source)
    # -> D_s(z_source)/D_ds(z_lens,z_source) < 4PiG D_d(z_lens) *dens/c^2
    # D_s(z_source)/D_ds(z_lens,z_source) is not easy to compute analytically, but we can sample it
    if z_lens>z_source_max:
        raise ValueError("The galaxy redshift is higher than the maximum allowed source redshift")
        #return 0
    try:
        dens_Ms_kpc2.value
    except:
        # dens_Ms_kpc2 is already given in Msun/kpc^2
        dens_Ms_kpc2 *= u.Msun/(u.kpc**2)
    assert dens_Ms_kpc2.unit==u.solMass/(u.kpc**2)
    """
    print("DEBUG z_lens",z_lens)
    print("DEBU cosmo",cosmo)
    print("DEBUG cosmo.angular_diameter_distance(z_lens)",cosmo.angular_diameter_distance(z_lens))
    print("DEBUG NOTE: the approx MW surf.dens. is 2*1e9Msun/kpc^2")
    print("DEBUG np.max(dens_Ms_kpc2)",np.max(dens_Ms_kpc2))
    print("DEBUG 4*np.pi*const.G",4*np.pi*const.G)
    print("DEBUG cosmo.angular_diameter_distance(z_lens)",cosmo.angular_diameter_distance(z_lens))
    print("DEBUG (const.c**2) ",(const.c**2) )
    """
    max_DsDds = np.max(dens_Ms_kpc2)*4*np.pi*const.G*cosmo.angular_diameter_distance(z_lens)/(const.c**2) 
    #print("DEBUG\n","np.max(dens_Ms_kpc2)",np.max(dens_Ms_kpc2.to("1e9Msun/kpc^2")))
    #print("DEBUG\n","max_DsDds",max_DsDds)
    max_DsDds = max_DsDds.to("") # assert(max_DsDds.unit==u.dimensionless_unscaled) -> equivalent
    max_DsDds = max_DsDds.value # dimensionless
    #print("DEBUG\n","max_DsDds",max_DsDds)
    #z_source_range = np.linspace(z_lens,z_source_max,100) # it's a very smooth funct->
    min_DsDds = cosmo.angular_diameter_distance(z_source_max)/cosmo.angular_diameter_distance_z1z2(z_lens,z_source_max) # this is the minimum
    min_DsDds = min_DsDds.to("") # dimensionless
    min_DsDds = min_DsDds.value
    
    z_source_range = np.linspace(z_lens+0.09,z_source_max,100) # it's a very smooth funct->
    DsDds = np.array([cosmo.angular_diameter_distance(z_s).to("Mpc").value/cosmo.angular_diameter_distance_z1z2(z_lens,z_s).to("Mpc").value for z_s in z_source_range])
    if not min_DsDds<max_DsDds:
        # to do: deal with this kind of output
        if verbose:
            print("Warning: the minimum z_source needed to have a lens is higher than the maximum allowed z_source")
            plt.plot(z_source_range,DsDds,ls="-",c="k",label=r"D$_{\text{s}}$/D$_{\text{ds}}$(z$_{source}$)")
            plt.xlabel(r"z$_{\text{source}}$")
            plt.axhline(max_DsDds,ls="--",c="r",label=r"max(dens)*4$\pi$*G*$D_l$/c$^2$")
            plt.legend()
            name = "tmp/DsDds.pdf"
            plt.savefig(name)
            print("Saved "+name)
        return 0
    else:
        # note that the successful test means only that there is AT LEAST 1 PIXEL that is supercritical
        minimise     = np.abs(DsDds-max_DsDds) 
        z_source_min = z_source_range[np.argmin(minimise)]
        # select a random source within the range
        z_source = np.random.uniform(z_source_min,z_source_max,1)[0]
        if verbose:
            print("Minimum z_source:",np.round(z_source_min,2))
            print("Chosen z_source:", np.round(z_source,2))
        return z_source

def get_dP(radius_kpc,pixel_num,arcXkpc=None,cosmo=None,Gal=None):
    if arcXkpc is None:
        if cosmo is None or Gal is None:
            raise RuntimeError("Give either arcXkpc or cosmo and Gal")
        arcXkpc = cosmo.arcsec_per_kpc_proper(Gal.z) # ''/kpc (ie inverse of Dd conv. from rad to arcsec)
            
    try:
        radius_kpc.value 
    except AttributeError:
        # hoping the radius is actually inserted in kpc
        radius_kpc *= u.kpc 
    return 2*radius_kpc*arcXkpc.to("arcsec/kpc")/(int(pixel_num.imag)*u.pix) #''/pix

#itype    = 1 #dm
# gas,dm, stars,bh : 0,1,4,5 
class NewGal:
    def __init__(self, Gn, SGn,M,Centre,sim=std_sim,z=None,snap=None): #,query="",CMx,CMy,CMz,M=None):
        self.sim    = sim
        z,snap      = get_z_snap(z,snap)
        self.snap   = snap
        self.z      = z
        self.M      = M
        self.Gn     = Gn
        self.SGn    = SGn
        self.Name   = f"G{Gn}SGn{SGn}" #note this is unique only within the snap
        self.centre = Centre
        self.a,self.h,self.boxsize = self.read_snap_header()
        self.cosmo   = FlatLambdaCDM(H0=self.h*100, Om0=1-self.h)
        # Define/Create Gal path
        self.set_gal_path() # works

        # only for DEBUGging you should set it false
        self._read_prev = True
        self.run(self._read_prev)
    def run(self,_read_prev=True):
        upload_successful = False
        if _read_prev:
            upload_successful = self.upload_prev()
        if not upload_successful:
            # useful to check Center of Mass
            self.xy_propr2comov = self.prop2comov("Coordinates") 
            self.m_propr2comov  = self.prop2comov("Mass") 
            self._init_path_snap()
            self._initialise_parts()
            # works:
            self._count_tot_part()
            self._mass_tot_part()
            self._verify_cnt()
            #
            """ -> the following I believe it somehow messes up my Y positioning 
            # check if gal exists and if it's the same, if so load that instead
            raise RuntimeError("For some reason when implementing the following, the position of the Y coordinates get completely messed up")
            try:
                NGL = load_whatever(self.gal_path)
                print("NGal loaded")
                if self.__eq__(NGL):
                    # this is what takes the longest
                    self.gas   = NGL.gas
                    self.dm    = NGL.dm
                    self.bh    = NGL.bh
                    self.stars = NGL.stars
                    to_store = False
                else:
                    raise RuntimeError("Galaxy file exists but is not the same")
            except:
                print("NGal not loaded, read from data")
                #self.query  = query # query from which this gal 
                self._init_path_snap()
                self._initialise_parts()
                to_store    = True
    
            if to_store:
                
            """
            self.store_gal()

    def upload_prev(self):
        if not self._read_prev:
            return False
        prev_Gal = ReadGal(self)
        if prev_Gal is False:
            return False
        else:
            for k in prev_Gal.__dict__.keys():
                self.__dict__[k] = prev_Gal.__dict__[k]
            return True
            
    
    def prop2comov(self,varType):
        """
        # copied from fnct
        aexp = self.aexp[varType]
        # physically motivated verification 
        if varType=="Coordinates":
            self.verbose_assert_almost_equal(aexp,1)
        elif varType=="Mass":
            self.verbose_assert_almost_equal(aexp,0)
        """
        # physically motivated 
        if varType=="Coordinates":
            aexp = 1
        elif varType=="Mass":
            aexp = 0
        return  1/(self.a**aexp)

    def read_snap_header(self):
        return read_snap_header_simple(z=self.z,snap=self.snap,sim=self.sim)

    def _init_path_snap(self):
        z_str          = prepend_str(str(int(self.z)),ln_str=3,fill="0")
        str_snap       = get_snap(self.z,3)
        fullsnap_str   = f"_{str_snap}_z{z_str}p???"
        self.path_snap = f"{part_data_path}/{self.sim}/snapshot{fullsnap_str}/snap{fullsnap_str}"
        return 0
        
    def _initialise_parts(self):
        if not hasattr(self,"path_snap"):
            self._init_path_snap()
        self.gas   = self.read_part(0)
        self.dm    = self.read_part(1)
        self.stars = self.read_part(4)
        self.bh    = self.read_part(5)
        return 0
    
    def read_part(self,itype):
        kw   = {}
        atts = ["GroupNumber","SubGroupNumber","Coordinates"]
        if itype!=1:
            atts.append("Mass")
            atts.append("SmoothingLength")
        else:
            """Special case for the mass of dark matter particles."""   
            fl =  glob.glob(f"{self.path_snap}.0.hdf5")
            if len(fl)!=1:
                raise RuntimeError(f"{fl} found to be not of lenght 1 from glob({self.path_snap}.0.hdf5)")
            fl = fl[0]
            with h5py.File(fl, 'r') as f:
                h = f['Header'].attrs.get('HubbleParam')
                a = f['Header'].attrs.get('Time')
                dm_mass = f['Header'].attrs.get('MassTable')[1]
                n_particles = f['Header'].attrs.get('NumPart_Total')[1]
                # Create an array of lenght n_particles each set to dm_mass
                m = np.ones(n_particles, dtype='f8') *dm_mass
                # Use the conversion factors from the mass entry in the gas particles.
                cgs  = f['PartType0/Mass'].attrs.get('CGSConversionFactor')
                aexp = f['PartType0/Mass'].attrs.get('aexp-scale-exponent')
                hexp = f['PartType0/Mass'].attrs.get('h-scale-exponent')
            # Convert to proper/physical mass 
            kw["Mass"] = np.multiply(m, cgs*(a**aexp)*(h**hexp), dtype='f8')
            
        for att in atts:
            data = []
            nfiles = 16
            for i in range(nfiles):
                fl =  glob.glob(f"{self.path_snap}.{i}.hdf5")
                if len(fl)!=1:
                    raise RuntimeError(f"{fl} found to be not of lenght 1 from glob({self.path_snap}.{i}.hdf5)")
                fl = fl[0]
                with h5py.File(fl,'r') as f:
                    tmp = f['PartType%i/%s'%(itype,att)][...]
                    data.append(tmp)
                    # Get conversion factors
                    cgs  = f['PartType%i/%s'%(itype,att)].attrs.get('CGSConversionFactor')
                    aexp = f['PartType%i/%s'%(itype,att)].attrs.get('aexp-scale-exponent')
                    hexp = f['PartType%i/%s'%(itype,att)].attrs.get('h-scale-exponent')
                    # Get expansion factor and Hubble parameter from the header
                    a = f['Header'].attrs.get('Time')
                    h = f['Header'].attrs.get('HubbleParam')
            if len(tmp.shape) > 1:
                data = np.vstack(data)
            else:
                data = np.concatenate(data)
            # Convert to physical/proper
            if data.dtype!=np.int32 and data.dtype!=np.int64:
                # note: it IS multiply by 1/h (h**hexp)
                data = np.multiply(data,cgs*(a**aexp)*(h**hexp),dtype='f8')
            kw[att] = data
        #print("self.Gn,self.SGn",self.Gn,self.SGn)
        mask   = np.logical_and(kw["GroupNumber"]==self.Gn,kw["SubGroupNumber"]==self.SGn)
        output         = {}
        output["mass"] = kw["Mass"][mask] * u.g.to(u.Msun)
        coords         = kw["Coordinates"][mask]*u.cm.to(u.Mpc)
        # Periodic wrap coordinates around centre.
        boxsize        = self.boxsize*(h**hexp)
        centre         = self.centre*(a**aexp) # given in comoving (but not 1/h)
        """
        #DEBUG
        print("itype,np.mean(coords,axis=0),centre,boxsize")
        print(itype,np.mean(coords,axis=0),centre,boxsize)
        """
        output['coords'] = np.mod(coords-centre+0.5*boxsize,boxsize)+centre-0.5*boxsize
        """
        #DEBUG
        print("np.mean(output[coords],axis=0)")
        print(np.mean(output["coords"],axis=0))
        """
        if itype!=1:
            output["smooth"] = kw["SmoothingLength"][mask]*u.cm.to(u.Mpc)
        return output
        
    def _count_tot_part(self):
        self.N_gas      = _count_part(self.gas)
        self.N_dm       = _count_part(self.dm)
        self.N_stars    = _count_part(self.stars)
        self.N_bh       = _count_part(self.bh)
        self.N_tot_part =  self.N_gas+self.N_dm+self.N_stars +self.N_bh
        return self.N_tot_part

    def _mass_tot_part(self):
        self.M_gas   = _mass_part(self.gas)
        self.M_dm    = _mass_part(self.dm)
        self.M_stars = _mass_part(self.stars)
        self.M_bh    = _mass_part(self.bh)
        
        self.M_tot   =  self.M_gas+self.M_dm+self.M_stars +self.M_bh
        self.verbose_assert_almost_equal(float(self.M_tot)/float(self.M),1,decimal=3,msg_title="Mass")
        return self.M_tot
        
    def verbose_assert_almost_equal(self,value1,value2=1,decimal=3,msg_title=None):
        # a verbose way of giving info if if fails
        try:
            np.testing.assert_almost_equal(value1,value2,decimal=decimal)
        except AssertionError as AssErr:
            if msg_title:
                print(msg_title)
            print("Error for \n"+str(self))
            #print("found in \n"+str(self.get_pkl_path()))
            raise AssertionError(AssErr)
        return 0
        
    def _verify_cnt(self):
        # copied from fnct
        # verify that the center of mass is indeed correct
        if not hasattr(self,"M_tot"):
            self._mass_tot_part()
        # conv. in comov. coordinates
        xy_dm  = self.dm["coords"]*self.xy_propr2comov
        xy_gas = self.gas["coords"]*self.xy_propr2comov
        xy_st  = self.stars["coords"]*self.xy_propr2comov
        xy_bh  = self.bh["coords"]*self.xy_propr2comov

        # m in comov coord
        m_dm  = self.m_propr2comov*np.broadcast_to(self.dm["mass"],(3,self.N_dm)).T
        m_gas = self.m_propr2comov*np.broadcast_to(self.gas["mass"],(3,self.N_gas)).T
        m_st  = self.m_propr2comov*np.broadcast_to(self.stars["mass"],(3,self.N_stars)).T
        m_bh  = self.m_propr2comov*np.broadcast_to(self.bh["mass"],(3,self.N_bh)).T
        
        cm_dm   = np.sum(xy_dm*m_dm,axis=0)
        cm_gs   = np.sum(xy_gas*m_gas,axis=0)
        cm_st   = np.sum(xy_st*m_st,axis=0)
        cm_bh   = np.sum(xy_bh*m_bh,axis=0)

        cnt_m  = (cm_dm+cm_gs+cm_st+cm_bh)/(self.M*self.m_propr2comov)
 
        center_actual  = np.array(cnt_m)
        center_desired = self.centre
        #DEBUG
        """
        print(center_actual)
        print(center_desired)
        """
        np.testing.assert_almost_equal(center_desired,center_desired,decimal=2)
        self.verbose_assert_almost_equal(center_desired,center_desired,decimal=2,msg_title="Centre")      
        return 0
    # works
    def get_pkl_path(self):
        if not getattr(self,"gal_snap_dir",False):
           self.gal_path,self.gal_snap_dir = get_gal_path(self,ret_snap_dir=True)
        if not getattr(self,"pkl_path",False):
            self.pkl_path = f"{self.gal_snap_dir}/Gn{self.Gn}SGn{self.SGn}.pkl"
        return self.pkl_path
    def set_gal_path(self):
        self.gal_path,self.gal_snap_dir = get_gal_path(self,ret_snap_dir=True)
        mkdir(self.gal_snap_dir)
        self.get_pkl_path()
        return 0

    #def image(self):
    # maybe to implement a way to plot it    

    def store_gal(self):
        if not hasattr(self,"pkl_path"):
            self.set_gal_path()
        # store this galaxy
        with open(self.pkl_path,"wb") as f:
            pickle.dump(self,f)
        print("Saved "+self.pkl_path)
        
    def __str__(self):
        str_gal = f"Gal {self.Gn}.{self.SGn}"
        str_gal += f", at z={str(np.round(self.z,3))}/snap={self.snap},"
        str_gal += f" with \nN={'%.1E'%Decimal(self.N_tot_part)} part.\nof \ntot Mass={'%.1E'%Decimal(self.M_tot)} [M_sun]\n"
        str_gal +=f" divided in N \n\
                Stars:{'%.1E'%Decimal(self.N_stars)}\n\
                Gas:{'%.1E'%Decimal(self.N_gas)}\n\
                DM:{'%.1E'%Decimal(self.N_dm)}\n\
                BH:{'%.1E'%Decimal(self.N_bh)}\n"
        str_gal +=f"and Mass in \n\
                    Stars:{'%.1E'%Decimal(self.M_stars)} [M_sun]\n\
                    Gas:{'%.1E'%Decimal(self.M_gas)} [M_sun]\n\
                    DM:{'%.1E'%Decimal(self.M_dm)} [M_sun]\n\
                    BH:{'%.1E'%Decimal(self.M_bh)} [M_sun]\n"
        return str_gal

# this function is a wrapper for convenience - it takes the class itself as input
def ReadGal(GAL,vebose=True):
    return LoadGal(path=GAL.pkl_path,verbose=verbose)

def LoadGal(path,verbose=True):
    if os.path.isfile(path):
        print("File "+path+" is present")
        try:
            return _LoadGal(path,verbose)
        except Exception as e:
            print("But failed to load: "+str(e))
            return False
    else:
        print("File not present")
        return False


def _LoadGal(path,verbose=True):
    with open(path,"rb") as f:
        GAL = pickle.load(f)
    if verbose:
        print(f"Loaded {GAL.pkl_path}")
    return GAL

def get_rnd_NG(sim=std_sim,min_mass = "1e12",min_z="0.2",max_z="2",
               pkl_name="massive_gals.pkl",check_prev=True,save_pkl=True):
    kw_gal   = get_rnd_gal_indexes(sim=sim,min_mass=min_mass,min_z=min_z,max_z=max_z,
                                   pkl_name=pkl_name,check_prev=check_prev,save_pkl=save_pkl)
    z        = kw_gal["z"] 
    M        = kw_gal["M"] 
    Gn,SGn   = kw_gal["Gn"],kw_gal["SGn"]
    Centre   = np.array([kw_gal["CMx"],kw_gal["CMy"],kw_gal["CMz"]])  
    #DEBUG
    #print("Gn,SGn,M,Centre,z")
    #print(Gn,SGn,M,Centre,z)
    NG       = NewGal(Gn,SGn,M,Centre,std_sim,z)
    return NG


def get_lens_dir(Gal):
    lens_dir = "sim_lens/"+str(Gal.sim)+"/snap"+str(Gal.snap)+"_G"+str(Gal.Gn)+"."+str(Gal.SGn)+"/"
    mkdir(lens_dir)
    Gal.lens_dir = lens_dir
    return lens_dir



# for debug:
if __name__=="__main__":
    
    NG = get_rnd_NG()
    plt.close()
    fig, ax = plt.subplots(3)
    nx = 100
    for name,part in zip(["stars","dm","gas"],[NG.stars,NG.dm,NG.gas]):
        coords = part["coords"]
        x,y,z  = coords.T
        print(np.std(coords,axis=0))
        ax[0].hist(x,bins=nx,alpha=.5,label=name)#,range=[xmin, xmax])
        ax[1].hist(y,bins=nx,alpha=.5,label=name)#,range=[ymin, ymax])
        ax[2].hist(z,bins=nx,alpha=.5,label=name)#,range=[ymin, ymax])
    ax[0].set_xlabel("X [kpc]")
    ax[2].set_xlabel("Z [kpc]")
    ax[1].set_xlabel("Y [kpc]")
    ax[2].legend()
    namefig = f"tmp/hist_by_hand_parts.png"
    plt.tight_layout()
    plt.savefig(namefig)
    plt.close()
    print("Saved "+namefig) 
    # up to now this make all sense -> not anymore!!-> not it works again
    """
    print("Testing dens map")
    from test_proj_part_hist import get_dens_map_rotate_hist
    NG.proj_dir = "./tmp/proj_part"
    mkdir(NG.proj_dir)
    dens_Ms_kpc2,radius,dP,cosmo = get_dens_map_rotate_hist(Gal=NG,pixel_num=100j,
                                                            z_source_max=5,verbose=True,plot=True)
    """