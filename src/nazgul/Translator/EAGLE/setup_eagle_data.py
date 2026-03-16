import os
import dill

from python_tools.get_res import load_whatever
from nazgul.Translator import min_z as std_min_z
from nazgul.Translator import max_z as std_max_z
from nazgul.Translator.EAGLE.fnct import get_snap
from nazgul.Translator.EAGLE import sim,usnm_pwd_file
from nazgul.pathfinder import std_data_dir as std_datadir_nazgul

std_datadir_EAGLE = "~/EAGLE_particle_data"
download_prog = "Translator/EAGLE/download_eagle_data.sh "

if __name__=="__main__":
    print("Script to setup the EAGLE data")
    print("\n\n")
    print("1) Create an account in https://virgodb.dur.ac.uk/")
    input("Press Enter once you have activated the account to continue...\n")
    print("2) Input here your username and password - note: this process is not the safest in principle")
    
    
    usnm = input(" - Username: ")
    pwd  = input(" - Password: ")
    
    if usnm=="" and pwd=="":
        try:
            usnm_pwd = load_whatever(usnm_pwd_file)
            usnm = usnm_pwd["usnm"]
            pwd  = usnm_pwd["pwd"]
            print("Loaded previously stored username and password")
        except FileNotFoundError:
            raise RuntimeError("Input a username and password")
    else:
        with open(usnm_pwd_file,"wb") as f:
            dill.dump({"usnm":usnm,"pwd":pwd},f)

    print("3) Download required EAGLE datasets")
    std_sim = str(sim[0])
    sim = str(input("   - Which simulation? (default:"+std_sim+")\n   ").strip() or std_sim)
    
    print("   - Which redshift range? (default: "+str(std_min_z)+"-"+str(std_max_z)+")")
    min_z = float(input("    - min z= ").strip() or 0.02)
    max_z = float(input("    - max z= ").strip() or 2)
    
    
    max_snap = get_snap(min_z)
    min_snap = get_snap(max_z)
    
    datadir_EAGLE = str(input("   - Where to store the data? (default: "+std_datadir_EAGLE+")\n   ").strip() or str(std_datadir_EAGLE))
    
    std_datadir_nazgul = str(std_datadir_nazgul)
    datadir_nazgul = str(input("   - Where to link the data in the nazgul structure? (best to leave it default: "+std_datadir_nazgul+")\n   ").strip() or str(std_datadir_nazgul))
    
    print("Running download...\n\n")
    
    #note: the order is important
    inputs = " ".join([usnm,pwd,sim,datadir_EAGLE,datadir_nazgul,min_snap,max_snap])
    command_str = download_prog +inputs 
    os.system(command_str)
