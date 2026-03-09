# dirty little script to produce many GL 
import numpy as np
from nazgul.mount_doom.generate_particle_lens_dom import wrapper_get_rnd_lens

aim_n_lenses = 200
n_lenses = 0
failed_lenses=0
while n_lenses<aim_n_lenses:
    try:
        lns = wrapper_get_rnd_lens(reload=True)
        perc = n_lenses*100/aim_n_lenses
        if perc%10==0:
            print(f"\n\n###########\n{np.round(perc,2)}% completed\n###############\n\n")
        n_lenses+=1
    except Exception as e:
        failed_lenses +=1
        print("Skipped galaxy due to error:\n"+str(e)) 
    if failed_lenses+n_lenses>500:
        print("Too many failures")
        print("Exiting")
        break
print("Stat: \n")
print("Fail: "+str(failed_lenses),"\n","Success: "+str(n_lenses))
print("Success!")
