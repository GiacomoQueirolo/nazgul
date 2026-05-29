from nazgul.configurations import nazgul_path

simsuite_name ="EAGLE"

# Available simulations:
sim     = ["RefL0025N0752","RefL0012N0188","RefTuto"]

# File where to store the credentials:
usnm_pwd_file = nazgul_path/"Translator/EAGLE/.eagle_account.dll"



# Define name of accepted particles
part_type_list = ["stars","gas","dm","bh"]

def check_part_type(part_type):
    assert type(part_type)==str
    # Trying to recover the part type avoiding stupid typos 
    part_type = part_type.lower()
    if "star" in part_type:
        part_type="stars"
    if "dark" in part_type:
        if "matter" in part_type:
            part_type = "dm"
        else:
            raise ValueError(f"Particle type {part_type} not recognised, do you mean dm (dark matter)?")
    if "hole" in part_type:
        if "black" in part_type:
            part_type = "bh"
        else:
            raise ValueError(f"Particle type {part_type} not recognised, do you mean bh (black holes)?")
    if not part_type in part_type_list:
        raise ValueError(f"Particle type {part_type} not an accepted particle type:\n{part_type_list}")
    return part_type
