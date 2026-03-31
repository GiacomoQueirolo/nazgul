from python_tools.get_res import load_whatever
from nazgul.Translator.EAGLE import usnm_pwd_file
from nazgul.pathfinder import path_nazgul 
try:
    usnm_pwd = load_whatever(path_nazgul/usnm_pwd_file)
except FileNotFoundError:
    raise FileNotFoundError("To automate the connection, first run nazgul/Translator/EAGLE/setup_eagle_data.py")

import eagleSqlTools as sql

# this should not be written in plain text but oh well
con = sql.connect(usnm_pwd["usnm"], password =usnm_pwd["pwd"])

def exec_query_orig(query):
    data = sql.execute_query(con,query)
    return data

def exec_query(query):
    """Execute query and return simulation data
    """
    data = exec_query_orig(query)
    data = {name: data[name] for name in data.dtype.names}
    data["query"] = query
    # extract simulation name from query
    qr_split    = query.split("FROM")[1].split(" ")
    data["sim"] = list(filter(("").__ne__, qr_split))[0].split("_Subhalo")[0]
    return data

