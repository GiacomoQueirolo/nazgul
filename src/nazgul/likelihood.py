# Likelihood classes 
# for now start small, with the idea of expanding it
import emcee
import numpy as np

# liberally and heavily inspired by lenstronomy Sampling
from lenstronomy.Sampling.Pool.pool import choose_pool

"""
kw_like = {"like_func"=func(x,params),
           "like_prms"=[params]}
"""
class Likelihood():
    def __init__(self,var_range,kw_like=None):
        if len(np.shape(var_range))!=2:
            var_range = [var_range]
        self.var_range = np.array(var_range)
        self.n_params  = len(var_range)
        self.kw_like   = kw_like

    def logL(self,x):
        if x>self.var_range.max(axis=1) or x<self.var_range.min(axis=1):
            return -1e18 
        like = 1
        if self.kw_like is not None:
            f     = kw_like["like_func"]
            prms  = kw_like["like_prms"]
            like *= f(x,*prms)
        logL = np.log(like)
        return logL

    def sample(self,n_samples=1,n_burn=100,threadCount=1,initpos=None,n_walkers=10,progress=True):
        """Sample the likelihood
        """
        pool = choose_pool(mpi=False, processes=threadCount, use_dill=True)
        
        sampler = emcee.EnsembleSampler(
            n_walkers, self.n_params, self.logL, pool=pool, backend=None
        )
    
        n_run_tot = n_burn+n_samples
        if initpos is None:
            initpos = np.random.uniform(self.var_range.min(axis=1),self.var_range.max(axis=1),n_walkers)
            initpos = initpos.reshape((n_walkers,1))
        sampler.run_mcmc(initpos, n_run_tot, progress=progress)
        samples = sampler.get_chain(discard=n_burn, thin=1, flat=True).T[0]
        return samples
        
