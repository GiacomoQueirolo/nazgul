import numpy as np

class BasicGal:
    """General useful class for galaxies (being ensemble of particles or already lenses)
    """
    # these will be attributes to not store and to recompute
    _large_attributes = []
    ### Class Structure ####
    ########################
    def _identity(self):
        # Returns tuple to identify uniquely this instance
        raise NotImplementedError
        
    def __hash__(self):
        # simplify the hash method
        return hash(self._identity())

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._identity() == other._identity()

    def __str__(self):
        raise NotImplementedError 
        
    def __getstate__(self):
        state = self.__dict__.copy()
        # remove large attributes (if present, can be loaded again)
        if self._large_attributes == []:
            raise NotImplementedError("Implement a list of     _large_attributes to delete before storing")
        for lg_att in self._large_attributes:
            state.pop(lg_att, None)
        return state

    def __setstate__(self, state):
        # Optional: restore defaults or trigger rebuild of heavy attributes
        self.__dict__.update(state)

    def _needs_unpacking(self):
        """Check whether the object is missing reconstructed attributes.
        """
        print("DEBUG")
        for attr in self._large_attributes:
            if not hasattr(self,attr):
                print(f"missing {attr}")
        return not all(
            hasattr(self, attr)
            for attr in self._large_attributes
        )
        
    def unpack(self):
        """Public wrapper for lazy reconstruction.
        """
        if self._needs_unpacking():
            self._unpack()
        return self

    def _unpack(self):
        """Reconstruct all attributes that were intentionally removed
        before serialization.
        """
        print("Unpacking class...")
        raise NotImplementedError

    ########################
    ########################
    def ReadClass(self,cl):
        # e.g. return ReadGal(cl)
        raise NotImplementedError
        
    def upload_prev(self,reload=True):
        if not reload:
            return False
        prev_Class = self.ReadClass(self)
        if prev_Class is False or prev_Class != self:
            return False
        # if common attribute, they are overwritten by previous:
        self.__dict__ = {**self.__dict__,**prev_Class.__dict__}
        return True
        
    def verbose_assert_almost_equal(self,value1,value2=1,decimal=3,msg_title=None):
        # a verbose way of giving info if if fails
        try:
            np.testing.assert_almost_equal(value1,value2,decimal=decimal)
        except AssertionError as AssErr:
            if msg_title:
                print(msg_title)
            print("Error for \n"+str(self))
            raise AssertionError(AssErr)
        return 0