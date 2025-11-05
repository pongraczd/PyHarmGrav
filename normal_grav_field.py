import numpy as np

class Ellipsoid:
    def __init__(self,param):
        if isinstance(param , str):
            self.name = param.lower().strip()
        else:
            self.name = 'custom'
        if self.name == 'wgs84':
            self.GM = 3986004.418e8                 # Geocentric gravitational constant of WGS84
            self.a = 6378137                        # Semimajor axis of WGS84
            self.f = 1/298.257223563                # Flattening of WGS84
            self.C_20 =  -0.484166774985e-3         #Fully normalized C_20 of WGS84
            self.omega = 7.292115e-05               # Angular velocity of Earth
            self.e=np.sqrt((self.f)*(2-(self.f)))   # First eccentricity
        elif self.name == 'grs80':
            self.GM=3986005e8                       # Geocentric gravitational constant of GRS80
            self.a=6378137                          # Semimajor axis of GRS80
            self.f = 1/298.257222101                # Flattening of GRS80
            self.C_20=-108263e-8/np.sqrt(5)         # Fully normalized C_20 of GRS80
            self.omega = 7.292115e-05               # Angular velocity of Earth
            self.e=np.sqrt((self.f)*(2-(self.f)))   # First eccentricity
        else:
            if not(isinstance(param,list) or isinstance(param,tuple),isinstance(param,dict)):
                raise ValueError('Parameters must be supplied as string of Ellipsoid name or with a parameter vector containing \
                           [GM a e C20 omega] or with a dictionary with these keys')
            if len(param)!=5:
                raise ValueError('5 parameters are needed : [GM a e C_20 omega]')
            if isinstance(param,dict):
                try:
                    self.GM = param['GM']
                    self.a = param['a']
                    self.e = param['e']
                    self.C_20 = param['C_20']
                    self.omega = param['omega']
                except KeyError:
                    raise KeyError("At least 1 key is incorrect. These keys needed: 'GM','a','e','C_20','omega'")
            else:
                self.GM,self.a,self.e,self.C_20,self.omega = param
            self.f = 1-np.sqrt(1-(self.e)**2)

        e0 = self.e / np.sqrt(1-(self.e)**2)
        q0 = ((1+3/(e0**2))*np.arctan(e0) - 3 / e0) /2
        q0v = 3*(1 + 1/(e0**2))*(1-(1/e0)*np.arctan(e0)) - 1
        b = (self.a)*np.sqrt(1-(self.e)**2)
        self.m = ((self.omega)**2*(self.a)**2*b)/(self.GM)
        self.gamma_e = (self.GM)/((self.a)*b) * (1 - self.m - ((self.m)*e0*q0v)/(6*q0))
        gamma_p = (self.GM)/((self.a)**2) * (1 + ((self.m)*e0*q0v)/(3*q0))
        self.k = (b*gamma_p)/((self.a)*(self.gamma_e)) - 1
    def __repr__(self):
        return f"Ellipsoid(name='{self.name}', GM={self.GM}, a ={self.a}, e ={self.e}, C_20 ={self.C_20}, omega ={self.omega}, f = {self.f}, m ={self.m}, gamma_e={self.gamma_e} , k={self.k})\n"
    
    def normalklm(self,GM,R):
        """
        Normal gravity field until degree 20 (even degrees computed - odd degrees are zero)
        based on level ellipsoid - GRS80 or WGS84
        Parameters
        ----------  
        typ : str       Type of ellipsoid, 'wgs84' or 'grs80'       
        GM : float      Geocentric gravitational constant of the Earth (m^3/s^2)        
        R : float       Reference radius (m)
        """
        GMEl = self.GM
        aEl = self.a
        CEl_20 = self.C_20
        eEl = self.e
        n = np.arange(0, 11)
        CEl=((-1)**n*(3*eEl**(2*n))/((2*n+1)*(2*n+3)*np.sqrt(4*n+1))*(1-n-5**(3/2)*n*CEl_20/eEl**2))*(aEl/R)**(2*n)*(GMEl/GM)
        l = np.arange(0, 21, 2)
        return CEl,l
    
    def subtract_normal_field(self,shcs,nmin=0):
        """
        Subtract normal gravity field coefficients from the spherical harmonic coefficients.
        The normal gravity field coefficients are computed up to degree 20.
    
        Parameters
        ---------
        shcs :  pyharm.shc.Shc      Spherical harmonic coefficients
        nmin : int                  Minimum number of expansion
        """
        # compute SH coefficients normal gravity field
        CEl,l = self.normalklm( shcs.mu, shcs.r)
        # handle case where normal field coefficients are computed over nmax
        CEl = CEl[l<=shcs.nmax]
        l = l[l<=shcs.nmax]
        # get low-degree coefficients to subtract normal field from
        low_c_coeffs,_ = shcs.get_coeffs(n=l,m=np.zeros(l.shape,dtype=int))
        # subtract normal field
        low_c_coeffs = low_c_coeffs - CEl
        # replace coefficients
        if nmin>0: # do not touch coefficients below nmin (zeroed out before)
            cond = (l>nmin)
            l = l[cond]
            low_c_coeffs = low_c_coeffs[cond]
        if len(l) == 0:
            return
        shcs.set_coeffs(n=l, m=np.zeros(l.shape,dtype=int), c=low_c_coeffs)
    
