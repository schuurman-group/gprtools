"""Module for defining and storing internal coordinates """

import numpy as np
import numpy.linalg as la

def normalize(vec):
    """normalize a vector. If norm is zero, return zero vector"""

    n = la.norm(vec)
    if n != 0:
        return vec/n
    else:
        return np.zeros(vec.shape[0], dtype=float)

# class that defines a single internal coordinate
class Intc:

    valid_types = ['stre', 'bend', 'tors', 'out']

    """Class constructor for an internal coordinate"""
    def __init__(self):
        self.ctypes       = []
        self.coefs        = []
        self.atoms        = []

    def copy(self):
        """
        return a copy of current object
        """
        new = Intc()
        new.ctypes = self.ctypes.copy()
        new.coefs = self.coefs.copy()
        new.atoms = self.atoms.copy()

        return new

    def add_prim(self, typ, coef, atms):
        """add a primitive to the coordinate"""

        if typ.lower() in self.valid_types:
            self.ctypes.append(typ)
            self.coefs.append(coef)
            self.atoms.append(atms)

    def n_prim(self):
        """return the number of primitives in the coordinate"""
        return len(self.ctypes)

    def types(self):
        """return the types of coordinates"""
        return self.ctypes

    def type(self, i):
        """return the type of coordinate i"""
        if i < self.n_prim():
            return self.ctypes[i]
        else:
            return None

    def coef(self, i):
        """return the coefficient on primitive i"""
        if i < self.n_prim():
            return self.coefs[i]
        else:
            return None

    def atms(self, i):
        """return the atoms associated with primitive i"""
        if i < self.n_prim():
            return self.atoms[i]
        else:
            return None

# class to hold the definition of an internal coordiante set
class Intdef:
    """Class constructor for Intdef object"""
    def __init__(self, intc_file = None):

        self.intcoords = []
        if intc_file is not None:
            self.read(intc_file)

    #
    def copy(self):
        """define function to copy Intdef object"""
        new = Intdef()
        new.intcoords = self.intcoords.copy()

        return new

    def read(self, file_name):
        """read in an internal coordinate defintion from file"""

        with open(file_name, 'r') as f:
            intf = f.readlines()

        # convert to lower case
        for i in range(len(intf)):
            intf[i] = intf[i].lower()

        ic  = -1
        new = False

        for i in range(len(intf)):
            line = intf[i]

            if line[0] == 'k':
                # if this is not the first coordinate, normalize
                # coefficients to normalization constant
                if len(self.intcoords) > 0:
                    cfs = np.asarray(self.intcoords[-1].coefs)
                    cfs *= nrm_con / la.norm(cfs)
                    self.intcoords[-1].coefs = cfs.tolist()

                self.intcoords.append(Intc())
                # if there is a factor that should applied to whole
                # coordinate, set it here
                if line.strip().split()[0] == 'k':
                    nrm_con = 1.
                else:
                    nrm_con = float(line.strip().split()[0].replace('k',''))

            if any([ctyp in line for ctyp in Intc.valid_types]):
                ind  = np.where([Intc.valid_types[i] in line
                          for i in range(len(Intc.valid_types))])[0][0]

                # pull out the type
                typ  = Intc.valid_types[ind]

                larr = line.strip().split()
                ind = np.where([typ in larr[i]
                               for i in range(len(larr))])[0][0]

                # now extract coefficient, somtimes adjacent to
                # to coordinate type
                if len(larr[ind].replace(typ,'')) > 0:
                    cf = float(larr[ind].lower().replace(typ,''))
                # sometimes coefficient is implied. Brutal.
                elif larr[ind-1] != 'k':
                    cf = float(larr[ind-1])
                else:
                    cf = 1.

                # now extract atoms
                # if 'K' in the string, then there will be an extra
                # number indicating coordinate index that we need to
                # skip
                if 'k' in larr[0]:
                    ind +=1
                atms = [int(float(larr[i]))-1
                                   for i in range(ind+1,len(larr))]

                # add the primitive to the current internal coordinate
                self.intcoords[-1].add_prim(typ, cf, atms)

        # convert final coord
        if len(self.intcoords) > 0:
            cfs = np.asarray(self.intcoords[-1].coefs)
            cfs *= nrm_con / la.norm(cfs)
            self.intcoords[-1].coefs = cfs.tolist()


    def n_q(self):
        """return the number of internal coordinates"""
        return len(self.intcoords)

    #
    def n_prim(self, i):
        """return the number of primitives associated with internal
            coordinate i"""
        return self.intcoords[i].n_prim()

    def q_atms(self, i):
        """return the atoms associated with ith internal coordinate"""
        return self.intcoords[i].atoms

    def q_types(self, i):
        """return the type of the ith internal coordinate"""
        return self.intcoords[i].ctypes

    def q_coefs(self, i):
            return self.intcoords[i].coefs

class Cart2int:
    """Class constructor for cart2int object"""
    def __init__(self, intc_def=None):
        self.intdef = intc_def

    #
    def copy(self):
        """define function to copy cart2int object"""
        new = Cart2int()

        new.intdef   = self.intdef.copy()
        return new

    #
    def cart2intc(self, geom):
        """convert a cartesian geometry into a set of internal
           coordinates"""

        return self.make_intc(geom)

    #
    def cart2intg(self, geom, grad):
        """convert a cartesian gradient into a gradient in terms
           of internal coordinates"""

        bmat   = self.make_bmat(geom)
        ni, nc = bmat.shape[0], bmat.shape[1]

        bbt    = bmat @ bmat.T
        bbtinv = la.pinv(bbt)

        return bbtinv @ bmat @ grad

    #
    def cart2inth(self, geom, hess):
        """convert a cartesian hessian into a hessian in terms of
           internal coordiantes"""

        bmat = self.make_bmat(geom)
        bhbt = bmat @ hess @ bmat.T

        gmat = bmat @ bmat.T
        ginv = la.pinv(gmat)

        return ginv @ bhbt @ ginv

    #
    def dint2cart(self, geom, dqs):
        """convert displacements in internal coordinates into a
           cartesian geometry"""

        maxs   = 1. 
        maxit  = 40
        maxdq  = 1.e-8

        bmat_ref = self.make_bmat(geom)
        q        = self.cart2intc(geom)
        ndq      = dqs.shape[0]
        dx_all   = np.zeros((ndq, geom.shape[0]), dtype=float)

        for i in range(ndq):
            dqi     = dqs[i,:].copy()
            newgeom = geom.copy()
            bmat    = bmat_ref.copy()
            nrmerr  = 1.
            it      = 0

            while nrmerr > maxdq and it < maxit:
                bbt    = bmat @ bmat.T
                bbtinv = la.pinv(bbt)
                bbtinq = bbtinv @ dqi
                dx     = bmat.T @ bbtinq

                if np.amax(np.abs(dx)) > maxs:
                    dx *= (maxs / np.amax(np.abs(dx)))

                newgeom += dx
                bmat   = self.make_bmat(newgeom)
                qnew   = self.cart2intc(newgeom)
                dqi    = (q + dqs[i]) - qnew
                nrmerr = la.norm(dqi)
                it    += 1

            dx_all[i, :] = newgeom - geom

            if nrmerr > maxdq:
                print('internal to cartesian displacements did not '+
                      ' converge for dq='+str(dqs[i]))

        return dx_all

    #
    def cart2intp(self, geom, momentum):
        """convert mommentum vector from cartesian coordinate to internal coordinate"""
        bmat = self.make_bmat(geom)
        momentum_inc = np.dot(bmat, momentum)
        # print("bmat:\n{:}".format(bmat))
        # print("momentum vector shape: {:}".format(momentum.shape))
        # print("momentum vector shape: {:}".format(momentum.shape))

        return momentum_inc

    #
    def make_bmat(self, geom):
        """construct a bmatrix at a particular geometry using internal
           coordinates defined in intdef"""

        ni   = self.intdef.n_q()
        nc   = geom.shape[0]
        na   = int(geom.shape[0]/3.)
        bmat = np.zeros((ni, nc), dtype=float)
        gm   = np.reshape(geom, (na, 3), order='C')

        for i in range(ni):
            #print('self.intdef.qcoef='+str(self.intdef.q_coefs(i)))
            for j in range(self.intdef.n_prim(i)):
                # print('i,j,row='+str(i)+','+str(j)+':'+str(self.bmat_row(
                                           # gm,
                                           # self.intdef.q_types(i)[j],
                                      # self.intdef.q_atms(i)[j])))
                # print("row:{:},col:{:},type:{:}".format(i,j,self.intdef.q_types(i)[j]))
                bmat[i, :] += self.intdef.q_coefs(i)[j] * self.bmat_row(
                                            gm,
                                            self.intdef.q_types(i)[j],
                                            self.intdef.q_atms(i)[j])

        return bmat

    #
    def make_gmat(self, geom, mass):
        """calcuate G matrix from bmat and mass"""
        bmat = make_bmat(geom)
        return np.einsum('aj,j,bj->ab',bmat, 1./mass, bmat)

    #
    def make_intc(self, geom):
        """construct a bmatrix at a particular geometry using internal
           coordinates defined in intdef"""

        ni   = self.intdef.n_q()
        na   = int(geom.shape[0]/3.)
        q    = np.zeros(ni, dtype=float)
        gm   = np.reshape(geom, (na, 3), order='C')

        for i in range(ni):
            for j in range(self.intdef.n_prim(i)):
                q[i] += self.intdef.q_coefs(i)[j] * self.intc_val(
                                      gm,
                                      self.intdef.q_types(i)[j],
                                      self.intdef.q_atms(i)[j])

        return q

    #
    def bmat_row(self, geom, typ, atms):
        """return the value of the internal coordiante and the
           bmatrix elements evaluate at geometry 'geom' """

        if typ == 'stre':
            return self.bdist(geom, atms)
        elif typ == 'bend':
            return self.bangle(geom, atms)
        elif typ == 'tors':
            return self.btors(geom, atms)
        elif typ == 'out':
            return self.boop(geom, atms)
        else:
            print('internal coord. '+str(typ)+' not recognized')

    #
    def intc_val(self, geom, typ, atms):
        """evaluate value of internal coordinate at geometry geom"""

        if typ == 'stre':
            return self.qdist(geom, atms)
        elif typ == 'bend':
            return self.qangle(geom, atms)
        elif typ == 'tors':
            return self.qtors(geom, atms)
        elif typ == 'out':
            return self.qoop(geom, atms)
        else:
            print('internal coord. '+str(typ)+' not recognized')

    #
    def qdist(self, geom, atms):
        """return the value of the internal coordinate and corresponding
           bmatrix elements"""
        return la.norm(geom[atms[0],:] - geom[atms[1],:])

    #
    def qangle(self, geom, atms):
        """return the value of the internal coordinate and corresponding
           bmatrix elements"""
        u = normalize(geom[atms[0],:] - geom[atms[2],:])
        v = normalize(geom[atms[1],:] - geom[atms[2],:])

        return np.arccos( np.dot(v,u) )

    #
    def qtors(self, geom, atms):
        """return the value of the internal coordinate and corresponding
           bmatrix elements"""

        _shift_ = 0.5*np.pi

        u = normalize(geom[atms[0],:] - geom[atms[1],:])
        v = normalize(geom[atms[2],:] - geom[atms[1],:])
        w = normalize(geom[atms[2],:] - geom[atms[3],:])
        z = normalize(np.cross(u, v))
        x = normalize(np.cross(w, v))

        co  = np.dot(z, x)
        tau = self.atan(-co)

        zx  = np.cross(z, x)
        co2 = np.dot(zx, v)

        if co2 < 0:
            tau *= -1
        if tau > _shift_:
            tau -= 2*np.pi
        if tau <= -2*np.pi:
            tau += 2*np.pi

        return -tau

    #
    def atan(self, x):
        """
        returns an arctran result that checks for argument bounds
        """
        if x >= 1.:
            return 0.

        if x <= -1.:
            return np.pi

        if abs(x) < 1.e-12:
            return 0.5*np.pi

        x1 = np.sqrt(1 - x**2)
        s = np.arctan( x1 / x)
        if x < 0.:
            s += np.pi

        return s

    #
    def qoop(self, geom, atms):
        """return the value of the internal coordinate and corresponding
           bmatrix elements"""
        u = normalize(geom[atms[0],:] - geom[atms[3],:])
        v = normalize(geom[atms[1],:] - geom[atms[3],:])
        w = normalize(geom[atms[2],:] - geom[atms[3],:])
        z  = normalize(np.cross(v, w))

        stheta = np.dot(u, z)
        ctheta = np.sqrt(1 - stheta**2)

        if ctheta >= 1.:
            return 0.
        if ctheta <= -1.:
            return np.pi
        if abs(ctheta) <= 1.e-11:
            return 0.5*np.pi

        y     = np.sqrt(1-ctheta**2)
        gamma = np.arctan(y/ctheta)
        if ctheta < 0:
            gamma += np.pi
        if stheta < 0:
            gamma *= -1.

        return gamma

    #
    def bdist(self, geom, atms):
        """return the value of the internal coordinate and corresponding
           bmatrix elements"""

        brow = np.zeros(geom.shape[0] * geom.shape[1], dtype=float)

        v    = normalize(geom[atms[0],:] - geom[atms[1],:])

        brow[ 3*atms[0] : 3*atms[0] + 3 ] =  v
        brow[ 3*atms[1] : 3*atms[1] + 3 ] = -v

        return brow

    #
    def bangle(self, geom, atms):
        """return the value of the internal coordinate and corresponding
           bmatrix elements"""

        brow = np.zeros(geom.shape[0] * geom.shape[1], dtype=float)

        u = geom[atms[0],:] - geom[atms[2],:]
        v = geom[atms[1],:] - geom[atms[2],:]
        un = normalize(u)
        vn = normalize(v)
        nu = la.norm(u)
        nv = la.norm(v)

        ctheta = np.dot(un, vn)
        stheta = np.sqrt( 1 - ctheta**2 )

        if stheta == 0:
            print('Linear angle detected: use linear bend coordinates')
            return brow

        v1 = (ctheta*un - vn) / (stheta*nu)
        v2 = (ctheta*vn - un) / (stheta*nv)
        brow[ 3*atms[0] : 3*atms[0] + 3 ] = v1
        brow[ 3*atms[1] : 3*atms[1] + 3 ] = v2
        brow[ 3*atms[2] : 3*atms[2] + 3 ] = -v1 - v2

        return brow

    #
    def btors(self, geom, atms):
        """return the value of the internal coordinate and corresponding
           bmatrix elements"""

        brow = np.zeros(geom.shape[0] * geom.shape[1], dtype=float)

        u = geom[atms[0],:] - geom[atms[1],:]
        v = geom[atms[2],:] - geom[atms[1],:]
        w = geom[atms[2],:] - geom[atms[3],:]

        nu = la.norm(u)
        nv = la.norm(v)
        nw = la.norm(w)

        un = u / nu
        vn = v / nv
        wn = w / nw

        z = normalize(np.cross(un, vn))
        x = normalize(np.cross(wn, vn))

        cos  = np.dot(un, vn)
        cos2 = np.dot(vn, wn)

        sin  = np.sqrt(1-cos**2)
        sin2 = np.sqrt(1-cos2**2)

        v0 = z / (nu * sin)
        v3 = x / (nw * sin2)
        v1 = (nu*cos/nv - 1.)*v0 - (nw*cos2/nv)*v3
        v2 = -v0 - v1 - v3
        brow[3*atms[0] : 3*atms[0]+3] = v0
        brow[3*atms[1] : 3*atms[1]+3] = v1
        brow[3*atms[2] : 3*atms[2]+3] = v2
        brow[3*atms[3] : 3*atms[3]+3] = v3

        return brow

    #
    def boop(self, geom, atms):
        """return the value of the internal coordinate and corresponding
           bmatrix elements"""

        brow = np.zeros(geom.shape[0] * geom.shape[1], dtype=float)

        u = geom[atms[0],:] - geom[atms[3],:]
        v = geom[atms[1],:] - geom[atms[3],:]
        w = geom[atms[2],:] - geom[atms[3],:]

        nu = la.norm(u)
        nv = la.norm(v)
        nw = la.norm(w)

        un = u/nu
        vn = v/nv
        wn = w/nw

        z  = np.cross(vn, wn)
        zn = z/la.norm(z)

        stheta = np.dot(un, zn)
        ctheta = np.sqrt(1 - stheta**2)

        cf1 = np.dot(vn, wn)
        sf1 = np.sqrt(1 - cf1**2)
        cf2 = np.dot(wn, un)
        cf3 = np.dot(vn, un)

        denom = ctheta * sf1**2
        b2  = (cf1*cf2 - cf3) / (nv*denom)
        b3  = (cf1*cf3 - cf2) / (nw*denom)

        v1 = zn * b2
        v2 = zn * b3
        brow[3*atms[1] : 3*atms[1] + 3] = v1
        brow[3*atms[2] : 3*atms[2] + 3] = v2

        x = np.cross(zn, un)
        xn = x / la.norm(x)

        y = np.cross(un, xn)
        yn = y / la.norm(y)

        v0 = yn / nu

        brow[3*atms[0] : 3*atms[0] + 3] = v0
        brow[3*atms[3] : 3*atms[3] + 3] = -v0 - v1 - v2

        return brow
