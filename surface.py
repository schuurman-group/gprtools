"""
The Surface ABC
"""
import os
import shutil
import numpy as np
import itertools
from abc import ABC, abstractmethod
import graci.core.libs as libs
import chempotpy
import constants as constants

class Surface(ABC):

    def __init__(self):
        super().__init__()
        self.have_gradients = False
        self.have_coupling  = False

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def gradient(self):
        pass

    @abstractmethod
    def hessian(self):
        pass

    @abstractmethod
    def coupling(self):
        pass

#
class Graci(Surface):
    """
    GRaCI surface evaluator
    """
    def __init__(self, ci_obj, nstates, scf_obj=None, mol_obj=None):
        super().__init__()
        self.graci_ci    = None
        self.graci_scf   = None
        self.graci_mol   = None
        self.nstates     = None
        self.ci_type     = ''
        self.nroots      = 0
        self.valid_types = ['dftmrci','dftmrci2']

        ci_type  = ci_obj.__class__.__name__.lower()
        if ci_type not in self.valid_types:
            print('CI type: ' + str(ci_type) +
                  ' not a recognized GRaCI object',flush=True)
            return None

        # set the CI object and quiet output
        self.graci_ci = ci_obj.copy()
        self.ci_type  = self.graci_ci.__class__.__name__.lower()
        self.nstates  = nstates
        self.nroots   = self.graci_ci.n_states()
        self.graci_ci.verbose  = False

        # set the SCF object and quiet output
        if scf_obj is not None:
            self.graci_scf = scf_obj.copy()
        else:
            self.graci_scf = self.graci_ci.scf.copy()
        self.graci_scf.verbose = False

        # set the Molecule object
        if mol_obj is not None:
            self.graci_mol = mol_obj.copy()
        else:
            self.graci_mol = self.graci_scf.mol.copy()

        # load the bitci shared library
        libs.lib_load('bitci')

    #
    def evaluate(self, gms, scr_dir=None, propagate=True, clean=True):
        """
        evaluate the energy at passed geometry, gm
        """

        # move to appropriate scratch directory
        if scr_dir is None:
            tmpdir = 'scratch'
        else:
            tmpdir = scr_dir+'/scratch'
        os.mkdir(tmpdir)
        os.environ['PYSCF_TMPDIR'] = tmpdir
        os.chdir(tmpdir)

        # get info about the ci calc to be run
        atms    = self.graci_ci.scf.mol.asym
        natm    = len(atms)

        # if the number of states is not specified, use
        # the default number of stats in ci object
        self.nroots = self.nstates
        self.graci_ci.nstates = np.asarray([self.nroots], dtype=int)

        energies = np.zeros((gms.shape[0], self.nroots), dtype=float)
        scf_fail = []
        ci_fail  = []

        # sort geometries so they're in optimal order for propagating
        # orbitals and/or reference spaces
        origin     = self.graci_mol.cart().flatten(order='C')
        ordr, dist = self.sort_geoms(origin, gms)

        # iterate over all gms passed, propagating orbitals and reference
        # space as we go
        scf_guess = None
        ci_guess  = None
        for i in range(len(ordr)):
            geom = gms[ordr[i],:]

            # update the geometry
            self.graci_mol.set_geometry(atms, geom.reshape(natm,3))
            self.graci_mol.run()

            # run the KS-SCF, use previous scf as a guess by default
            scf_ener = self.graci_scf.run(self.graci_mol, scf_guess)

            # if the scf failed, label as such and move to next
            # geometry
            if scf_ener is None:
                scf_fail.append(ordr[i])
                continue

            # if we're propgating the scf orbitals, update the scf
            # guess
            elif propagate:
                scf_guess = self.graci_scf.copy()

            # run the CI. Use previous reference space as a guess by
            # default. Currently only enabled for DFT/MRCI(2)
            conv = self.graci_ci.run(self.graci_scf, ci_guess)

            if conv:
                energies[ordr[i],:] = np.asarray(self.graci_ci.energy(
                                     range(self.nroots)), dtype=float)
                # if we're propagating the CI reference space,
                # update the ci_guess
                if propagate:
                    ci_guess = self.graci_ci.copy()
            else:
                ci_fail.append(ordr[i])

        # move back up to main directory
        os.chdir('../')

        #remove scratch if requested
        if clean:
            shutil.rmtree('scratch')

        return energies, scf_fail, ci_fail

    #
    def gradient(self, geoms):
        """
        not defined for GRaCI surfaces
        """
        return None

    #
    def coupling(self, geoms):
        """
        time-derivative couplings will be added in the future
        """
        return None

    #
    def sort_geoms(self, origin, geoms):
        """
        sort geometries so next geometry corresponds to minimum change
        at each step
        """

        gms = np.vstack([origin, geoms])

        # construct tensor that is all unique differences
        r,c = np.triu_indices(gms.shape[0], 1)
        dif = gms[r,:] - gms[c,:]

        # compute the distances between all unique pairs of geoms
        dist = np.sqrt(np.einsum('ij,ij->i',dif, dif))

        # construct the distance matrix
        dmat      = -np.identity(gms.shape[0], dtype=float)
        dmat[r,c] = dmat[c,r] = dist

        # order the geometries so each step takes you to closest
        # unique geometry
        ordr    = []
        ndist   = []
        current = 0
        for i in range(geoms.shape[0]):
            valid   = np.where(dmat[current,:] >= 0.)[0]
            nearest = valid[dmat[current, valid].argmin()]
            mindist = dmat[current, nearest]
            ndist.append(mindist)
            # decrement closest by 1: first geometry is the origin
            ordr.append(nearest-1)
            # remove this pair as a future possibility
            dmat[current, :] = dmat[:, current] = -1
            # move to next geometry
            current = nearest

        # if something goes wrong, return just sequential ordering
        if len(set(ordr)) != geoms.shape[0]:
            print('error sorting geometries: ' + str(len(set(ordr))) +
                  ' != '+str(geoms.shape[0]))
            ordr = [i for i in range(geoms.shape[0])]

        return ordr, ndist

#
class Kdc(Surface):
    """
    KDC Vibronic surface evaluator
    """
    def __init__(self):
        super().__init__()
        self.ham            = Kdc_ham()
        self.nmodes         = None
        self.nstates        = None
        self.norder         = 0
        self.have_gradients = True
        self.have_coupling  = True

        if os.path.isfile(op_file):
            self.ham.parse_op_file(op_file)
        else:
            return None

        self.nmodes  = self.ham.nmodes
        self.nstates = self.ham.nstates

    #
    def evaluate(self, gms, n_s=None, rep='adiabatic'):
        """
        Evaluate the energies  energies
        """
        if n_s is not None and n_s <= self.ham.nstates:
            nst = n_s
        else:
            nst = self.ham.nstates

        energies = np.zeros((gms.shape[0], nst), dtype=float)
        fail     = []

        for i in range(gms.shape[0]):
            energy = self.ham.energy(gms[i,:], rep=rep)
            if energy is None:
                fail.append(i)
            else:
                energies[i,:] = energy[:nst]

        return energies, fail

#
class Kdc_ham():
    """
    class for holding Taylor expanded potentials
    """
    def __init__(self):
        self.cfs     = None
        self.terms   = None
        self.nmodes  = None
        self.nstates = None

    #
    def h(self, gm):
        """
        evaluate the hamiltonian for given geometry
        """

        # might as well initialize to the constant values
        h = self.cfs[0].copy()

        for n in range(1, len(self.cfs)):
            if len(self.terms[n]) > 0:
                for ordr in range(self.cfs[n].shape[2]):
                    exps   = self.terms[n][ordr]
                    tensor = np.power(gm, exps[0])
                    for i in range(1, len(exps)):
                        tensor = np.outer(tensor, np.power(gm, exps[i]))

                    h += np.einsum('ij...,...->ij',
                                       self.cfs[n][:,:,ordr,...],
                                       tensor, optimize=True)

        return h

    #
    def energy(self, gm, rep='adiabatic'):
        """
        return either the adiabatic or diabatic energies
        """

        if rep == 'adiabatic':
            hmat = self.h(gm)
            #print('hmat='+str(hmat))
            e, vec = np.linalg.eigh(self.h(gm))
        else:
            e = np.diagonal(self.h(gm))

        return e

    #
    def parse_op_file(self, op_file):
        """
        static method for parsing quantics input file, return a
        kdc_ham object
        """

        if not os.path.isfile(op_file):
            return None

        unit_conv = {'ev': 1./27.2114, 'au': 1.}

        # nst      = total number of states
        # nq       = the number of physical modes
        # n_max    = the 'n' in the n-HDMR represntation
        # ordr_max = for each n-mode term, the highest polynomial
        #            order that is present
        nst, nq, n_max, ordr_max, el = self.scan_op_file(op_file)
        self.nstates = nst
        self.nmodes  = nq

        # generate a list of terms in the hamiltonian
        self.terms = [[] for n in range(n_max+1)]
        for n in range(n_max+1):
            if n > 0:
                for k in range(n,ordr_max[n]+1):
                    self.terms[n].extend(self.partition(n,k))

        # initialize the coefficient arrays
        self.cfs = []
        for n in range(n_max+1):
            dim     = (nst, nst)
            if len(self.terms[n]) > 0:
                dim += (len(self.terms[n]),)
                dim += (nq,)*n
            self.cfs.append(np.zeros(dim, dtype=float))

        # now we parse the file again and fill in all the
        # non-zero terms
        kdc_params = {}
        nq         = 0
        with open(op_file, 'r') as f:
            line       = f.readline()
            read_param = False
            param_done = False
            read_ham   = False
            ham_done   = False

            while line:

                if 'end-parameter-sec' in line:
                    param_done = True
                    read_param = False
                elif 'parameter-section' in line and not param_done:
                    read_param = True
                elif 'end-hamiltonian-sec' in line:
                    read_ham   = False
                    ham_done   = True
                elif 'hamiltonian-section' in line and not ham_done:
                    read_ham   = True

                # store all the parameters in a dictionary
                if read_param and not param_done:
                    key, value, units = self.parse_param_line(line)

                    # if we couldn't parse this line, move on
                    if key is not None:
                        #..else set the parameter
                        kdc_params[key] = float(value)*unit_conv[units]

                if read_ham and not ham_done:
                    num, key, qlst, stlst = self.parse_term_line(line, el)

                    # if we couldn't parse this line, move on
                    if num is not None:
                        # else, set the appropriate arrays
                        n, indices, fac = self.get_cf_index(nst, stlst, qlst)
                        if n is not None:
                            for ind in indices:
                                self.cfs[n][ind] += num * kdc_params[key] / fac
                        else:
                            print('ERROR: term not found -- STATES=' +
                                  str(stlst) + ' Q=' + str(qlst))

                line = f.readline()

        #print('cfs='+str(self.cfs),flush=True)

        return

    #
    def get_cf_index(self, nst, states, crds):
        """
        return in indices corresponding to the states and crds arrays
        """

        # determine how many unique modes
        uniq  = list(set(crds))
        nhdmr = len(uniq)

        if len(crds) > 0:
            term  = [crds.count(un) for un in uniq]
            term.sort(reverse=True)
            try:
                oind  = self.terms[nhdmr].index(term)
            except ValueError:
                return None, None

        # determine state blocks
        if len(states) == 0:
            sts = [[i,i] for i in range(nst)]
        else:
            sts = [ states, [states[1],states[0]] ]

        # number of coordinate permutations
        crd_perm = set(list(itertools.permutations(uniq)))

        inds = []
        for st in sts:
            ind = (st[0],st[1],)

            if len(self.terms[nhdmr]) > 0:
                ind += (oind,)

                # we're using unrestricted summations`
                for perm in crd_perm:
                    inds.append(ind+perm)

            else:
                inds.append(ind)

        inds = list(set(inds))
        return nhdmr, inds, max(len(crd_perm),1)

    #
    def scan_op_file(self, op_file):
        """
        scan an operator file to determine the number of states,
        modes, etc.
        """

        read_ham = False
        ham_done = False
        el       = None

        crd_lst   = []
        ordr_max  = [0]*8
        stmax     = 0
        nmode_max = 0
        nstates   = 0

        with open(op_file,'r') as f:
            line = f.readline()
            while line:

                # only read the first hamiltonian section
                if 'end-hamiltonian-section' in line:
                    ham_done = True
                    read_ham = False
                elif 'hamiltonian-section' in line:
                    read_ham = True

                if read_ham and not ham_done:

                    if all(x in line for x in ["S","&"]):
                        l_arr  = line.strip().split()
                        st_str = l_arr[-1]
                        ind    = st_str.index("&")
                        stmax  = max(int(st_str[1:ind]),
                                 int(st_str[ind+1:]))
                        if stmax > nstates:
                            nstates = stmax

                    if 'modes|' in line:
                        crd = line.replace('modes','').replace('|',' ')
                        crd_lst.extend(crd.strip().split())
                        if 'el' in crd_lst:
                            el = crd_lst.index('el')

                    else:
                        num, key, qlst, slst = self.parse_term_line(line,el)
                        if num != None:
                            ordr = len(qlst)
                            nm   = len(set(qlst))
                            if nm > nmode_max:
                                nmode_max = nm
                            if ordr > ordr_max[nm]:
                                ordr_max[nm] = ordr

                line = f.readline()

        nq = len(crd_lst) - crd_lst.count('el')
        return nstates, nq, nmode_max, ordr_max[:nmode_max+1], el

    #
    def parse_param_line(self, line):
        """
        parse a parameter line
        """
        if '=' in line:
            eq    = line.index('=')
            cm    = line.index(',')
            key   = line[:eq].strip()
            if cm != 0:
                value = float(line[eq+1:cm].strip())
                unit  = line[cm+1:].strip()
            else:
                value = float(line[eq+1:].strip())
                unit  = 'au'

            return key, value, unit

        else:
            return None, None, None

    #
    def parse_term_line(self, line, el):
        """
        parse a hamiltonian term line
        """
        if '|' in line and 'modes' not in line and 'KE' not in line:

            # parse the value of the coefficient
            cfstr = line[:line.index('|')].strip()
            coef  = cfstr.split('*')
            if len(coef) == 2:
                num = float(coef[0])
                key = coef[1]
            else:
                num = 1.
                key = coef[0]

            # in this case, electronic states
            # are explicitly given
            crds   = []
            states = []

            parsed = line
            ncrds = line.count('|')
            for i in range(ncrds):
                parsed = parsed[parsed.index('|')+1:]
                cdef   = parsed.strip().split()
                crdi   = int(cdef[0])-1

                # if current coordinate is electronic coord,
                # append to the states list
                if crdi == el:
                    sts = cdef[1].replace('S','').strip().split('&')
                    # states run from 0..ns-1
                    states = [int(st)-1 for st in sts]

                # else this is a vibrational coord -- determine
                # identity and the order
                else:
                    if '^' in cdef[1]:
                        cnt = int(cdef[1].strip().split('^')[1])
                    else:
                        cnt = 1
                    # coord indices run from 0..nq-1
                    crds += [crdi]*cnt

            return num, key, crds, states

        #
        else:
            return None, None, None, None

    #
    def gen_partition(self, k, n):
        """
        Generator for producing partitions on integer n

        Integer partitions of n into k parts, in colex order.
        The algorithm follows Knuth v4 fasc3 p38 in rough outline;
        Knuth credits it to Hindenburg, 1779.
        """

        # guard against special cases
        if k == 0:
            if n == 0:
                yield []
            return
        if k == 1:
            if n > 0:
                yield [n]
            return
        if n < k:
            return

        partition = [n - k + 1] + (k-1)*[1]
        while True:
            yield partition
            if partition[0] - 1 > partition[1]:
                partition[0] -= 1
                partition[1] += 1
                continue
            j = 2
            s = partition[0] + partition[1] - 1
            while j < k and partition[j] >= partition[0] - 1:
                s += partition[j]
                j += 1
            if j >= k:
                return
            partition[j] = x = partition[j] + 1
            j -= 1
            while j > 0:
                partition[j] = x
                s -= x
                j -= 1
            partition[0] = s

    #
    def partition(self, k, n):
        """
        use generator partition to generate a list of partitions
        for p_k(n)
        """

        plist = []
        for pki in self.gen_partition(k,n):
            plist.append(pki.copy())
        return plist

class ChemPotPy(Surface):
    """
    ChemPotPy surface evaluator
    """
    def __init__(self, molecule, surface_name, nstates, ref_geom,
                   e_units='eV', g_units='Angstrom'):
        super().__init__()
        self.molecule = molecule
        self.surface  = surface_name
        self.nstates  = nstates
        self.atms     = ref_geom.atms

        if e_units.lower() == 'ev':
            self.econv = constants.ev2au
        elif e_units.lower() == 'au':
            self.econv = 1.
        else:
            print('e_units='+str(e_units)+' not recognized.')
            os.abort()

        if g_units.lower() == 'angstrom':
            self.gconv = constants.ang2bohr
        elif g_units.lower() == 'bohr':
            self.gconv = 1.
        else:
            print('g_units='+str(g_units)+' not recognized.')
            os.abort()

        self.ref_geom = self._chempotpygeom(ref_geom.x / self.gconv)

        self.have_gradients = True
        self.have_coupling  = True

    #
    def evaluate(self, gms, states=None):
        """
        evaluate the potential at the passed geometries. Geometries
        are assumed to be a 2D numpy array
        """
        if states == None:
            states = [i for i in range(self.nstates)]
        elif max(states) > self.nstates:
            print('surface only defined for ' +str(self.nstates) +
                   ': Exiting...')
            os.abort()

        nst = len(states)
        ngm = gms.shape[0]
        energies = np.zeros((nst, ngm), dtype=float)

        for i in range(ngm):
            gm             = self._chempotpygeom(gms[i,:] / self.gconv)
            cppsurf        = chempotpy.p(self.molecule, self.surface, gm)
            energies[:,i] = cppsurf[[states]]

        energies *= self.econv

        return energies

    #
    def gradient(self, gms, states = None):
        """
        evaluate the gradients at the passed geometries. Geometries
        are assumed to be a 2D numpy array
        """

        if states == None:
            states = [i for i in range(self.nstates)]
        elif max(states) > self.nstates:
            print('surface only defined for ' +str(self.nstates) +
                   ': Exiting...')
            os.abort()

        nst = len(states)
        nat = len(self.atms)
        ngm = gms.shape[0]
        grads    = np.zeros((nst, ngm, 3*nat), dtype=float)

        for i in range(ngm):
            gm           = self._chempotpygeom(gms[i,:] / self.gconv)
            cppsurf      = chempotpy.pg(self.molecule, self.surface, gm)
            grads[:,i,:] = np.reshape(cppsurf[1][[states]], (nst, 3*nat))

        grads    *= (self.econv / self.gconv)

        return grads

    #
    def hessian(self, gms, states = None):
        """
        evaluate the hessian on states 'states'. If states=None, return
        hessian for all defined states
        """
        if states == None:
            states = [i for i in range(self.nstates)]
        elif max(states) > self.nstates:
            print('surface only defined for ' +str(self.nstates) +
                   ': Exiting...')
            os.abort()

        nst = len(states)
        nat = len(self.atms)
        ngm = gms.shape[0]
        hessian  = np.zeros((nst, ngm, 3*nat, 3*nat), dtype=float)
        for i in range(ngm):
            for j in range(natm):
                gm  = self._chempotpygeom(gms[i,:] / self.gconv)
                cppsurf = chempotpy.pg(self.molecule, self.surface, gm)
                hessian[:,i,:,:] = np.reshape(cppsurf[1][[states]], 
                                                   (nst, 3*nat, 3*nat))

        return hessian

    #
    def coupling(self, gms, pairs = None):
        """
        evaluate the NACs at the passed geometries. Geometries
        are assumed to be a 2D numpy array
        """

        npair = len(pairs)
        nat   = len(self.atms)
        ngm   = gms.shape[0]
        nacs  = np.zeros((npair, ngm, 3*nat), dtype=float)

        for i in range(ngm):
            gm        = self._chempotpygeom(gms[i,:] / self.gconv)
            cppsurf   = chempotpy.pgd(self.molecule, self.surface, gm)
            for j in range(npair):
                nacs[j,i,:] = cppsurf[2][pairs[j][0], pairs[j][1],:].ravel()

        nacs    *= (self.econv / self.gconv)

        return nacs

    #
    def _chempotpygeom(self, gm):
        """
        convert a numpy array geometry to chempotpy format
        """
        cgm = []
        for i in range(len(self.atms)):
            xyz = gm[3*i:3*i+3].tolist()
            cgm.append([self.atms[i]] + xyz)
        return cgm
