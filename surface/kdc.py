"""
KDC vibronic-coupling surface evaluator (Kdc) and its Taylor Hamiltonian
container (Kdc_ham).
"""
import os
import numpy as np
import itertools
import timer as timer
from .base import Surface

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
    @timer.timed
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

