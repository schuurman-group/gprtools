#!/usr/bin/env python
# system import
import os as os
import sys as sys
from os.path import abspath, join, dirname, basename
# third party import
import numpy as np
# local import
project_dir = abspath(join(dirname(__file__),'//home/bsonghao/gprtools/'))
sys.path.insert(0, project_dir)
import constants as constants
import molecule as molecule
import descriptor as descriptor
import surrogate as surrogate
import surface as surface
import sample as sample
import intc as intc


class construct_GPR(object):
    """perform trajectory based adaptive sampling to construct GPR"""
    def __init__(self, initial_geom, initial_momentum):
        """
        set initial geometry and initial momentum as class instances
        """
        self.initial_geom = initial_geom
        self.initial_momentum = initial_momentum

        mass = [[self.initial_geom.masses[i]]*3
                                for i in range(len(self.initial_geom.masses))]

        self.m = np.array(sum(mass, []), dtype=float)

        self.num_atom = int(len(self.initial_geom.x)/3)

        # ChemPotPy data base
        self.potential = surface.ChemPotPy('C6H6O', 'PHOH_APRP', 3,
                               self.initial_geom, e_units='eV', g_units='Angstrom')

        # initialize intc library
        intdef = initial_geom.qdef
        self.cart2int = intc.Cart2int(intdef)

        # store the types of the internal coordinate into a python dictoionary
        self.types = {}
        for i in range(3*self.num_atom-6):
            for j in range(intdef.n_prim(i)):
                self.types[i] =  intdef.q_types(i)[j]

    def intc2cart(self,  current_geom_inc):
        """"convert intc coordinate to cartesian coordinate giving a reference geometry"""
        ref_geom = self.initial_geom.x.copy()
        ref_geom_inc = self.initial_geom.q.copy()

        dq = np.zeros_like(ref_geom_inc)
        dq += current_geom_inc
        dq -= ref_geom_inc
        # print("dq:\n{:}".format(dq.shape))
        dx = self.cart2int.dint2cart(ref_geom, dq.reshape(-1,1).transpose())

        return ref_geom + dx

    def convert2intc(self, input_geom,debug=False):
        """convert input geometry from Cartisian coordinate into internal coordinate"""
        # flatten the input geometry and convert into Angstrom unit
        input_geom_convert = input_geom.ravel()

        # call subroutine to convert 3N Cartisian coordinate into 3N-6 internal coordinate
        output_geom  = self.cart2int.cart2intc(input_geom_convert)

        # convert unit (length to Bohr, angle to degree)
        output_geom = [x if self.types[i] == 'stre' else x * constants.rad2deg for i, x in enumerate(output_geom)]

        if debug:
            # test if the geometry conversion is correct
            ref_geom_convert = self.equil_geom.ravel()
            ref_geom_intc = self.cart2int.cart2intc(ref_geom_convert)
            dq = output_geom.copy() - ref_geom_intc.copy()
            print("dq:\n{:}".format(dq))
            dx = self.cart2int.dint2cart(ref_geom_convert, dq.reshape(-1,1).transpose()).ravel()
            print("dx:\n{:}".format(dx))
            print(ref_geom_convert+dx-input_geom_convert)
            output_geom_test = self.cart2int.cart2intc(ref_geom_convert+dx)
            print(abs(output_geom_test-output_geom).max())
            assert np.allclose(output_geom_test, output_geom)
            # assert np.allclose(ref_geom_convert+dx, input_geom_convert)

        return output_geom

    def convert2intp(self, input_geom, input_momentum, debug=False):
        """convert momentum vector into internal coordinate basis"""
        output_momentum = self.cart2int.cart2intp(input_geom, input_momentum)

        if debug:# perform checks
            print("momentum in internal coordinate basis:{:}".format(output_momentum))
            print("Shape of momentum: {:}".format(output_momentum.shape))
            # calculate normal mode vector along stretching of OH bond
            # Hessian, freq, vec = self.cal_Hessian(self.equil_geom)
            # mode_vec = vec[:,38].copy()
            # print("Frequencies:{:}".format(freq * self.Hartree2wavenumber))
            vec = self.equil_geom[12,:].copy() - self.equil_geom[6,:].copy()
            vec /= np.sqrt(sum(vec**2))
            mode_vec = np.zeros((self.num_atom, 3))
            mode_vec[12,:] += vec
            test_momentum =  100 * mode_vec.ravel()
            test_momentum_inc = self.cart2int.cart2intp(self.equil_geom.ravel(), test_momentum)
            print("test momentum vector:\n{:}".format(test_momentum))
            print("transformed test momentum vector:\n{:}".format(test_momentum_inc))

        return output_momentum

    def cal_bound(self, input_geom, input_momentum, time):
        """
        calculate bound of LHS sampling
        """
        # convert time to atomic unit
        time *= constants.fs2au

        num_mode = int(3*self.num_atom - 6)

        bound = np.zeros((num_mode, 2))

        # convert geometry and momentum into internal coordinate
        # print(input_geom.shape)
        geom_inc = self.convert2intc(input_geom)
        momentum_inc = self.convert2intp(input_geom, input_momentum / self.m)

        # set upper bound
        upper_bound = momentum_inc * time
        bound[:,0] += upper_bound

        # set lower bound
        bound[:,1] -= 0.1 * upper_bound

        return bound

    def update_sample(self, geometry, momentum, num_sample, time):
        """sampling along the GPR trajectory"""
        bnds = self.cal_bound(geometry.x, momentum, time)
        # rnd  = 1024
        lhs_sampler = sample.LHS(geometry, bnds, crd='intc')
        dgeoms = lhs_sampler.sample(num_sample, cartesian=True)

        # evaluate the surface at the sampled geometries
        labels = self.potential.evaluate(dgeoms, states=[0])

        return dgeoms, labels

    def angle_chk(self, x, p, state):
        """
        compare the angle between the surrogate and surface.
        If angle is great than 5 degrees, return false
        """
        gm = np.array([x], dtype=float)
        grad_surr = self.surr.gradient(gm)[0,0]
        grad_surf = self.potential.gradient(gm, states=[state])[0,0]

        angle = abs(np.dot(grad_surr, grad_surf)) * constants.rad2deg / np.linalg.norm(grad_surr) / np.linalg.norm(grad_surf)

        print(f'angle:{angle}')

        return  angle

    def update_GPR(self, t_term, num_sample, threshold, debug=True):
        """
        method to construct GPR that calling inividual subroutines
        """
        # initialize the descriptor
        soap = descriptor.Soap(self.initial_geom, 3., 6, 8, 0.1)
        # initialize gpr surrogate
        self.surr = surrogate.Adiabat(soap, 'RBF', 50, hparam=[0.1, 0.1])
        # perform initial LHS sampling
        dgeoms, labels = self.update_sample(self.initial_geom, self.initial_momentum, num_sample, 10)
        # generate initial  surrogate using the LHS sampled training data
        hyperpara = self.surr.create([dgeoms, labels])
        self.surr.save("phenol_GPR_surrogate")

        t_current = 0
        iteration = 0

        t_term_au = t_term * constants.fs2au

        geometry = self.initial_geom
        momentum = self.initial_momentum.copy()


        # perform update of GPR surrogate
        while t_current < t_term_au:
            if debug:
                print(f'At time {t_current} a.u.')
                print(f'geomertry:\n{geometry.x}')
                print(f'momentum:\n{momentum}')
                print(f'hyper parameters:{hyperpara}')

            # print(gshape)
            # os._exit(0)
            surr_traj = molecule.Trajectory(geometry,
                                            1, surface=self.surr)
            res_surr = surr_traj.propagate(x0=geometry.x, p0=momentum, s0=0, t0=t_current, dt=10,
                                                chk_func=self.angle_chk,
                                                chk_thresh=threshold)

            # update geometry, momentum and time
            geometry.update_geom(res_surr.x[-1,:])
            momentum = res_surr.p[-1,:]
            t_current = res_surr.t[-1]

            # resample along the trajectory
            dgeoms_new, labels_new = self.update_sample(geometry, momentum, num_sample, 10)

            dgeoms = np.concatenate((dgeoms, dgeoms_new))
            labels = np.concatenate((labels, labels_new))

            print(dgeoms.shape)
            print(labels.shape)
            # update surrogate
            self.surr = surrogate.Adiabat(soap, 'RBF', 50, hparam=hyperpara)
            hyperpara = self.surr.create([dgeoms, labels])
            self.surr.save("phenol_GPR_surrogate")



            iteration += 1
            if iteration > 1000:
                break

        return
