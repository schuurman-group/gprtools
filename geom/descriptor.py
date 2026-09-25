"""
The Surface ABC
"""
import os
import numpy as np
from abc import ABC, abstractmethod
from ase import Atoms
from dscribe.descriptors import SOAP
import constants
import timer as timer
from . import intc as internal_coordinates

class Descriptor(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def descriptor_gradient(self):
        pass

    def generate_with_gradient(self, gms):
        """Return descriptor values and Cartesian Jacobians together.

        Descriptors with a backend capable of producing both in one operation
        override this method.  The default preserves the public conventions by
        calling the existing methods independently.
        """
        return self.generate(gms), self.descriptor_gradient(gms)

#
class InversePairDistance(Descriptor):
    """Inverse distances for every unique atom pair in a fixed molecule."""

    _distance_tolerance = 1.e-12

    def __init__(self, ref_gm):
        """Build an inverse pair-distance descriptor for ``ref_gm``.

        Features are ordered lexicographically by atom index, ``i < j``, and
        contain the raw values ``1/r_ij``. Input coordinates are in bohr, so
        descriptor values have units bohr**-1.
        """
        super().__init__()
        self.atoms = ref_gm.atms
        self.n_atoms = len(self.atoms)
        self.pairs = [(i, j) for i in range(self.n_atoms)
                              for j in range(i + 1, self.n_atoms)]
        self.n_features = len(self.pairs)
        self._pair_i = np.array([pair[0] for pair in self.pairs], dtype=int)
        self._pair_j = np.array([pair[1] for pair in self.pairs], dtype=int)

    def _prepare_geometries(self, gms):
        """Return a validated geometry batch and whether the input was 1D."""
        try:
            geometries = np.asarray(gms, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError('Geometries must contain numeric coordinates.') \
                from exc

        if geometries.ndim == 1:
            single_geometry = True
            geometries = geometries.reshape(1, -1)
        elif geometries.ndim == 2:
            single_geometry = False
        else:
            raise ValueError(
                'Geometries must be a flattened 1D geometry or a 2D batch; '
                f'got an array with {geometries.ndim} dimensions.')

        expected_coordinates = 3*self.n_atoms
        if geometries.shape[1] != expected_coordinates:
            raise ValueError(
                f'Expected {expected_coordinates} Cartesian coordinates for '
                f'{self.n_atoms} atoms, got {geometries.shape[1]}.')
        if not np.all(np.isfinite(geometries)):
            raise ValueError('Geometry coordinates must all be finite.')

        return geometries, single_geometry

    def _pair_displacements(self, geometries):
        """Return pair vectors and distances after collision validation."""
        coordinates = geometries.reshape(-1, self.n_atoms, 3)
        displacements = (coordinates[:, self._pair_i, :]
                         - coordinates[:, self._pair_j, :])
        distances = np.linalg.norm(displacements, axis=2)

        too_close = np.argwhere(distances < self._distance_tolerance)
        if too_close.size:
            geometry_index, pair_index = too_close[0]
            atom_i, atom_j = self.pairs[pair_index]
            distance = distances[geometry_index, pair_index]
            raise ValueError(
                f'Atom separation for pair ({atom_i}, {atom_j}) in geometry '
                f'{geometry_index} is {distance:.3e} bohr, below the '
                f'{self._distance_tolerance:.1e} bohr tolerance.')

        return displacements, distances

    @timer.timed
    def generate(self, gms):
        """Generate raw inverse pair distances for one geometry or a batch."""
        geometries, single_geometry = self._prepare_geometries(gms)
        _, distances = self._pair_displacements(geometries)
        descriptors = 1./distances

        if single_geometry:
            return descriptors[0]
        return descriptors

    @timer.timed
    def descriptor_gradient(self, gms, delta=None):
        """Differentiate inverse pair distances with respect to Cartesians.

        The returned axes are ``(geometry, coordinate, feature)``, matching
        :meth:`Soap.descriptor_gradient`. If ``delta`` is supplied, use a
        central finite difference instead of the default analytic derivative.
        """
        geometries, _ = self._prepare_geometries(gms)
        ng, nc = geometries.shape
        gradient = np.zeros((ng, nc, self.n_features), dtype=float)

        if delta is not None:
            if not np.isscalar(delta):
                raise ValueError('Finite-difference delta must be a scalar.')
            try:
                step = float(delta)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    'Finite-difference delta must be a finite positive value.') \
                    from exc
            if not np.isfinite(step) or step <= 0.:
                raise ValueError(
                    'Finite-difference delta must be a finite positive value.')

            coordinate_displacements = np.eye(nc, dtype=float)*step
            for geometry_index in range(ng):
                origin = geometries[geometry_index]
                plus = self.generate(origin + coordinate_displacements)
                minus = self.generate(origin - coordinate_displacements)
                gradient[geometry_index] = (plus - minus)/(2.*step)
            return gradient

        displacements, distances = self._pair_displacements(geometries)
        pair_derivatives = -displacements/(distances[..., None]**3)
        for feature_index, (atom_i, atom_j) in enumerate(self.pairs):
            derivative = pair_derivatives[:, feature_index, :]
            gradient[:, 3*atom_i:3*atom_i + 3, feature_index] = derivative
            gradient[:, 3*atom_j:3*atom_j + 3, feature_index] = -derivative

        return gradient


class RedundantInternals(Descriptor):
    """Bonded-graph redundant internal coordinates for a fixed molecule.

    The covalent graph is detected once from ``ref_gm`` and is then held
    fixed. Features are ordered as inverse bond distances, bond angles,
    dihedrals, and pyramidalisation (signed out-of-plane) angles. A dihedral
    contributes ``sin(phi), cos(phi)`` by default so that the descriptor is
    continuous through the angular branch cut.

    Cartesian inputs and inverse bond distances use bohr; all angular
    coordinates use radians. No normalization or standardization is applied.
    """

    _distance_tolerance = 1.e-12
    _angular_tolerance = 1.e-12

    def __init__(self, ref_gm, bond_scale=1.3,
                 linear_tolerance_degrees=0.0,
                 periodic_dihedrals=True):
        """Construct the fixed bonded graph and redundant coordinate list.

        Parameters
        ----------
        ref_gm : Geometry
            Reference geometry. ``ref_gm.x`` must contain flattened Cartesian
            coordinates in bohr and ``ref_gm.atms`` the element labels.
        bond_scale : float
            Covalent-radius multiplier used to detect reference bonds.
        linear_tolerance_degrees : float
            Reference bond angles this close to 0 or 180 degrees are omitted
            because their scalar-angle Cartesian derivative is singular. The
            default zero requests every nonsingular graph angle and torsion.
        periodic_dihedrals : bool
            If true (default), represent every graph dihedral by sine and
            cosine features. If false, store the raw angle in radians.
        """
        super().__init__()
        self.atoms = tuple(ref_gm.atms)
        self.n_atoms = len(self.atoms)
        if self.n_atoms < 2:
            raise ValueError('RedundantInternals requires at least two atoms.')

        try:
            reference = np.asarray(ref_gm.x, dtype=float)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                'ref_gm must provide numeric flattened coordinates in ref_gm.x.') \
                from exc
        if reference.ndim != 1 or reference.size != 3*self.n_atoms:
            raise ValueError(
                f'Reference geometry must contain {3*self.n_atoms} flattened '
                'Cartesian coordinates.')
        if not np.all(np.isfinite(reference)):
            raise ValueError('Reference geometry coordinates must all be finite.')
        try:
            bond_scale = float(bond_scale)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                'bond_scale must be a finite positive scalar.') from exc
        if not np.isfinite(bond_scale) or bond_scale <= 0.:
            raise ValueError('bond_scale must be a finite positive scalar.')
        try:
            linear_tolerance_degrees = float(linear_tolerance_degrees)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                'linear_tolerance_degrees must be finite and in [0, 90).') \
                from exc
        if not np.isfinite(linear_tolerance_degrees) \
                or not 0. <= linear_tolerance_degrees < 90.:
            raise ValueError(
                'linear_tolerance_degrees must be finite and in [0, 90).')
        if not isinstance(periodic_dihedrals, (bool, np.bool_)):
            raise ValueError('periodic_dihedrals must be boolean.')

        self.reference_geometry = reference.copy()
        self.bond_scale = bond_scale
        self.linear_tolerance_degrees = linear_tolerance_degrees
        self.periodic_dihedrals = bool(periodic_dihedrals)

        definition = internal_coordinates.Intdef()
        self.bonds = [tuple(pair) for pair in definition.generate_redundant(
            self.reference_geometry,
            self.atoms,
            coords='internals',
            scale=self.bond_scale,
            lin_tol=self.linear_tolerance_degrees,
        )]
        if not self.bonds:
            raise ValueError(
                'The reference geometry produced an empty bonded graph; '
                'adjust bond_scale or check its coordinates and units.')

        self._primitive_types = []
        self._primitive_atoms = []
        for coordinate_index in range(definition.n_q()):
            # generate_redundant creates one primitive per coordinate.
            self._primitive_types.append(
                definition.q_types(coordinate_index)[0])
            self._primitive_atoms.append(tuple(
                definition.q_atms(coordinate_index)[0]))

        self.angles = []
        self.dihedrals = []
        self.pyramidalizations = []
        for typ, atoms in zip(self._primitive_types, self._primitive_atoms):
            if typ == 'bend':
                # intc stores (outer1, outer2, centre); expose the usual
                # (outer1, centre, outer2) convention publicly.
                self.angles.append((atoms[0], atoms[2], atoms[1]))
            elif typ == 'tors':
                self.dihedrals.append(atoms)
            elif typ == 'out':
                # Public convention: (apex, centre, plane1, plane2).
                self.pyramidalizations.append(
                    (atoms[0], atoms[3], atoms[1], atoms[2]))

        self.feature_types = []
        self.feature_atoms = []
        self._raw_dihedral_feature_indices = []
        for bond in self.bonds:
            self.feature_types.append('inverse_distance')
            self.feature_atoms.append(bond)
        for angle in self.angles:
            self.feature_types.append('angle')
            self.feature_atoms.append(angle)
        for dihedral in self.dihedrals:
            if self.periodic_dihedrals:
                self.feature_types.extend(('dihedral_sin', 'dihedral_cos'))
                self.feature_atoms.extend((dihedral, dihedral))
            else:
                self._raw_dihedral_feature_indices.append(
                    len(self.feature_types))
                self.feature_types.append('dihedral')
                self.feature_atoms.append(dihedral)
        for pyramidalization in self.pyramidalizations:
            self.feature_types.append('pyramidalization')
            self.feature_atoms.append(pyramidalization)

        self.n_internal_coordinates = (
            len(self.bonds) + len(self.angles) + len(self.dihedrals)
            + len(self.pyramidalizations))
        self.n_features = len(self.feature_types)

        # Dense index tables allow every primitive of a given type, and every
        # geometry in a batch, to be evaluated in one NumPy operation. Empty
        # coordinate classes retain their two-dimensional shape so downstream
        # indexing needs no special cases beyond a size check.
        self._bond_atoms = np.asarray(self.bonds, dtype=int).reshape(-1, 2)
        self._angle_atoms = np.asarray(self.angles, dtype=int).reshape(-1, 3)
        self._dihedral_atoms = np.asarray(
            self.dihedrals, dtype=int).reshape(-1, 4)
        self._pyramidalization_atoms = np.asarray(
            self.pyramidalizations, dtype=int).reshape(-1, 4)

        self._bond_feature_start = 0
        self._angle_feature_start = len(self.bonds)
        self._dihedral_feature_start = (
            self._angle_feature_start + len(self.angles))
        dihedral_width = 2 if self.periodic_dihedrals else 1
        self._pyramidalization_feature_start = (
            self._dihedral_feature_start
            + dihedral_width*len(self.dihedrals))

        # Fail at construction rather than waiting for the first call if the
        # reference itself contains a singular generated coordinate.
        self._internal_data(
            self.reference_geometry.reshape(1, self.n_atoms, 3))

    def _prepare_geometries(self, gms):
        try:
            geometries = np.asarray(gms, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError('Geometries must contain numeric coordinates.') \
                from exc
        if geometries.ndim == 1:
            single_geometry = True
            geometries = geometries.reshape(1, -1)
        elif geometries.ndim == 2:
            single_geometry = False
        else:
            raise ValueError(
                'Geometries must be a flattened 1D geometry or a 2D batch; '
                f'got an array with {geometries.ndim} dimensions.')
        expected = 3*self.n_atoms
        if geometries.shape[1] != expected:
            raise ValueError(
                f'Expected {expected} Cartesian coordinates for '
                f'{self.n_atoms} atoms, got {geometries.shape[1]}.')
        if not np.all(np.isfinite(geometries)):
            raise ValueError('Geometry coordinates must all be finite.')
        return geometries, single_geometry

    def _singularity_error(self, geometry_index, kind, atoms, detail):
        raise ValueError(
            f'Cannot evaluate {kind} coordinate {tuple(atoms)} in geometry '
            f'{geometry_index}: {detail}.')

    def _raise_first(self, mask, kind, atoms, detail):
        """Raise for the first invalid batch/primitive entry.

        Error handling may inspect individual entries; the normal evaluation
        path remains fully vectorized.
        """
        locations = np.argwhere(mask)
        if locations.size == 0:
            return
        geometry_index, coordinate_index = locations[0]
        message = (detail(geometry_index, coordinate_index)
                   if callable(detail) else detail)
        self._singularity_error(
            int(geometry_index), kind, atoms[coordinate_index], message)

    @staticmethod
    def _norm(vectors):
        return np.sqrt(np.einsum('...i,...i->...', vectors, vectors))

    def _internal_data(self, xyz):
        """Evaluate reusable primitive data for a complete geometry batch."""
        data = {}

        atom_i = self._bond_atoms[:, 0]
        atom_j = self._bond_atoms[:, 1]
        bond_vectors = xyz[:, atom_i, :] - xyz[:, atom_j, :]
        bond_lengths = self._norm(bond_vectors)
        self._raise_first(
            bond_lengths < self._distance_tolerance,
            'inverse-distance', self._bond_atoms,
            lambda g, q: (
                f'separation {bond_lengths[g, q]:.3e} bohr is below the '
                f'{self._distance_tolerance:.1e} bohr tolerance'))
        data['bond_vectors'] = bond_vectors
        data['bond_lengths'] = bond_lengths

        if self._angle_atoms.shape[0]:
            atom_a, centre, atom_c = self._angle_atoms.T
            u = xyz[:, atom_a, :] - xyz[:, centre, :]
            v = xyz[:, atom_c, :] - xyz[:, centre, :]
            norm_u = self._norm(u)
            norm_v = self._norm(v)
            unit_u = u/norm_u[..., None]
            unit_v = v/norm_v[..., None]
            cosine = np.einsum('...i,...i->...', unit_u, unit_v)
            sine = self._norm(np.cross(unit_u, unit_v))
            self._raise_first(
                sine < self._angular_tolerance,
                'angle', self._angle_atoms,
                'the angle is collinear and its derivative is singular')
            data['angle_norm_u'] = norm_u
            data['angle_norm_v'] = norm_v
            data['angle_unit_u'] = unit_u
            data['angle_unit_v'] = unit_v
            data['angle_cosine'] = cosine
            data['angle_sine'] = sine

        if self._dihedral_atoms.shape[0]:
            atom_a, atom_b, atom_c, atom_d = self._dihedral_atoms.T
            u = xyz[:, atom_a, :] - xyz[:, atom_b, :]
            v = xyz[:, atom_c, :] - xyz[:, atom_b, :]
            w = xyz[:, atom_c, :] - xyz[:, atom_d, :]
            norm_u = self._norm(u)
            norm_v = self._norm(v)
            norm_w = self._norm(w)
            unit_u = u/norm_u[..., None]
            unit_v = v/norm_v[..., None]
            unit_w = w/norm_w[..., None]
            normal_left_raw = np.cross(unit_u, unit_v)
            normal_right_raw = np.cross(unit_w, unit_v)
            sine_left = self._norm(normal_left_raw)
            sine_right = self._norm(normal_right_raw)
            self._raise_first(
                np.minimum(sine_left, sine_right)
                < self._angular_tolerance,
                'dihedral', self._dihedral_atoms,
                'a defining bond angle is collinear')
            normal_left = normal_left_raw/sine_left[..., None]
            normal_right = normal_right_raw/sine_right[..., None]

            # This is the vectorized equivalent of Cart2int.qtors, retained
            # exactly so existing dihedral signs and branch placement do not
            # change. The default sin/cos representation removes that branch
            # cut from the descriptor itself.
            normal_cosine = np.einsum(
                '...i,...i->...', normal_left, normal_right)
            tau = np.arccos(np.clip(-normal_cosine, -1., 1.))
            orientation = np.einsum(
                '...i,...i->...',
                np.cross(normal_left, normal_right), unit_v)
            tau = np.where(orientation < 0., -tau, tau)
            tau = np.where(tau > 0.5*np.pi, tau - 2.*np.pi, tau)
            tau = np.where(tau <= -2.*np.pi, tau + 2.*np.pi, tau)

            data['dihedral_norm_u'] = norm_u
            data['dihedral_norm_v'] = norm_v
            data['dihedral_norm_w'] = norm_w
            data['dihedral_unit_u'] = unit_u
            data['dihedral_unit_v'] = unit_v
            data['dihedral_unit_w'] = unit_w
            data['dihedral_sine_left'] = sine_left
            data['dihedral_sine_right'] = sine_right
            data['dihedral_normal_left'] = normal_left
            data['dihedral_normal_right'] = normal_right
            data['dihedral_phi'] = -tau

        if self._pyramidalization_atoms.shape[0]:
            apex, centre, plane_1, plane_2 = \
                self._pyramidalization_atoms.T
            u = xyz[:, apex, :] - xyz[:, centre, :]
            v = xyz[:, plane_1, :] - xyz[:, centre, :]
            w = xyz[:, plane_2, :] - xyz[:, centre, :]
            norm_u = self._norm(u)
            norm_v = self._norm(v)
            norm_w = self._norm(w)
            unit_u = u/norm_u[..., None]
            unit_v = v/norm_v[..., None]
            unit_w = w/norm_w[..., None]
            normal_raw = np.cross(unit_v, unit_w)
            plane_sine = self._norm(normal_raw)
            self._raise_first(
                plane_sine < self._angular_tolerance,
                'pyramidalization', self._pyramidalization_atoms,
                'the two plane-defining bonds are collinear')
            normal = normal_raw/plane_sine[..., None]
            sine = np.einsum('...i,...i->...', unit_u, normal)
            sine = np.clip(sine, -1., 1.)
            cosine = np.sqrt(np.maximum(0., 1. - sine**2))
            self._raise_first(
                cosine < self._angular_tolerance,
                'pyramidalization', self._pyramidalization_atoms,
                'the out-of-plane angle is at a singular +/-pi/2 limit')
            data['pyramid_norm_u'] = norm_u
            data['pyramid_norm_v'] = norm_v
            data['pyramid_norm_w'] = norm_w
            data['pyramid_unit_u'] = unit_u
            data['pyramid_unit_v'] = unit_v
            data['pyramid_unit_w'] = unit_w
            data['pyramid_plane_sine'] = plane_sine
            data['pyramid_normal'] = normal
            data['pyramid_sine'] = sine
            data['pyramid_cosine'] = cosine

        return data

    def _coordinate_values(self, xyz):
        data = self._internal_data(xyz)
        values = np.empty((xyz.shape[0], self.n_features), dtype=float)

        start = self._bond_feature_start
        stop = self._angle_feature_start
        values[:, start:stop] = 1./data['bond_lengths']

        if self._angle_atoms.shape[0]:
            start = self._angle_feature_start
            stop = self._dihedral_feature_start
            values[:, start:stop] = np.arccos(
                np.clip(data['angle_cosine'], -1., 1.))

        if self._dihedral_atoms.shape[0]:
            start = self._dihedral_feature_start
            stop = self._pyramidalization_feature_start
            phi = data['dihedral_phi']
            if self.periodic_dihedrals:
                values[:, start:stop] = np.stack(
                    (np.sin(phi), np.cos(phi)), axis=-1).reshape(
                        xyz.shape[0], -1)
            else:
                values[:, start:stop] = phi

        if self._pyramidalization_atoms.shape[0]:
            start = self._pyramidalization_feature_start
            values[:, start:] = np.arcsin(data['pyramid_sine'])

        return values

    @staticmethod
    def _set_atom_derivatives(gradient, atoms, features, derivatives):
        """Assign ``(batch, primitive, xyz)`` values without Python loops."""
        cartesian = (3*atoms[:, None] + np.arange(3)).reshape(-1)
        feature = np.repeat(features, 3)
        gradient[:, cartesian, feature] = derivatives.reshape(
            gradient.shape[0], -1)

    @timer.timed
    def generate(self, gms):
        """Generate redundant bonded internals for one geometry or a batch."""
        geometries, single_geometry = self._prepare_geometries(gms)
        xyz = geometries.reshape(-1, self.n_atoms, 3)
        descriptors = self._coordinate_values(xyz)
        return descriptors[0] if single_geometry else descriptors

    @timer.timed
    def descriptor_gradient(self, gms, delta=None):
        """Return axes ``(geometry, Cartesian coordinate, feature)``."""
        geometries, _ = self._prepare_geometries(gms)
        ng, nc = geometries.shape
        gradient = np.zeros((ng, nc, self.n_features), dtype=float)

        if delta is not None:
            if not np.isscalar(delta):
                raise ValueError('Finite-difference delta must be a scalar.')
            try:
                step = float(delta)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    'Finite-difference delta must be a finite positive value.') \
                    from exc
            if not np.isfinite(step) or step <= 0.:
                raise ValueError(
                    'Finite-difference delta must be a finite positive value.')
            displacements = np.eye(nc, dtype=float)*step
            for geometry_index, origin in enumerate(geometries):
                plus = self.generate(origin + displacements)
                minus = self.generate(origin - displacements)
                difference = plus - minus
                for feature_index in self._raw_dihedral_feature_indices:
                    difference[:, feature_index] = (
                        difference[:, feature_index] + np.pi) % (2.*np.pi) \
                        - np.pi
                gradient[geometry_index] = difference/(2.*step)
            return gradient

        xyz = geometries.reshape(-1, self.n_atoms, 3)
        data = self._internal_data(xyz)

        bond_features = np.arange(
            self._bond_feature_start, self._angle_feature_start)
        bond_derivative = (-data['bond_vectors']
                           / data['bond_lengths'][..., None]**3)
        self._set_atom_derivatives(
            gradient, self._bond_atoms[:, 0], bond_features,
            bond_derivative)
        self._set_atom_derivatives(
            gradient, self._bond_atoms[:, 1], bond_features,
            -bond_derivative)

        if self._angle_atoms.shape[0]:
            cosine = data['angle_cosine'][..., None]
            sine = data['angle_sine'][..., None]
            derivative_a = (
                cosine*data['angle_unit_u'] - data['angle_unit_v']) \
                / (sine*data['angle_norm_u'][..., None])
            derivative_c = (
                cosine*data['angle_unit_v'] - data['angle_unit_u']) \
                / (sine*data['angle_norm_v'][..., None])
            derivative_centre = -derivative_a - derivative_c
            angle_features = np.arange(
                self._angle_feature_start, self._dihedral_feature_start)
            self._set_atom_derivatives(
                gradient, self._angle_atoms[:, 0], angle_features,
                derivative_a)
            self._set_atom_derivatives(
                gradient, self._angle_atoms[:, 2], angle_features,
                derivative_c)
            self._set_atom_derivatives(
                gradient, self._angle_atoms[:, 1], angle_features,
                derivative_centre)

        if self._dihedral_atoms.shape[0]:
            norm_u = data['dihedral_norm_u']
            norm_v = data['dihedral_norm_v']
            norm_w = data['dihedral_norm_w']
            sine_left = data['dihedral_sine_left']
            sine_right = data['dihedral_sine_right']
            cosine_left = np.einsum(
                '...i,...i->...', data['dihedral_unit_u'],
                data['dihedral_unit_v'])
            cosine_right = np.einsum(
                '...i,...i->...', data['dihedral_unit_v'],
                data['dihedral_unit_w'])
            derivative_a = (data['dihedral_normal_left']
                            / (norm_u*sine_left)[..., None])
            derivative_d = (data['dihedral_normal_right']
                            / (norm_w*sine_right)[..., None])
            derivative_b = (
                (norm_u*cosine_left/norm_v - 1.)[..., None]*derivative_a
                - (norm_w*cosine_right/norm_v)[..., None]*derivative_d)
            derivative_c = (
                -derivative_a - derivative_b - derivative_d)

            base = self._dihedral_feature_start
            if self.periodic_dihedrals:
                phi = data['dihedral_phi']
                sin_features = base + 2*np.arange(len(self.dihedrals))
                cos_features = sin_features + 1
                sin_factor = np.cos(phi)[..., None]
                cos_factor = -np.sin(phi)[..., None]
                for atoms, derivative in (
                        (self._dihedral_atoms[:, 0], derivative_a),
                        (self._dihedral_atoms[:, 1], derivative_b),
                        (self._dihedral_atoms[:, 2], derivative_c),
                        (self._dihedral_atoms[:, 3], derivative_d)):
                    self._set_atom_derivatives(
                        gradient, atoms, sin_features,
                        sin_factor*derivative)
                    self._set_atom_derivatives(
                        gradient, atoms, cos_features,
                        cos_factor*derivative)
            else:
                features = base + np.arange(len(self.dihedrals))
                for atoms, derivative in (
                        (self._dihedral_atoms[:, 0], derivative_a),
                        (self._dihedral_atoms[:, 1], derivative_b),
                        (self._dihedral_atoms[:, 2], derivative_c),
                        (self._dihedral_atoms[:, 3], derivative_d)):
                    self._set_atom_derivatives(
                        gradient, atoms, features, derivative)

        if self._pyramidalization_atoms.shape[0]:
            unit_u = data['pyramid_unit_u']
            unit_v = data['pyramid_unit_v']
            unit_w = data['pyramid_unit_w']
            normal = data['pyramid_normal']
            cosine = data['pyramid_cosine']
            plane_sine = data['pyramid_plane_sine']
            cosine_vw = np.einsum('...i,...i->...', unit_v, unit_w)
            cosine_wu = np.einsum('...i,...i->...', unit_w, unit_u)
            cosine_vu = np.einsum('...i,...i->...', unit_v, unit_u)
            denominator = cosine*plane_sine**2
            scale_v = ((cosine_vw*cosine_wu - cosine_vu)
                       / (data['pyramid_norm_v']*denominator))
            scale_w = ((cosine_vw*cosine_vu - cosine_wu)
                       / (data['pyramid_norm_w']*denominator))
            derivative_v = normal*scale_v[..., None]
            derivative_w = normal*scale_w[..., None]
            x = np.cross(normal, unit_u)
            x /= self._norm(x)[..., None]
            y = np.cross(unit_u, x)
            y /= self._norm(y)[..., None]
            derivative_u = y/data['pyramid_norm_u'][..., None]
            derivative_centre = -derivative_u - derivative_v - derivative_w
            features = self._pyramidalization_feature_start \
                + np.arange(len(self.pyramidalizations))
            for atoms, derivative in (
                    (self._pyramidalization_atoms[:, 0], derivative_u),
                    (self._pyramidalization_atoms[:, 2], derivative_v),
                    (self._pyramidalization_atoms[:, 3], derivative_w),
                    (self._pyramidalization_atoms[:, 1], derivative_centre)):
                self._set_atom_derivatives(
                    gradient, atoms, features, derivative)
        return gradient


class Soap(Descriptor):
    """
    GRaCI surface evaluator
    """
    def __init__(self, ref_gm, r_max, n_max, l_max, sigma):
        """
        Build a SOAP descriptor generator for the molecule given by ref_gm.

        SOAP encodes each atom's local environment as an expansion of the
        smoothed neighbour density, within a cutoff sphere of radius r_max,
        in n_max radial basis functions and spherical harmonics up to degree
        l_max, with every neighbour smeared by a Gaussian of width sigma. The
        per-site power spectra are L2-normalized and outer-averaged over sites
        (generate()) into one molecular descriptor.

        Parameters
        ----------
        ref_gm : Geometry     reference geometry (supplies the species list)
        r_max  : float, Ang   neighbour cutoff radius
        n_max  : int          number of radial basis functions
        l_max  : int          maximum spherical-harmonic degree
        sigma  : float, Ang   Gaussian width of each atom's density

        Choosing n_max / l_max / r_max / sigma
        --------------------------------------
        The descriptor length -- and hence the cost of EVERY descriptor and,
        more importantly, every descriptor_gradient (the per-step force cost)
        -- scales roughly as

            n_feature ~ (l_max + 1) * n_max^2 * n_species^2 / 2

        (e.g. C/H/O, n_max=l_max=8  ->  ~2700 features). So pick the SMALLEST
        values that resolve the chemistry: over-resolving is quadratically
        expensive in n_max and also overfits on small training sets.

        r_max (cutoff) -- chosen by chemistry, not accuracy. It must enclose
            the interactions that matter (bonds + first/second neighbours),
            typically 3-6 Ang. Bigger is NOT better: once the cutoff exceeds
            the molecular diameter every site sees every atom, so the analytic
            derivative tensor becomes fully dense (no locality to exploit) and
            cost rises with nothing gained. Use the smallest cutoff that still
            captures the relevant couplings.

        sigma (density width) -- small sigma (~0.2-0.3) is sharp and sensitive
            to fine geometric detail but needs higher n_max/l_max to represent
            and reacts strongly to small displacements; large sigma (~0.4-0.6)
            smooths the density into a gentler, lower-resolution, cheaper-to-
            represent descriptor. 0.3-0.5 is a sane default; set it near the
            displacement scale you actually need to resolve.

        n_max (radial resolution) -- how finely DISTANCE is resolved within
            the cutoff. Raise it for larger cutoffs or multi-shell radial
            structure. Usual range 4-8; cost ~ n_max^2.

        l_max (angular resolution) -- how finely ANGLES / neighbour orientation
            are resolved (e.g. distinguishing bond-angle or cis/trans changes).
            Usual range 3-6; cost ~ (l_max + 1). Keep it balanced with n_max --
            l_max >> n_max buys little -- and note dscribe caps l_max at 9 for
            the default basis.

        Practical recipe
        ----------------
        Start moderate -- n_max=l_max=4-6, sigma~0.4, r_max covering first/
        second neighbours -- and only raise n_max/l_max if the surrogate
        underfits (train error stays high once enough data is present); lower
        them for speed whenever accuracy already suffices, since the force-
        evaluation cost is set almost entirely by n_feature. For a small
        molecule with few species (phenol, C/H/O) 5-6 / 4-6 is usually ample;
        n_max=l_max=8 is high-resolution and rarely necessary.
        """
        super().__init__()
        self.atoms = ref_gm.atms
        self.generator = SOAP(
            species = list(set(self.atoms)), # list of unique elements
                                             # in the system
            periodic = False,   # non-periodic system
            r_cut = r_max,      # cutoff radius
            n_max = n_max,      # maximum radial basis functions
            l_max = l_max,      # maximum degree of spherical harmonics
            sigma = sigma,      # Gaussian smearing of atomic densities
            average = "off"     # per-site SOAP; the outer (site) average
                                # is taken explicitly in generate(). This
                                # keeps analytic derivatives available --
                                # dscribe does not provide them for
                                # averaged output, and the outer average
                                # is linear in the per-site descriptors.
        )

    #
    @timer.timed
    def generate(self, gms):
        """
        evaluate the energy at passed geometry, gm
        """

        descriptors = []
        natm        = len(self.atoms)

        if len(gms.shape) == 1:
            eval_gms = np.array([gms], dtype=float)
        else:
            eval_gms = gms

        for i in range(eval_gms.shape[0]):
            gm        = np.reshape(
                            eval_gms[i,:]*constants.bohr2ang,(natm,3))
            molecule  = Atoms(symbols=self.atoms, positions=gm)
            # per-site SOAP, outer-averaged over sites
            descriptor = np.mean(self.generator.create(molecule), axis=0)
            descriptors.append(descriptor/np.linalg.norm(descriptor))

        # return the geometries
        if len(gms.shape) == 1:
            return np.array(descriptors[0])
        else:
            return np.array(descriptors)


    @timer.timed
    def descriptor_gradient(self, gms, delta=None):
        """
        gradient of the (normalized, outer-averaged) SOAP descriptor
        with respect to the cartesian coordinates.

        delta is None  : analytic gradient -- dscribe per-site analytic
                         derivatives, averaged over sites, carrying the
                         generate() normalization and the bohr -> ang
                         unit conversion through the chain rule.
        delta not None : central finite difference of generate() with
                         the given step size (kept for backwards
                         compatibility / cross-checking).

        gradient is returned in a numpy array with
        shape = [ng, nc, n_feature]
        """
        ng   = gms.shape[0]
        nc   = gms.shape[1]
        natm = len(self.atoms)

        n_feature = self.generate(gms[0,:]).shape[0]
        des_grad  = np.zeros((ng, nc, n_feature), dtype=float)

        # backwards-compatible numerical differentiation
        if delta is not None:
            for i in range(ng):
                origin  = np.tile(gms[i,:], (nc, 1))
                disps   = origin + np.diag(np.array([delta]*nc))
                p_grad  = self.generate(disps)
                disps   = origin - np.diag(np.array([delta]*nc))
                m_grad  = self.generate(disps)
                des_grad[i,:,:] = (p_grad - m_grad)/(2.*delta)
            return des_grad

        # analytic differentiation (default)
        for i in range(ng):

            gm       = np.reshape(gms[i,:]*constants.bohr2ang, (natm,3))
            molecule = Atoms(symbols=self.atoms, positions=gm)

            # per-site analytic derivatives and per-site descriptors;
            # attach=True so each SOAP center moves with its atom
            der, des = self.generator.derivatives(molecule,
                                                  method='analytical',
                                                  attach=True,
                                                  return_descriptor=True)
            # der.shape = (nsite, natm, 3, n_feature)
            # des.shape = (nsite, n_feature)

            # outer-averaged (raw) descriptor and its derivative are the
            # mean over sites
            d_raw  = np.mean(des, axis=0)                          # (nf,)
            dd_raw = np.mean(der, axis=0).reshape(nc, n_feature)   # (nc,nf)

            # normalization: dhat = d/||d||
            #   d(dhat) = (1/||d||)(I - dhat dhat^T) d(d)
            # Apply the projector as a RANK-1 update rather than forming the
            # (nf x nf) matrix I - dhat dhat^T: dd_raw @ (I - dhat dhat^T) =
            # dd_raw - (dd_raw . dhat) outer dhat. This is O(nc*nf) instead of
            # O(nc*nf^2) and allocates no nf^2 array -- a large saving since
            # nf is in the thousands (see the cost note in __init__).
            nrm  = np.linalg.norm(d_raw)
            dhat = d_raw/nrm
            dd_n = (dd_raw - np.outer(dd_raw @ dhat, dhat))/nrm    # (nc,nf)

            # chain rule for the bohr -> angstrom coordinate scaling
            des_grad[i,:,:] = dd_n*constants.bohr2ang

        return des_grad

    @timer.timed
    def generate_with_gradient(self, gms):
        """Generate normalized SOAP values and analytic Jacobians together.

        DScribe already returns the per-site descriptor while forming its
        analytic derivative.  Reusing that value avoids a second complete SOAP
        evaluation in a force-bearing surface call.
        """
        geometries = np.asarray(gms, dtype=float)
        single = geometries.ndim == 1
        if single:
            geometries = geometries[None, :]
        natm = len(self.atoms)
        if geometries.ndim != 2 or geometries.shape[1] != 3*natm:
            raise ValueError(
                f'Expected geometries with {3*natm} Cartesian coordinates')
        if not np.all(np.isfinite(geometries)):
            raise ValueError('Geometry coordinates must all be finite')

        molecules = [Atoms(
            symbols=self.atoms,
            positions=row.reshape(natm, 3)*constants.bohr2ang)
            for row in geometries]
        derivative, value = self.generator.derivatives(
            molecules, method='analytical', attach=True,
            return_descriptor=True, n_jobs=1)
        value = np.asarray(value, dtype=float)
        derivative = np.asarray(derivative, dtype=float)
        # DScribe omits the structure axis for a one-item list.
        if value.ndim == 2:
            value = value[None, ...]
            derivative = derivative[None, ...]

        raw = np.mean(value, axis=1)
        raw_jacobian = np.mean(derivative, axis=1).reshape(
            len(geometries), 3*natm, raw.shape[-1])
        norm = np.linalg.norm(raw, axis=1)
        if np.any(norm == 0.):
            raise ValueError('SOAP descriptor has zero norm')
        normalized = raw/norm[:, None]
        projection = np.einsum(
            'gcf,gf->gc', raw_jacobian, normalized, optimize=True)
        jacobian = (
            raw_jacobian
            - projection[..., None]*normalized[:, None, :]
        )/norm[:, None, None]
        jacobian *= constants.bohr2ang

        if single:
            return normalized[0], jacobian
        return normalized, jacobian
