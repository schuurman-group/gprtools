"""Shared target normalization for distributed surrogate experts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        # Passing the temporary name to np.savez would silently append
        # ``.npz`` because the name ends in ``.tmp``.  Write to the open file
        # descriptor so the file replaced below is exactly the one created by
        # mkstemp.
        with os.fdopen(fd, 'wb') as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class GlobalTargetScaler:
    """One immutable target transform shared by every BCM expert.

    Samples occupy axis zero in the public :meth:`fit`/:meth:`transform`
    interface.  The ``inverse_model_*`` methods accept the CP convention with
    targets on axis zero, ``(n_targets, ...)``.
    """

    minimum_target_scale: float = 1.e-12
    normalization_version: int = 1
    target_names: tuple[str, ...] = ()
    target_units: tuple[str, ...] = ()
    fitted_from_dataset: str = 'initial_shared_pool'
    target_mean: np.ndarray | None = None
    target_scale: np.ndarray | None = None

    def __post_init__(self):
        self.minimum_target_scale = float(self.minimum_target_scale)
        if (not np.isfinite(self.minimum_target_scale)
                or self.minimum_target_scale <= 0.):
            raise ValueError(
                'minimum_target_scale must be positive and finite')
        self.normalization_version = int(self.normalization_version)
        if self.normalization_version < 1:
            raise ValueError('normalization_version must be positive')
        self.target_names = tuple(str(value) for value in self.target_names)
        self.target_units = tuple(str(value) for value in self.target_units)
        if self.target_mean is not None:
            self.target_mean = np.atleast_1d(
                np.asarray(self.target_mean, dtype=float))
        if self.target_scale is not None:
            self.target_scale = np.atleast_1d(
                np.asarray(self.target_scale, dtype=float))
        if (self.target_mean is None) != (self.target_scale is None):
            raise ValueError(
                'target_mean and target_scale must be supplied together')
        if self.target_mean is not None:
            self._validate_fitted()

    @property
    def n_targets(self):
        self._validate_fitted()
        return int(self.target_mean.size)

    @property
    def normalization_id(self):
        self._validate_fitted()
        digest = hashlib.sha256()
        digest.update(np.asarray(self.target_mean, dtype='<f8').tobytes())
        digest.update(np.asarray(self.target_scale, dtype='<f8').tobytes())
        digest.update(str(self.normalization_version).encode())
        digest.update('\0'.join(self.target_names).encode())
        digest.update('\0'.join(self.target_units).encode())
        return (f'global-target-v{self.normalization_version}-'
                f'{digest.hexdigest()[:16]}')

    def _validate_fitted(self):
        if self.target_mean is None or self.target_scale is None:
            raise RuntimeError('GlobalTargetScaler has not been fitted')
        if self.target_mean.ndim != 1 or self.target_scale.ndim != 1:
            raise ValueError('target statistics must be one-dimensional')
        if self.target_mean.shape != self.target_scale.shape:
            raise ValueError('target mean and scale dimensions do not match')
        if not np.all(np.isfinite(self.target_mean)):
            raise ValueError('target means contain non-finite values')
        if (not np.all(np.isfinite(self.target_scale))
                or np.any(self.target_scale <= 0.)):
            raise ValueError('target scales must be positive and finite')
        count = self.target_mean.size
        if self.target_names and len(self.target_names) != count:
            raise ValueError('target_names dimension does not match targets')
        if self.target_units and len(self.target_units) != count:
            raise ValueError('target_units dimension does not match targets')

    def fit(self, targets):
        values = np.asarray(targets, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError(
                'fit targets must have shape (n_samples, n_targets)')
        if not np.all(np.isfinite(values)):
            raise ValueError('fit targets contain non-finite values')
        self.target_mean = np.mean(values, axis=0)
        self.target_scale = np.maximum(
            np.std(values, axis=0), self.minimum_target_scale)
        self._validate_fitted()
        return self

    def _sample_layout(self, value, label):
        self._validate_fitted()
        array = np.asarray(value, dtype=float)
        scalar_layout = False
        if array.ndim == 0:
            if self.n_targets != 1:
                raise ValueError(
                    f'{label} scalar is invalid for {self.n_targets} targets')
            array = array.reshape(1)
            scalar_layout = True
        elif array.ndim == 1:
            if self.n_targets == 1:
                # One target and many samples.
                scalar_layout = True
            elif array.shape[0] != self.n_targets:
                raise ValueError(
                    f'{label} must contain {self.n_targets} targets; '
                    f'received shape {array.shape}')
            # Otherwise this is one multi-target sample; ordinary trailing-
            # axis broadcasting below is exactly the desired layout.
        elif array.shape[-1] != self.n_targets:
            raise ValueError(
                f'{label} must have last dimension {self.n_targets}; '
                f'received shape {array.shape}')
        if not np.all(np.isfinite(array)):
            raise ValueError(f'{label} contains non-finite values')
        return array, scalar_layout

    def transform(self, targets):
        values, scalar = self._sample_layout(targets, 'targets')
        if scalar:
            return (values - self.target_mean[0])/self.target_scale[0]
        return (values - self.target_mean)/self.target_scale

    def inverse_transform_mean(self, mean):
        values, scalar = self._sample_layout(mean, 'mean')
        if scalar:
            return self.target_mean[0] + self.target_scale[0]*values
        return self.target_mean + self.target_scale*values

    def inverse_transform_std(self, std):
        values, scalar = self._sample_layout(std, 'standard deviation')
        if np.any(values < 0.):
            raise ValueError('standard deviations cannot be negative')
        if scalar:
            return self.target_scale[0]*values
        return self.target_scale*values

    def inverse_transform_variance(self, variance):
        values, scalar = self._sample_layout(variance, 'variance')
        if np.any(values < -1.e-14):
            raise ValueError('variances cannot be negative')
        if scalar:
            return self.target_scale[0]**2*values
        return self.target_scale**2*values

    def _model_layout(self, value, label):
        self._validate_fitted()
        array = np.asarray(value, dtype=float)
        if array.ndim < 1 or array.shape[0] != self.n_targets:
            raise ValueError(
                f'{label} first dimension must be {self.n_targets}')
        if not np.all(np.isfinite(array)):
            raise ValueError(f'{label} contains non-finite values')
        shape = (self.n_targets,) + (1,)*(array.ndim - 1)
        return array, shape

    def inverse_model_mean(self, mean):
        values, shape = self._model_layout(mean, 'model mean')
        return (self.target_mean.reshape(shape)
                + self.target_scale.reshape(shape)*values)

    def inverse_model_std(self, std):
        values, shape = self._model_layout(std, 'model standard deviation')
        return self.target_scale.reshape(shape)*values

    def inverse_model_variance(self, variance):
        values, shape = self._model_layout(variance, 'model variance')
        return self.target_scale.reshape(shape)**2*values

    def metadata(self):
        self._validate_fitted()
        return {
            'normalization_version': self.normalization_version,
            'normalization_id': self.normalization_id,
            'target_names': list(self.target_names),
            'target_units': list(self.target_units),
            'fitted_from_dataset': self.fitted_from_dataset,
            'minimum_target_scale': self.minimum_target_scale,
            'n_targets': self.n_targets,
            'noise_convention': 'normalized_numerical_jitter_variance',
        }

    def save(self, path, metadata_path=None):
        path = Path(path)
        _atomic_npz(
            path,
            target_mean=np.asarray(self.target_mean, dtype=float),
            target_scale=np.asarray(self.target_scale, dtype=float))
        metadata = self.metadata()
        metadata['npz_sha256'] = _file_sha256(path)
        meta_path = (Path(metadata_path) if metadata_path is not None
                     else path.with_name(path.stem + '_metadata.json'))
        _atomic_json(meta_path, metadata)

    @classmethod
    def load(cls, path, metadata_path=None):
        path = Path(path)
        meta_path = (Path(metadata_path) if metadata_path is not None
                     else path.with_name(path.stem + '_metadata.json'))
        if not path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                f'target scaler artifacts are incomplete: {path}, '
                f'{meta_path}')
        metadata = json.loads(meta_path.read_text())
        expected = metadata.get('npz_sha256')
        if expected and _file_sha256(path) != expected:
            raise RuntimeError('target scaler hash does not match metadata')
        with np.load(path) as data:
            scaler = cls(
                minimum_target_scale=metadata['minimum_target_scale'],
                normalization_version=metadata['normalization_version'],
                target_names=tuple(metadata.get('target_names', ())),
                target_units=tuple(metadata.get('target_units', ())),
                fitted_from_dataset=metadata.get(
                    'fitted_from_dataset', 'initial_shared_pool'),
                target_mean=np.asarray(data['target_mean'], dtype=float),
                target_scale=np.asarray(data['target_scale'], dtype=float))
        if metadata.get('normalization_id') not in (
                None, scaler.normalization_id):
            raise RuntimeError('target scaler normalization ID mismatch')
        return scaler
