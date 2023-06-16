from __future__ import annotations

from functools import singledispatchmethod
from typing import Optional

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.numpy.linalg as jnla
import lineax as lx

from jax.experimental import sparse
from jaxtyping import Array, ArrayLike


def _default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


@jax.jit
@sparse.sparsify
def _get_model(geno: ArrayLike, covar: ArrayLike) -> Array:
    m, n = covar.shape
    dtype = covar.dtype
    rcond = jnp.finfo(dtype).eps * max(n, m)
    u, s, vt = jnla.svd(covar, full_matrices=False)
    mask = s >= jnp.array(rcond, dtype=s.dtype) * s[0]

    safe_s = jnp.where(mask, s, 1).astype(covar.dtype)
    s_inv = jnp.where(mask, 1 / safe_s, 0)[:, jnp.newaxis]
    uTb = jnp.matmul(u.conj().T, geno, precision=lax.Precision.HIGHEST)

    beta = jnp.matmul(vt.conj().T, s_inv * uTb, precision=lax.Precision.HIGHEST)

    return beta


@jax.jit
@sparse.sparsify
def _get_var(geno: sparse.JAXSparse):
    n, p = geno.shape

    # def _inner(_, variant):
    #     var_idx = jnp.mean(variant **2) - jnp.mean(variant) ** 2
    #     return _, var_idx
    #
    # _, var_geno = lax.scan(_inner, 0.0, geno.T)
    var_geno = jnp.mean(geno**2, axis=0) - jnp.mean(geno, axis=0) ** 2

    return var_geno


class _SparseMatrixOperator(lx.AbstractLinearOperator):
    matrix: sparse.JAXSparse

    def mv(self, vector: ArrayLike):
        return sparse.sparsify(jnp.matmul)(
            self.matrix, vector, precision=lax.Precision.HIGHEST
        )

    def as_matrix(self):
        raise ValueError("Refusing to materialise sparse matrix.")
        # Or you could do:
        # return self.matrix.todense()

    def transpose(self):
        return _SparseMatrixOperator(self.matrix.T)

    def in_structure(self):
        _, in_size = self.matrix.shape
        return jax.ShapeDtypeStruct((in_size,), _default_floating_dtype())

    def out_structure(self):
        out_size, _ = self.matrix.shape
        return jax.ShapeDtypeStruct((out_size,), _default_floating_dtype())


class SparseGenotype(lx.AbstractLinearOperator):
    geno: lx.AbstractLinearOperator

    def __init__(
        self, geno: sparse.JAXSparse, covar: Optional[ArrayLike] = None, scale=False
    ):
        n, p = geno.shape
        if covar is None:
            covar = jnp.ones((n, 1))
        if scale:
            wgt = jnp.sqrt(_get_var(geno).todense()).astype(covar.dtype)
            wgt = 1.0 / wgt
        else:
            wgt = jnp.ones((p,))

        beta = _get_model(geno, covar)

        geno_op = _SparseMatrixOperator(geno)
        scale_op = lx.DiagonalLinearOperator(wgt)

        center_op = lx.MatrixLinearOperator(covar) @ lx.MatrixLinearOperator(beta)
        self.geno = (geno_op - center_op) @ scale_op

    @singledispatchmethod
    def __matmul__(self, other: lx.AbstractLinearOperator):
        return self.geno @ other

    @__matmul__.register
    def _(self, other: ArrayLike):
        return self.mv(other)

    @singledispatchmethod
    def __rmatmul__(self, other: lx.AbstractLinearOperator):
        return other @ self.geno

    @__rmatmul__.register
    def _(self, other: ArrayLike):
        return self.T.mv(other.T).T

    def mv(self, vector: ArrayLike):
        return self.geno.mv(vector)

    def as_matrix(self):
        return self.geno.as_matrix()

    def transpose(self):
        return self.geno.T

    def in_structure(self):
        return self.geno.in_structure()

    def out_structure(self):
        return self.geno.out_structure()

    # @property
    # def shape(self):
    #     return self.geno.out_structure(), self.geno.in_structure()


@lx.is_symmetric.register(_SparseMatrixOperator)
@lx.is_symmetric.register(SparseGenotype)
def _(op):
    return False


@lx.is_negative_semidefinite.register(_SparseMatrixOperator)
@lx.is_negative_semidefinite.register(SparseGenotype)
def _(op):
    return False


@lx.linearise.register(_SparseMatrixOperator)
@lx.linearise.register(SparseGenotype)
def _(op):
    return op
