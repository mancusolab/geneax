from __future__ import annotations

from typing import Optional

from plum import dispatch

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.numpy.linalg as jnla
import lineax as lx

from jax._src.dtypes import JAXType  # ug...
from jax.experimental import sparse
from jaxtyping import Array, ArrayLike, Float, Num


_sparse_mean = sparse.sparsify(jnp.mean)


@jax.jit
@sparse.sparsify
def _get_mean_terms(geno: ArrayLike, covar: ArrayLike) -> Array:
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


def _get_dense_var(geno: sparse.JAXSparse, dense_dtype: JAXType):
    # def _inner(_, variant):
    #     var_idx = jnp.mean(variant **2) - jnp.mean(variant) ** 2
    #     return _, var_idx
    #
    # _, var_geno = lax.scan(_inner, 0.0, geno.T)
    var_geno = (
        _sparse_mean(geno**2, axis=0, dtype=dense_dtype)
        - _sparse_mean(geno, axis=0, dtype=dense_dtype) ** 2
    )

    return var_geno.todense()


class _SparseMatrixOperator(lx.AbstractLinearOperator):
    matrix: sparse.JAXSparse
    dense_dtype: JAXType

    def __init__(self, matrix, dense_dtype: JAXType = jnp.float64):
        self.matrix = matrix
        self.dense_dtype = dense_dtype

    def mv(self, vector: ArrayLike):
        return sparse.sparsify(jnp.matmul)(
            self.matrix, vector, precision=lax.Precision.HIGHEST
        )

    def as_matrix(self) -> Float[Array, "n p"]:
        # raise ValueError("Refusing to materialise sparse matrix.")
        # Or you could do:
        return self.matrix.todense()

    def transpose(self) -> "_SparseMatrixOperator":
        return _SparseMatrixOperator(self.matrix.T, self.dense_dtype)

    def in_structure(self) -> jax.ShapeDtypeStruct:
        _, in_size = self.matrix.shape
        return jax.ShapeDtypeStruct((in_size,), self.dense_dtype)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        out_size, _ = self.matrix.shape
        return jax.ShapeDtypeStruct((out_size,), self.dense_dtype)


class SparseGenotype(lx.AbstractLinearOperator):
    geno: lx.AbstractLinearOperator

    @dispatch
    def __init__(self, geno: lx.AbstractLinearOperator):
        self.geno = geno

    @dispatch
    def __init__(
        self,
        geno: sparse.JAXSparse,
        covar: Optional[ArrayLike] = None,
        scale: bool = False,
        dense_dtype: JAXType = jnp.float64,
    ):
        n, p = geno.shape
        geno_op = _SparseMatrixOperator(geno, dense_dtype)

        if covar is None:
            covar = jnp.ones((n, 1), dtype=dense_dtype)
            beta = _sparse_mean(geno, axis=0, dtype=dense_dtype).todense()
            beta = beta.reshape((1, p))
        else:
            beta = _get_mean_terms(geno, covar)

        center_op = lx.MatrixLinearOperator(covar) @ lx.MatrixLinearOperator(beta)

        if scale:
            wgt = jnp.sqrt(_get_dense_var(geno, dense_dtype))
            scale_op = lx.DiagonalLinearOperator(1.0 / wgt)
            self.geno = (geno_op - center_op) @ scale_op
        else:
            self.geno = geno_op - center_op

    @property
    def dense_dtype(self) -> JAXType:
        return self.out_structure().dtype

    @dispatch
    def __matmul__(self, other: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
        return self.geno @ other

    @dispatch
    def __matmul__(self, vector: Num[ArrayLike, " p"]) -> Float[Array, " n"]:
        return self.mv(vector)

    @dispatch
    def __matmul__(self, matrix: Num[ArrayLike, "p k"]) -> Float[Array, "p k"]:
        return self.mm(matrix)

    @dispatch
    def __rmatmul__(
        self, other: lx.AbstractLinearOperator
    ) -> lx.AbstractLinearOperator:
        return other @ self.geno

    @dispatch
    def __rmatmul__(self, vector: Num[ArrayLike, " n"]) -> Float[Array, " p"]:
        return self.T.mv(vector.T).T

    @dispatch
    def __rmatmul__(self, vector: Num[ArrayLike, "k n"]) -> Float[Array, "k p"]:
        return self.T.mm(vector.T).T

    def mv(self, vector: Num[ArrayLike, " p"]) -> Float[Array, " n"]:
        return self.geno.mv(vector)

    def mm(self, matrix: Num[ArrayLike, "p k"]) -> Float[Array, "n k"]:
        return jax.vmap(self.geno.mv, (1,), 1)(matrix)

    def as_matrix(self) -> Float[Array, "n p"]:
        return self.geno.as_matrix()

    def transpose(self) -> "SparseGenotype":
        return SparseGenotype(self.geno.T)

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return self.geno.in_structure()

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return self.geno.out_structure()


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
