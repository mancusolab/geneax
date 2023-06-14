from __future__ import annotations

from typing import Optional

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.numpy.linalg as jnla
import lineax as lx
from jax.experimental import sparse
from jaxtyping import Array, ArrayLike, Float  # pyright: ignore


@jax.jit
@sparse.sparsify
def _matmul(
    geno: sparse.JAXSparse,
    scale: Float[ArrayLike, " p"],
    covar: Float[ArrayLike, "n k"],
    beta: Float[ArrayLike, "k p"],
    x: Float[ArrayLike, "p ..."],
) -> Float[Array, "p ..."]:
    g_x = jnp.einsum("np,p,p...->n...", geno, scale, x, optimize=True)
    m_x = jnp.einsum("n...,...p,p...->n...", covar, beta, x, optimize=True)
    return g_x - m_x


@jax.jit
@sparse.sparsify
def _rmatmul(
    geno: sparse.JAXSparse,
    scale: ArrayLike,
    covar: ArrayLike,
    beta: ArrayLike,
    x: ArrayLike,
) -> Array:
    x_g = (x @ geno) * scale
    x_m = jnp.einsum("...n,n...,...p->...p", x, covar, beta, optimize=True)
    return x_g - x_m


@jax.jit
@sparse.sparsify
def _matmul_t(
    geno: sparse.JAXSparse,
    scale: ArrayLike,
    covar: ArrayLike,
    beta: ArrayLike,
    x: ArrayLike,
) -> Array:
    g_x = scale * (geno @ x)
    m_x = jnp.einsum(
        "p...,...n,n...->p...", beta, covar, x
    )  # jnp.outer(loc, jnp.sum(x, axis=0))
    return g_x - m_x


@jax.jit
@sparse.sparsify
def _rmatmul_t(
    geno: sparse.JAXSparse,
    scale: ArrayLike,
    covar: ArrayLike,
    beta: ArrayLike,
    x: ArrayLike,
) -> Array:
    x_g = (x * scale) @ geno
    x_m = jnp.einsum(
        "...p,p...,...n->...n", x, beta, covar
    )  # (x @ loc)[:, jnp.newaxis]
    return x_g - x_m


@jax.jit
@sparse.sparsify
def _wgt_sumsq(
    geno: sparse.JAXSparse,
    scale: ArrayLike,
    covar: ArrayLike,
    beta: ArrayLike,
    wgt: ArrayLike,
) -> float:
    n, p = geno.shape

    def __inner(i, val):
        t1 = jnp.sum(geno[:, i] ** 2) * scale[i] ** 2 * wgt[i]
        wgt_cbeta = covar @ beta[:, i]
        t2 = jnp.sum(wgt_cbeta**2) * wgt[i]
        t3 = jnp.sum(geno[:, i] * wgt_cbeta) * scale[i] * wgt[i]
        return val + t1 - 2 * t2 + t3

    wgt_ss = lax.fori_loop(0, p, __inner, 0.0)

    return wgt_ss


@jax.jit
@sparse.sparsify
def _wgt_sq_diag(
    geno: sparse.JAXSparse,
    scale: ArrayLike,
    covar: ArrayLike,
    beta: ArrayLike,
    wgt: ArrayLike,
) -> Array:
    n, p = geno.shape

    def _inner(idx, c):
        local = covar @ beta[:, idx]
        g_scale = geno[:, idx] * scale
        term1 = g_scale**2 * wgt[idx]
        term2 = local**2 * wgt[idx]
        term3 = 2 * g_scale * local * wgt[idx]

        return c + (term1 + term2 - term3)

    result = lax.fori_loop(0, p, _inner, jnp.zeros((n,)))

    return result


@jax.jit
@sparse.sparsify
def _get_model(geno: ArrayLike, scale: ArrayLike, covar: ArrayLike) -> Array:
    m, n = covar.shape
    dtype = covar.dtype
    rcond = jnp.finfo(dtype).eps * max(n, m)
    u, s, vt = jnla.svd(covar, full_matrices=False)
    mask = s >= jnp.array(rcond, dtype=s.dtype) * s[0]

    safe_s = jnp.where(mask, s, 1).astype(covar.dtype)
    s_inv = jnp.where(mask, 1 / safe_s, 0)[:, jnp.newaxis]
    uTb = jnp.matmul(u.conj().T, geno, precision=lax.Precision.HIGHEST)

    beta = jnp.matmul(vt.conj().T, s_inv * uTb * scale, precision=lax.Precision.HIGHEST)

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


@jax.jit
@sparse.sparsify
def _bilinear_trace(
    geno: sparse.JAXSparse,
    scale: ArrayLike,
    covar: ArrayLike,
    beta: ArrayLike,
    L: ArrayLike,
    R: ArrayLike,
):
    term1 = jnp.einsum("kp,np,p,nk->", L, geno, scale, R)
    term2 = jnp.einsum("kp,n...,...p,nk->", L, covar, beta, R)
    return term1 - term2


class SparseGenotype(lx.FunctionLinearOperator):
    def __init__(
        self, geno: sparse.JAXSparse, covar: Optional[ArrayLike] = None, scale=False
    ):
        n, p = geno.shape
        if covar is None:
            covar = jnp.ones((n, 1))
        if scale:
            wgt = jnp.sqrt(_get_var(geno).todense())
            wgt = 1.0 / wgt
        else:
            wgt = jnp.ones((p,))

        beta = _get_model(geno, wgt, covar)

        out_size, in_size = geno.shape
        geno_func = lambda vec: _matmul(geno, wgt, covar, beta, vec)
        in_struct = jax.ShapeDtypeStruct((in_size,), jnp.float32)
        return super().__init__(geno_func, in_struct)


@lx.is_symmetric.register
def _(op: SparseGenotype):
    return False


@lx.is_negative_semidefinite.register
def _(op: SparseGenotype):
    return False


@lx.linearise.register
def _(op: SparseGenotype):
    return op
