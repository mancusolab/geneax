from typing import Type

import numpy.testing as nptest
import pytest

import jax.experimental.sparse as xsp
import jax.numpy as jnp
import jax.random as rdm

from jax.config import config

import geneax as gx


config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "highest")


def _get_geno(key, N, P):
    G = jnp.sum(rdm.bernoulli(key, 0.5, shape=(N, P, 2)), axis=-1).astype(jnp.int8)
    return G


@pytest.mark.parametrize(
    "N,P,K,sp_cls",
    [
        (50, 100, 1, xsp.BCOO),
        (50, 100, 10, xsp.BCOO),
        (100, 50, 10, xsp.BCOO),
        # (50, 100, 1, xsp.BCSR),
        # (50, 100, 10, xsp.BCSR),
        # (100, 50, 10, xsp.BCSR),
    ],
)
def test_matmul(N: int, P: int, K: int, sp_cls: Type[xsp.JAXSparse], seed: int = 0):
    key = rdm.PRNGKey(seed)
    key, g_key = rdm.split(key)

    G = _get_geno(g_key, N, P)
    M = jnp.mean(G, axis=0, dtype=jnp.float64)
    S = 1.0 / jnp.std(G, axis=0, dtype=jnp.float64)
    Z = (G - M) * S
    geno = gx.GenotypeMatrix.init(
        sp_cls.fromdense(G),
        scale=True,
    )

    key, r_key = rdm.split(key)
    R = rdm.normal(r_key, shape=(P, K))
    if K == 1:
        # edge-case that breaks due to broadcasting between (K,) * (K,1) = (K,K)
        R = R.flatten()

    observed = geno @ R
    expected = Z @ R
    nptest.assert_allclose(observed, expected)


@pytest.mark.parametrize(
    "N,P,K,sp_cls",
    [
        (50, 100, 1, xsp.BCOO),
        (50, 100, 10, xsp.BCOO),
        (100, 50, 10, xsp.BCOO),
        # (50, 100, 1, xsp.BCSR),
        # (50, 100, 10, xsp.BCSR),
        # (100, 50, 10, xsp.BCSR),
    ],
)
def test_rmatmul(N: int, P: int, K: int, sp_cls: Type[xsp.JAXSparse], seed: int = 0):
    key = rdm.PRNGKey(seed)
    key, g_key = rdm.split(key)

    G = _get_geno(g_key, N, P)
    M = jnp.mean(G, axis=0, dtype=jnp.float64)
    S = 1.0 / jnp.std(G, axis=0, dtype=jnp.float64)
    Z = (G - M) * S
    geno = gx.GenotypeMatrix.init(
        sp_cls.fromdense(G),
        scale=True,
    )

    key, l_key = rdm.split(key)
    L = rdm.normal(l_key, shape=(K, N))

    observed = L @ geno
    expected = L @ Z
    nptest.assert_allclose(observed, expected)
