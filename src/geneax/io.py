import warnings

from typing import Tuple

import pandas as pd

from jax import config


config.update("jax_enable_x64", True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from bgen_reader import open_bgen
    from cyvcf2 import VCF
    from pandas_plink import read_plink

    import jax.numpy as jnp

import geno

from jax.experimental import sparse


## Modified from Sushie https://github.com/mancusolab/sushie/blob/main/sushie/io.py#L194
def read_triplet(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, geno.SparseGenotype]:
    """Read in genotype data in `plink 1 <https://www.cog-genomics.org/plink/1.9/input#bed>`_ format.

    Args:
        path: The path for plink genotype data (suffix only).

    Returns:
        :py:obj:`Tuple[pd.DataFrame, pd.DataFrame, jnp.ndarray]`: A tuple of
            #. SNP information (bim; :py:obj:`pd.DataFrame`),
            #. individuals information (fam; :py:obj:`pd.DataFrame`),
            #. genotype matrix (bed; :py:obj:`jnp.ndarray`).

    """

    bim, fam, bed = read_plink(path, verbose=False)
    bim = bim[["chrom", "snp", "pos", "a0", "a1"]]
    fam = fam[["iid"]]
    # we want bed file to be nxp
    bed = geno.SparseGenotype(
        sparse.BCOO.fromdense(jnp.array(bed.compute().T, dtype="float64"))
    )
    return bim, fam, bed


def read_vcf(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, geno.SparseGenotype]:
    """Read in genotype data in `vcf <https://en.wikipedia.org/wiki/Variant_Call_Format>`_ format.

    Args:
        path: The path for vcf genotype data (full file name).

    Returns:
        :py:obj:`Tuple[pd.DataFrame, pd.DataFrame, jnp.ndarray]`: A tuple of
            #. SNP information (bim; :py:obj:`pd.DataFrame`),
            #. participants information (fam; :py:obj:`pd.DataFrame`),
            #. genotype matrix (bed; :py:obj:`jnp.ndarray`).

    """

    vcf = VCF(path, gts012=True)
    fam = pd.DataFrame(vcf.samples).rename(columns={0: "iid"})
    bim_list = []
    bed_list = []
    for var in vcf:
        # var.ALT is a list of alternative allele
        bim_list.append([var.CHROM, var.ID, var.POS, var.ALT[0], var.REF])
        tmp_bed = 2 - var.gt_types
        bed_list.append(tmp_bed)

    bim = pd.DataFrame(bim_list, columns=["chrom", "snp", "pos", "a0", "a1"])
    bed = geno.SparseGenotype(
        sparse.BCOO.fromdense(jnp.array(bed_list, dtype="float64").T), scale=True
    )

    return bim, fam, bed


def read_bgen(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, geno.SparseGenotype]:
    """Read in genotype data in `bgen <https://www.well.ox.ac.uk/~gav/bgen_format/>`_ 1.3 format.

    Args:
        path: The path for bgen genotype data (full file name).

    Returns:
        :py:obj:`Tuple[pd.DataFrame, pd.DataFrame, jnp.ndarray]`: A tuple of
            #. SNP information (bim; :py:obj:`pd.DataFrame`),
            #. individuals information (fam; :py:obj:`pd.DataFrame`),
            #. genotype matrix (bed; :py:obj:`jnp.ndarray`).

    """

    bgen = open_bgen(path, verbose=False)
    fam = pd.DataFrame(bgen.samples).rename(columns={0: "iid"})
    bim = pd.DataFrame(
        data={"chrom": bgen.chromosomes, "snp": bgen.rsids, "pos": bgen.positions}
    )
    allele = (
        pd.DataFrame(bgen.allele_ids)[0]
        .str.split(",", expand=True)
        .rename(columns={0: "a0", 1: "a1"})
    )
    bim = pd.concat([bim, allele], axis=1).reset_index(drop=True)[
        ["chrom", "snp", "pos", "a0", "a1"]
    ]
    bed = geno.SparseGenotype(
        sparse.BCOO.fromdense(
            jnp.array(
                jnp.einsum("ijk,k->ij", bgen.read(), jnp.array([0, 1, 2])),
                dtype="float64",
            )
        ),
        scale=True,
    )

    return bim, fam, bed
