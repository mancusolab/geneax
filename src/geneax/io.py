from typing import Tuple

import pandas as pd

from bed_reader import open_bed
from bgen_reader import open_bgen
from cyvcf2 import VCF

import jax.numpy as jnp

from jax.experimental import sparse
from jax.experimental.sparse import BCOO

from . import geno


def read_plink(
    path: str, scale: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, geno.GenotypeMatrix]:
    """Read in genotype data in `plink 1 <https://www.cog-genomics.org/plink/1.9/input#bed>`_ format.

    Args:
        path: The path for plink genotype data (suffix only).
        scale:

    Returns:
        :py:obj:`Tuple[pd.DataFrame, pd.DataFrame, geno.GenotypeMatrix]`: A tuple of
            #. SNP information (bim; :py:obj:`pd.DataFrame`),
            #. individuals information (fam; :py:obj:`pd.DataFrame`),
            #. genotype matrix (:py:obj:`geno.GenotypeMatrix`).

    """

    with open_bed(path) as bed:
        sparse_matrix = bed.read_sparse(dtype="int8")  # a scipy.sparse.csc_matrix
        sparse_matrix = BCOO.from_scipy_sparse(sparse_matrix.tocoo())

    # we want bed file to be nxp
    geno_matrix = geno.GenotypeMatrix.init(sparse_matrix, scale=scale)
    bim = pd.DataFrame(
        {
            "chrom": bed.chromosome,
            "snp": bed.sid,
            "pos": bed.bp_position,
            "a0": bed.allele_1,
            "a1": bed.allele_2,
        }
    )
    fam = pd.DataFrame({"iid": bed.iid})
    return bim, fam, geno_matrix


def read_vcf(
    path: str, scale: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, geno.GenotypeMatrix]:
    """Read in genotype data in `vcf <https://en.wikipedia.org/wiki/Variant_Call_Format>`_ format.

    Args:
        path: The path for vcf genotype data (full file name).
        scale:

    Returns:
        :py:obj:`Tuple[pd.DataFrame, pd.DataFrame, geno.GenotypeMatrix]`: A tuple of
            #. SNP information (bim; :py:obj:`pd.DataFrame`),
            #. participants information (fam; :py:obj:`pd.DataFrame`),
            #. genotype matrix (:py:obj:`geno.GenotypeMatrix`).

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
    geno_matrix = geno.GenotypeMatrix.init(
        sparse.BCOO.fromdense(jnp.array(bed_list, dtype="float64").T),
        scale=scale,
    )

    return bim, fam, geno_matrix


def read_bgen(
    path: str, scale: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, geno.GenotypeMatrix]:
    """Read in genotype data in `bgen <https://www.well.ox.ac.uk/~gav/bgen_format/>`_ 1.3 format.

    Args:
        path: The path for bgen genotype data (full file name).
        scale:

    Returns:
        :py:obj:`Tuple[pd.DataFrame, pd.DataFrame, geno.GenotypeMatrix]`: A tuple of
            #. SNP information (bim; :py:obj:`pd.DataFrame`),
            #. individuals information (fam; :py:obj:`pd.DataFrame`),
            #. genotype matrix (:py:obj:`geno.GenotypeMatrix`).

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
    geno_matrix = geno.GenotypeMatrix.init(
        sparse.BCOO.fromdense(
            jnp.array(
                jnp.einsum("ijk,k->ij", bgen.read(), jnp.array([0, 1, 2])),
                dtype="float64",
            )
        ),
        scale=scale,
    )

    return bim, fam, geno_matrix
