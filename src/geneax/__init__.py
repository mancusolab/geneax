try:
    from ._version import version as __version__, version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

from .geno import GenotypeMatrix as GenotypeMatrix
from .io import (
    read_bgen as read_bgen,
    read_plink as read_plink,
    read_vcf as read_vcf,
)
