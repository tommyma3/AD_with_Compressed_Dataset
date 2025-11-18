from .ad import AD
from .compressed_ad import CompressedAD
from .ad_compressed import CompressedAD as CompressedADv2

MODEL = {
    "AD": AD,
    "CompressedAD": CompressedAD,
    "CompressedADv2": CompressedADv2,
}