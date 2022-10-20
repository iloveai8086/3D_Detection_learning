"""

https://zhuanlan.zhihu.com/p/438209175
冰锐大佬的稀疏卷积粗略的理解一下

1.分类
    对于稀疏卷积有两种：
    一种是Spatially Sparse Convolution（空间稀疏卷积） ，在spconv中为SparseConv3d。



"""

import spconv
print(spconv.__version__)

import spconv.pytorch as spconv
print(spconv.SubMConv3d(10,10,(3,3,3)))
import spconv as spconv_core
from spconv.benchmark.core import get_voxel_data, get_voxel_data_large
spconv_core.constants.SPCONV_ALLOW_TF32 = True
# import spconv.pytorch as spconv
# from spconv.pytorch import functional as Fsp
# from torch import nn
# from spconv.pytorch.utils import PointToVoxel
# from spconv.pytorch.hash import HashTable


