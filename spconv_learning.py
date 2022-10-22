"""

https://zhuanlan.zhihu.com/p/438209175
冰锐大佬的稀疏卷积粗略的理解一下,并且附带了second算法

1.分类
    对于稀疏卷积有两种：
    一种是Spatially Sparse Convolution（空间稀疏卷积） ，在spconv中为SparseConv3d。
    就像普通的卷积一样，只要kernel 覆盖一个 active input site，就可以计算出output site。
    对应论文SECOND: Sparsely Embedded Convolutional Detection

    另一种是Submanifold Sparse Convolution（子流形稀疏卷积）， 在spconv中为SubMConv3d。
    只有当kernel的中心覆盖一个 active input site时，卷积输出才会被计算。
    对应论文：3D Semantic Segmentation with Submanifold Sparse Convolutional Networks

    SubMConv3d输入与输出feature map上不为空的位置相同，保持了稀疏性(sparity=不为空的位置/所有位置和）
    因为子流形卷积只有卷积kernel中心碰到不为空的才去做卷积
    而SparseConv3d会增加稀疏性，从而也就增加了计算量。
    但是如果只用SubMConv3d，卷积核的感受野会限制在一定范围内，所以要结合stride=2的SparseConv3d一起使用，在尽量保持稀疏性的同时增大感受野。
    下采样了，stride=2

    暂时看不懂这个人写的，先看另一个大佬写的

    现在大致知道在干什么了：
    看看冰锐大佬的解释：先Rulebook的第一列表示activate input set在kernel中的位置
    这里是3x3的卷积核，因此有9不同的值；
    （我感觉这边有问题啊，之前的文章里面只有8个值，卷积过程里面，记录每个元素的位置，去除掉重复的也就是8个）
    第二列 count 表示在当前卷积核位置处有多少个激活值，用于构建矩阵形状；  （其实就是卷积核碰到了几个非空的数据）



"""

"""

https://zhuanlan.zhihu.com/p/382365889
稀疏卷积的通俗理解：不知道是spconv1.x？ or 2.x
    本文的理论部分是在“3D Semantic Segmentation with Submanifold Sparse Convolutional Networks”的基础上完成的。
    实现部分，是基于“SECOND: Sparsely Embedded Convolutional Detection”的论文。
    
    为什么提出稀疏卷积？它有什么好处？：
        卷积神经网络已经被证明对于二维图像信号处理是非常有效的。然而，对于三维点云信号，额外的维数 z 显著增加了计算量。
        另一方面，与普通图像不同的是，大多数三维点云的体素是空的，这使得三维体素中的点云数据通常是稀疏信号。
        我们是否只能有效地计算稀疏数据的卷积，而不是扫描所有的图像像素或空间体素？
        否则这些空白区域带来的计算量太多余了。
        这就是 sparse convolution 提出的motivation。
    
    定义：
        本文以二维稀疏图像处理为例
        由于稀疏信号采用数据列表和索引列表表示，二维和三维稀疏信号没有本质区别。
        
        2D input signal，rank=3，shape=【c=3，h=5，w=5】
        如图所示，我们有一个5 × 5的3通道图像。除了 P1和 P2两点外，所有像素都是(0,0,0) （虽然0这个假设也很不严谨）
        P1和 P2，这种非零元素也称为active input sites。
        
        在稀疏格式中，数据列表是[[0.1,0.1,0.1] ，[0.2,0.2,0.2] ，索引列表是[1,2] ，[2,3] ，并且是 YX 顺序。
    这边涉及的图文很多，直接看md文件吧
    
    我这边再补充几个经典的评论：
    我这么理解对吗，稀疏卷积就是用rulebook代替划窗，只去有效的位置找到需要的值进行计算。
    
"""

"""

目前较为流行的3D点云稀疏加速四大框架，具体分为， 
mit-han-lab的torchsparse、NVIDAI的MinkowskiEngine、 Tusimple的spconv、facebookresearch的SparseConvNet，
具体进行实验以Torchsparse为研究起始，但发现其仅仅是完成了submanifold的实现且功能也不完整（还不成熟），
随后经过对spconv的深入研究，发现若要直接的在pytorch层进行修改优化以达到模型稀疏加速的目的是非常有限的，需要在CUDA底层的优化设计进行考虑。

"""

"""

我认为稀疏卷积的难点是什么：
我直接从最大的那个图开始看起：
普通卷积im2col 然后展平kernel，或者复制（看老潘）就知道怎么算gemm
但是稀疏卷积就是在做那个表去im2col
举例子用的都是单个元素在做的，而且看单个卷积的过程，就是卷积核不为0，但是image上面为0
单个元素做完后，就是也像遍历滑窗一样，遍历完特征图，假如不padding
那么5*5 -> 3*3后，肯定最后的3*3的特征图有很多地方是空的，因为有的地方input全是空的
输入是有一个索引的我们可以拿到，非零的索引
那么卷积核碰到多个非空元素怎么办？

按照作者一开始的思路，就是单个元素单个元素的计算，每一个非空的元素其实都是有一个input的索引的
就是hash_in
而hash_out图解看着是在做着和卷积类似的操作，实际上就是有一个single channel kernel tempate
p_in我们知道的
p_out 的求取就是类似在做卷积，得到每一个非空元素在输出特征图上面的索引位置
对每个非空的点，利用p_in - p_out得到一个坐标，然后去single channel kernel tempate找对应的位置，注意是横着为x，纵向为y
得到了（i，j）
而每一个（i，j）都有对应的一个或者多个input，可能只对应一个input，也可能对应两个input，或许多个

（这里还需要补充下，做减法这个也是我总结看出来的，确实就是和分解单个元素九次卷积的过程一致，确实就是input对应在卷积和索引的位置）
（真实神奇了，怎么想到的，或者说理论依据是什么？）

当然冰锐大佬给出的解释就是3*3的卷积核，就是九个（i，j），但是我从举例子发现，如果多个input都没碰到卷积核，那么自然那个位置
不会被记录到hash_out里面去
自然也就没有（i,j）了
多个输入重复在（i，j）做卷积的，就是会有count来加一了，为什么这么做，首先整个特征图是共享卷积核的，自然你这个输入位置碰到了卷积核的位置
那就是对后续的权重做了贡献
但是每次碰到的，在特征图的位置不同，就得根据哈希表里面的v_out知道key_out，知道当前的元素最后该加到输出特征图的哪边
就像传统的卷积一样，是3*3 与3*3对应元素相乘相加到一个元素
那我这边count不为零的位置，就是一个卷积核的某一个位置多次碰到了不为零的元素，所以得给她加起来，但是最后加的位置是根据索引来的
所以才有冰锐大佬，每次都是3*3的就是9个（i，j）

这样rulebook hash-in 和 hash-out就知道怎么对应了

剩下来的就是怎么做到im2col了
我看老潘讲的im2col，kernel三通道都变成一行，然后多行就是多个卷积核
大图里面的
 0.1 0.1 0.1       F0 F0 
            gemm   F0 F0
                   F0 F0
转置过来了？
im2col 需要补吗？                               
按照表（-1，1）确实就是F0位置的卷积，然后三个通道就是竖着，一共两个通道就是两列，然后每个通道的加起来成一个数
就是1*3 * 3*2 = 1*2的，两个值，两个通道                   ★就是现在是三通道两个卷积核，根据卷积核位置的索引，那么就是这样子构建矩阵的

对于p1 代表的 (0.1, 0.1, 0.1)，分别跟深色和浅色两个kernel进行卷积运算，得到深黄色和浅黄色两个channel的输出。
可以看到红色操作和蓝色操作有相同的output index (v_out），没事的，直接把他们的输出加起来就好了。


看着就不像有im2col的过程

但是看评论：
被卷积图片是 5x5 的卷积核是 3x3，padding = 0，按原始 im2col 卷积的话，矩阵乘法则为 9x9 矩阵和 9x1 矩阵相乘
这个没错，确实是5*5 需要变成9*9的矩阵，然后单个通道展平成9*1
稀疏矩阵中，因为有两个点不为零，按 rulebook/im2col 转换，矩阵乘法则为 8x9 矩阵和 9x1 矩阵相乘；
在这个例子里面，最后复杂度上优化了一排计算，实际情况下，具体优化多少要看实际分布。


为什么是8*9的矩阵？
就是看hash_out的构建过程，里面只有一次是两个不为空的数据，卷积核都没碰到两个数据，自然就没有矩阵，因为根本就没有构建hash_out
可能就是这样子的，才会有8*9的矩阵，是一个卷积核出现8*9的矩阵；然后自然卷积核就是9*1
最后就是8*1的输出；
其实这边如果是多个输入通道； 8*9*3 和 9*1*3
wait
换个理解的
从9这个维度理解
9自然就是一次卷积里面的运算
竖着的9个元素，一共八个列，就是8*9；卷积核是9*1*3  其实就是拉平的27那种
8*9三通道之后就是27*8
然后1*27*27*8？   最后得到的就是1*8的矩阵，当有多个卷积核就是2*8  n*8
就和他图里面的9*2是类似的，图里面是每个卷积核滑块的时候索引，都会碰到非空元素
这样子这个矩阵就不是一个定长的序列，自然就tensorrt不太好弄啊
上面这个结合老潘的im2col的图理解才行
一个是大图流程，一个是老潘的图：
https://mp.weixin.qq.com/s?__biz=Mzg3ODU2MzY5MA==&mid=2247488158&idx=1&sn=3722bc7433811d494e179cb828dade32&chksm=cf108a9bf867038d4e7e451212429925a48dcbd47d9811315c132271f104540fe503555e7611&token=1276531538&lang=zh_CN#rd
有没有什么好的方法知道他是8*9的矩阵，或者就是9*8的矩阵？、
感觉就是看hash_out了，他就8个v_out key_out 所以这个就可以构建输入



"""

"""

接下来就是冰锐大佬的解释了。
就是对那个大图片的解释：
首先Rulebook的第一列表示activate input set在kernel中的位置，这里是3x3的卷积核，因此有9不同的值；
第二列 count 表示在当前卷积核位置处有多少个激活值，用于构建矩阵形状；  怎么构建仔细想想，构建什么矩阵的形状
第三列的in ，对应 hash-in的索引值  就是第几个输入
最后一列out 的对应hash-out 的索引值，就是hash-out表的位置，实际还需要根据位置找到坐标
hash-in 和hash-out 的key这里表示为输入特征的坐标形式，value代表这个坐标对应的特征的哪一行特征。
在计算过程中根据Rulebook，逐个卷积核元素位置计算，如图中的红色或者蓝色箭头所示，
首先分别找到卷积核和输入特征元素，组成矩阵，然后调用GEMM加速矩阵计算，最后根据hash-out恢复空间位置。
为什么根据这个恢复，因为这个表就是记录了输出在什么位置，就是pout，还可以根据pin算出offset
与传统的卷积计算过程类似，都是一个《Gather-GEMM-Scatter》的过程，
不同的是传统卷积的Gather过程是经典的im2col，Scatter恢复原始空间位置的也相对容易；
而稀疏卷积的Gather和Scatter过程则需要根据提前构建好的Rulebook以及hash表寻找对应位置元素，从而构建矩阵和恢复空间位置。
传统卷积和稀疏卷积相同的是都采用GEMM做矩阵运算的加速。
在spconv中，直接调用torch.mm_out函数（依赖torch已经做好的GEMM）。  当时可能是这样的1.2的版本，现在不知道了
GEMM并未改变运算量，是通过把循环分开写，通过更多的利用cache储存来代替内存访问，从而减少时间。


"""

# import tensorrt
# print(tensorrt.__version__)

import spconv
print(spconv.__version__)

import spconv.pytorch as spconv
print(spconv.SubMConv3d(10,10,(3,3,3)))
print(spconv.SparseConv3d(10,10,(3,3,3)))
import spconv as spconv_core
from spconv.benchmark.core import get_voxel_data, get_voxel_data_large
spconv_core.constants.SPCONV_ALLOW_TF32 = True
# import spconv.pytorch as spconv
# from spconv.pytorch import functional as Fsp
# from torch import nn
# from spconv.pytorch.utils import PointToVoxel
# from spconv.pytorch.hash import HashTable


