"""

几处需要修改的地方，首先是必须安装spconv，因为需要python -m spconv.gencode
然后我发现，就是cmake3.20这种高版本的，都是用这种project(spconv LANGUAGES CXX CUDA)
去吧cuda拿到，这样就会导致我找不到编译器nvcc
之前编译pytorch 源码的时候我直接在cmake的cache里面把编译器指定了，这次也能把编译器的坑走过去
但是这次提示我c++的编译器找不到了，然后上网搜就是得把build文件夹删掉重新cmake，这个就很难受我必须要cache才行啊
所以我就：
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.1/bin/nvcc
直接指定就没问题了，其实上面的问题我感觉在cache里面设置一下路径还是能找到的，就是和正常的cache对比然后看看自己的问题就找到了
但是既然有解决方法就不搞了
编译完了就会有一个so了

./main ../benchmark-pc.jarr ：输出
Hello libspconv!!!
0 23 724 818
0 49 685 566
0 19 845 893
0 42 874 739
0 20 1042 956
num voxels [125562, 3] [200000, 3]
[3, 3, 3] [80, 1600, 1600] [80, 1600, 1600]
native example
implicit gemm example
selected conv algo Turing_f16f16f16f16f16tnt_m32n64k32m32n32k16A0T1688_200_C301LLL_SK
selected conv algo Turing_f16f16f16f16f16tnt_m32n64k32m32n32k16A0T1688_200_C301LLL_SK




---------------------------------------------------------------------------------------------------------------
如何自己安装自己的版本的额spconv，利用及时编译技术
首先确保你的环境是干净的：
pip list | grep spconv
pip list | grep cumm
也就是上面这两个库没有被安装

git clone https://github.com/FindDefinition/cumm, cd ./cumm, pip install -e .
git clone https://github.com/traveller59/spconv, cd ./spconv, pip install -e .
就可以构建库
但是还得在python里面，in python, import spconv and wait for build finish.
就会编译了，编译的文件很大

上面是采用了及时编译技术，导致下一次再去import spconv速度也是很慢的

所以我直接编译了whl文件



Build wheel from source (not recommend, this is done in CI.)
You need to rebuild cumm first if you are build along a CUDA version that not provided in prebuilts.
Linux
    install build-essential, install CUDA
    run export SPCONV_DISABLE_JIT="1"      应该主要就是这个起作用了
    run pip install pccm cumm wheel
    run python setup.py bdist_wheel+pip install dists/xxx.whl
表面上说不推荐，看着还行啊，目前还没遇到坑

cumm 也需要重新构建一下
Build in your environment

    install build-essential, install CUDA
    set env for installed cuda version. for example, export CUMM_CUDA_VERSION="11.4". If you want to build CPU-only,
    run export CUMM_CUDA_VERSION="". If CUMM_CUDA_VERSION isn't set, you need to ensure cuda libraries are inside OS search path,
    and the built wheel name will be cumm, otherwise cumm-cuxxx
    run export CUMM_DISABLE_JIT="1"
    run python setup.py bdist_wheel+pip install dists/xxx.whl

不然会找不到cumm，而且直接pip用预编译的0.3.4的，也提示问题，测试benchmark的时候
直接按照他的要求进行编译cumm即可

https://blog.csdn.net/popuff/article/details/124854726    这篇文章讲得很好
我就只有一个cuda version没有set
安装spconv v2.x的时候要注意把文件夹中pyproject.toml里的cumm依赖删掉，这点github的readme里也有提到。
这个坑不知道会不会影响我
同时不知道各自的spconv.egg-info，会不会影响我，以后采用jit
反正我知道怎么构建了，直接再来一遍就是，如果真有问题
反正自己需要把pccm装好，pccm有没有需要编译的？
Python C++ Code Manager.   可能不需要编译，反正我这没报错，如果报错了给她弄个最新的试试
------------------------------------------------------------------------------------------------------------------------


依赖，这个依赖是之前的装jit版本的
cumm肯定是我自己装的了，pccm不知道
(pytorch112) ros@lxw:/media/ros/A666B94D66B91F4D/ros/learning/3D-detection/spconv_build/spconv$ pip show cumm
Name: cumm
Version: 0.3.5
Summary: CUda Matrix Multiply library
Home-page: https://github.com/FindDefinition/cumm
Author: Yan Yan
Author-email: yanyan.sub@outlook.com
License: MIT
Location: /home/ros/.conda/envs/pytorch112/lib/python3.9/site-packages
Requires: fire, numpy, pccm, pybind11
Required-by: spconv
(pytorch112) ros@lxw:/media/ros/A666B94D66B91F4D/ros/learning/3D-detection/spconv_build/spconv$ pip show pccm
Name: pccm
Version: 0.4.2
Summary: Python C++ Code Manager.
Home-page: https://github.com/FindDefinition/PCCM
Author: Yan Yan
Author-email: yanyan.sub@outlook.com
License: MIT
Location: /home/ros/.conda/envs/pytorch112/lib/python3.9/site-packages
Requires: ccimport, fire, lark, portalocker, pybind11
Required-by: cumm, spconv


benchmnark测试下来速度也差不多
(pytorch112) ros@lxw:/media/ros/A666B94D66B91F4D/ros/learning/3D-detection/spconv_build/spconv$ python -m spconv.benchmark bench_basic f16
basic[f16|ConvAlgo.Native|forward] 12.63622465133667
basic[f16|ConvAlgo.Native|backward] 26.048126678466797
basic[f16|ConvAlgo.MaskImplicitGemm|forward] 10.460615711212158
basic[f16|ConvAlgo.MaskImplicitGemm|backward] 15.38819709777832
basic[f16|ConvAlgo.MaskSplitImplicitGemm|forward] 11.29066556930542
basic[f16|ConvAlgo.MaskSplitImplicitGemm|backward] 13.987599449157715
(pytorch112) ros@lxw:/media/ros/A666B94D66B91F4D/ros/learning/3D-detection/spconv_build/spconv$ python -m spconv.benchmark bench_basic tf32
basic[f32|ConvAlgo.Native|forward] 25.547851791381834
basic[f32|ConvAlgo.Native|backward] 51.76684646606445
basic[f32|ConvAlgo.MaskImplicitGemm|forward] 18.255941734313964
basic[f32|ConvAlgo.MaskImplicitGemm|backward] 34.02391952514648
basic[f32|ConvAlgo.MaskSplitImplicitGemm|forward] 18.14474117279053
basic[f32|ConvAlgo.MaskSplitImplicitGemm|backward] 29.78966262817383
(pytorch112) ros@lxw:/media/ros/A666B94D66B91F4D/ros/learning/3D-detection/spconv_build/spconv$ python -m spconv.benchmark bench_large f16
basic-L[f16|ConvAlgo.Native|forward] 75.40727355957031
basic-L[f16|ConvAlgo.Native|backward] 125.0893862915039
basic-L[f16|ConvAlgo.MaskImplicitGemm|forward] 60.618518295288084
basic-L[f16|ConvAlgo.MaskImplicitGemm|backward] 99.11282348632812
basic-L[f16|ConvAlgo.MaskSplitImplicitGemm|forward] 70.61516258239746
basic-L[f16|ConvAlgo.MaskSplitImplicitGemm|backward] 88.2761099243164
(pytorch112) ros@lxw:/media/ros/A666B94D66B91F4D/ros/learning/3D-detection/spconv_build/spconv$ python -m spconv.benchmark bench_large tf32
basic-L[f32|ConvAlgo.Native|forward] 158.3447399902344
basic-L[f32|ConvAlgo.Native|backward] 286.7404736328125
basic-L[f32|ConvAlgo.MaskImplicitGemm|forward] 103.12301162719727
basic-L[f32|ConvAlgo.MaskImplicitGemm|backward] 216.64170654296876
basic-L[f32|ConvAlgo.MaskSplitImplicitGemm|forward] 111.78061706542968
basic-L[f32|ConvAlgo.MaskSplitImplicitGemm|backward] 180.48415283203124



"""