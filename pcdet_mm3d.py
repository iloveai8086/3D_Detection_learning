'''

1.
    x前，y左，z上这样的，而长宽高就是分别对应三轴的。
其中，(cx, cy, cz) 为物体3D框的几何中心位置，(dx, dy, dz)分别为物体3D框在heading角度为0时沿着x-y-z三个方向的长度，heading为物体在俯视图下的朝向角 (沿着x轴方向为0度角，逆时针x到y角度增加)。
	基于 PCDet 所采用的标准化3D框定义，我们再也不用纠结到底是物体3D中心还是物体底部中心；再也不用纠结物体三维尺寸到底是l-w-h排列还是w-l-h排列；再也不用纠结heading 0度角到底是哪，到底顺时针增加还是逆时针增加。
	如何支持新的数据集？
	如之前所说，PCDet的数据-模型分离框架设计与规范化的坐标表示使得其很容易扩展到新的数据集上。具体来说，研究者只需要在自己的dataloader里面做以下两件事:
(1) 在 self._getitem_() 中加载自己的数据，并将点云与3D标注框均转至前述统一坐标定义下，送入数据基类提供的 self.prepare_data()；
(2) 在 self.generate_prediction_dicts()中接收模型预测的在统一坐标系下表示的3D检测框，并转回自己所需格式即可。
	如何组合、改进旧模型+支持新的模型？
	如图3所示，PCDet中实际上已经支持了绝大部分的模块。对于一个新的(组合的)3D检测模型来说，只要在PCDet框架中实现其所特有的模块（比如新的backbone或新的head）来替换掉原有模块，并修改响应模型配置文件，其他模块以及数据处理部分直接利用PCDet中已有部分即可。

	经典评论：
	range image：
	range image对高效大范围检测确实有优势，不过估计还得研究研究怎么在这个变形的视角更好学3D特征
	嗯，是啊。。。range view下主要是没有损失任何信息，而且representation的效率高了很多，这点我比较关注
	或许skip掉backbone3d，在这个框架里用2dcnn+head也可以做range image....[飙泪笑]，这块没试过之后可以考虑一下RCD
	arxiv.org/pdf/2005.09927.pdf 看起来有点前途。。。
	range view好像就Uber做的性能可以达到跟bev的
	我也认同range view是下一个方向，直接用传感器的原始数据，现在这么多论文基本都是voxelnet那一套，实际能用的没多少，range view的方法Uber又不开源，GitHub上也没有复现的，所以很少人做吧。
	RCD waymo的挺work啊，上面也提到了
	rcd也是在lasernet上做的，可以说是对lasernet的改进，还加了第二个stage
	嗯，第二个stage 又用回了point cloud 感觉有点偏回去了（变慢..）

	pillar speed / accuracy 都还好吧..     本质是手工对xyz拆成xy和z，这面就已经损失了不少信息。小物体差挺多的  嗯，也是，不过大物体我们nuScenes pp voxelnet基本一样了.   大物体确实都差不多。。
	换句话说，在空间中相近的点拆到两个pillar中关系就很弱了。这也是我认为pvrcnn去显式model neighbour 很有帮助的原因
	我感觉pillar是voxel方法里比较能够商用的，你说小物体pillar检测不好，那是肯定的，毕竟小物体就那么点，怎么体现cnn这种从低维到高维提取特征．cvpr2020一篇论文HVnet就是对pillar的一个扩展，每个点的feature能够对应多个尺度的pillar, 能够把小物体的ap提上来，速度也没降多少
	嗯，商用好像pillar比较多。hvnet还 没仔细看

	实际落地小目标检测相当难的，点少是一方面，行人和骑自行车的，然后再来一个人推着自行车走。。
	了解.. 慢慢改进，看着最近一个月井喷了一堆3d detection repo, 大概不用几年就和2d差不多了..  （直至今日，也不行啊）

————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	0.3版本的pcdet的改进:
	V0.3.0版本中支持了PointRCNN，PartA2-Free等point-based和anchor-free的网络结构，我们OpenPCDet架构中的每个部分终于都填上了。同时现在大家也可以自由切换PointNet++和SparseConv的backbone自由组合了。
	另外，相比我们最早发布的PointRCNN代码，OpenPCDet版本的PointRCNN支持了joint训练以及多类预测，只需要3小时即可训好，同时也可以稳定的复现PointRCNN Paper上的KITTI性能了。(也就是把二阶段的网络直接一阶段训练了)
	同时，我们也在OpenPCDet上提供两个强baseline算法，PointPillar-Multihead和Second-Multihead，包括去年NuScenes比赛中的第一算法CBGS，目前NDS可以稳定达到62+。


关于anchor_bottom_heights这个设置的问题：看保存的issue和微信的聊天记录，可能是手误了


这篇文章主要收集了3D物体检测的算法，先笼统列出来，后续在进行分类：
pointnet && pointnet++										√
voxelnet													√
second														√
pointpillars												√
centerpoint 												
pointrcnn													√
pvrcnn 														√
pvrcnn++
voxelrcnn													√
SST 
bevformer
rangedet
fcos3d
SSN
RSN
3DSSD
smoke
lidarRCNN
BEVFormer
CIASSD
partA2
Bevfusion（ali）
BEVfusion（MIT HAN）
CIA-SSD
SA-SSD
IA-SSD
PillarNet
transfusion
AVOD	
MV3D														√
ComplexYOLO													√
YOLO3D														√
DETR3D



==========================PointPillars======================== 算法原理
==========================PointPillars======================== 算法原理
==========================PointPillars======================== 算法原理
	https://blog.csdn.net/qq_41366026/article/details/123006401
	预处理：pillar和voxel不一样，z轴就一个高度，pcdet里面是调用spconv生成的pillar，VoxelGeneratorWrapper，需要给一些参数
	给定每个pillar的大小  [0.16, 0.16, 4] 、给定点云的范围 [0, -39.68, -3, 69.12, 39.68, 1]
	给定每个点云的特征维度，这里是x，y，z，r 其中r是激光雷达反射强度
	给定每个pillar中最多能有多少个点 32，如果一个pillar中的点云数量超过32,那么就会随机采样，选取32个点；如果一个pillar中的点云数量少于32；那么会对这个pillar使用0样本填充。
	最多选取多少个pillar，因为生成的pillar中，很多都是没有点在里面的
	grid_size:（432，496，1）这样的     而voxel-size则是（0.16，0.16，4）这样的
	利用这个生成pillar的函数最后可以得到几个输出：voxels代表了每个生成的pillar数据，维度是[M,32,4]、coordinates代表了每个生成的pillar所在的zyx轴坐标，维度是[M,3],其中z恒为0、num_points代表了每个生成的pillar中有多少个有效的点维度是[m,]，因为不满32会被0填充
	得到了这些值之后还会对pillar做一个简单的数据增强（x,y,z,i,xc,yc,zc,xp,yp,zp）十维的，c下标是每个点到所有点的平均值的偏移，p下标就是到中心点的偏移   然后就得到了（特征维度，多少个pillar，每个pillar多少个点）
	在经过映射后，就获得了一个（D，P，N）的张量；接下来这里使用了一个简化版的pointnet网络对点云的数据进行特征提取（即将这些点通过MLP升维，然后跟着BN层和Relu激活层），得到一个（C，P，N）形状的张量，之后再使用maxpool操作提取每个pillar中最能代表该pillar的点。那么输出会变成（C，P，N）->（C，P）；在经过上述操作编码后的点，需要重新放回到原来对应pillar的x,y位置上生成伪图象数据。（也就是在点那个维度做了maxpooling）（64通道，多少个pillar，1）
	原始论文在提取特征的时候用的卷积，但是pcdet里面用的是linear层，用了BN1D（BN有些细节得看看）
	最终就是每一个pillar抽象出一个64维的特征，（pillar个数，64）

	************************scatterBEV**************************:
	对应到论文中就是stacked pillars，将生成的pillar按照坐标索引还原到原空间中   （432，496，1）是他的一个相对的坐标，而非实际的大小
	怎么理解这个算子呢：首先我们拿到之前生成grid的坐标数据coord（batch_id,z,y,x）
	然后我们拿到已经提取好的pillar的数据 （pillars个数，64）；
	那么关键就是  indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
	这个计算了，1维度就是z，全是0，后面是y乘上每一行的个数，就是相当于一行行的跳，加上行的偏移；
	我们需要明确一点，pillar特征和coord是对应的（起码在shape的第一个维度是一样的），
	因为在原始的grid里面也许有一些空的pilalr是不要填充特征的。
	需要计算到当前的pillar的索引，nx就是行多少个元素（列），所以索引2就是当前的行数，
	当前行*每行多少个 + 偏移到哪一列就是当前pillar的索引，其实需要注意的就是索引横着是x，竖着是y，这个和cuda里面那个也容易混淆。
	在总结下：
	this_coords[:, 2]  就是y 哪一行
	self.nx 每一行多少个数
	this_coords[:, 3] 在那一列，就是x
	这么加起来后就得到了每一个pillar在原始grid里面的索引了，并且这个索引应该是直接过滤掉了空的
	而spatial_feature 则是那种被拉平的（64，428544）那种的
	spatial_feature[:, indices] = pillars 通过之前的索引，知道了哪些pillar需要填充的，这样一来每一个batch里面的spatial_feature都被填充了
	然后去view成（1，64，496，864）这样的，也是y在前面，x在后面返回，此时这个数据就类似一个伪图像，（1，3，640，640）后面去走2D的backbone就可以了
	那么很明显我们看到的就是一个没有用高度信息，bev视角的检测了，坐标的z都是0，xy的坐标可能还有一些，
	反正就是因为z只有一个常量值，但是之前处理的时候，是对z的均值和中心值也做了一些操作的，也许这个里面就存在了Z的概念了。

	**************************backbone***********************
	经过上面的操作之后，得到了伪图像，只有在有pillar非空的坐标处有提取的点云数据，其余地方都是0数据，
	所以得到的一个（batch_size，64, 432, 496）的张量还是很稀疏的。这就是点云的稀疏性，虽然grid很大，但是从实际的图上看，非常的稀疏，很多地方都没有值。
	backbone走完后，到了FPN，下采样两次上采样两次然后拼接，得到了（batch_size, 128, 248, 216）*3 ；cocnat之后得到了（batch_size, 384, 248, 216）

	**********************single—head*************************
	openpcdet的实现中，直接使用了一个网络训练车、人、自行车三个类别；没有像原论文中对车、人使用两种不同的网络结构。
	kitti上面是一共有三个类别的先验框，每个先验框都有两个方向分别是BEV视角下的0度和90度，每个类别的先验证只有一种尺度信息；也就是六个anchor；
	然后就是target assign了，也是2D IOU匹配。实在bev视角下面，pcdet的源码直接用的标准IOU，连rotateIOU都没用，而是做了一个角度的规范化，
	使得角度都在-45-45之间，然后更好的匹配这个怎么理解？
	为什么不考虑高度信息：
	（1）、因为在kitti数据集中所有的物体都是在三维空间的同一个平面中的，没有车在车上面的一个情况。
	（2）、所有类别物体之间的高度差别不是很大，直接使用SmoothL1回归就可以得到很好的结果。
	匹配规则：
	车匹配iou阈值大于等于0.65为正样本，小于0.45为负样本，中间的不计算损失。
	人匹配iou阈值大于等于0.5为正样本，小于0.35为负样本，中间的不计算损失。
	自行车匹配iou阈值大于等于0.5为正样本，小于0.35为负样本，中间的不计算损失。
	输出是，(x, y, z, w, l, h, θ)，这里他解码的时候说的是预测了左上角的哪个角点相对的偏移，为什么？直接预测中心点的偏移不是更好么？
	因为在角度预测时候不可以区分两个完全相反的box，所以PiontPillars的检测头中还添加了对一个anchor的方向预测；这里使用了一个基于softmax的方向分类box的两个朝向信息。
	匹配代码比较复杂，可以看注释，不过这部分的研究意义不是很大，起码当前来说是的，章老师对SimOTA也加入了centerpoint这样的算法，
	但是实际的效果好像不是很好，需要研究cp这个网络
	********loss计算**********
	上述提到了(x, y, z, w, l, h, θ)，七个值的loss，loss和fastercnn的编码类似，残差加log，角度用的sin函数编码了。所以在loss里面可能出现八个值，把θ分解了
	分类损失就是focal loss
	softmax方向损失
	pcdet里，分别是anchor生成，anchorGT匹配，然后才是计算loss
	[(1，248，216，1，2，7），(1，248，216，1，2，7），(1，248，216，1，2，7）]  这个就是anchor，每一个类别两个anchor，
	和图像里面聚类得到的特征图的那种anchor不太类似，图像上尺度还可以不一样的
	这边由于是跟着类别走的，而且是刚体，没那么多的尺度的变化。
	匹配算法：
	至于匹配，逐类别和anchor匹配的，代码极其复杂，看注释
	有一点需要注意的是，这边的编码就是targets，而我之前分析yolov5那一套的代码的时候，他们还是匹配完了就是用相对中心点偏移和倍率作为targets，
	实际在loss计算做了编码，而这边就是直接先编码直接做loss
	在得到方向的target的时候，作者将角度先减小了45度，预测时候又加上45度了
	（说的呢就是因为大部分目标都集中在0度和180度，270度和90度，
      这样就会导致网络在一些物体的预测上面不停的摇摆。所以为了解决这个问题，
      将方向分类的角度判断减去45度再进行判断，
      这里减掉45度之后，在预测推理的时候，同样预测的角度解码之后
      也要减去45度再进行之后测nms等操作）
	最后对方向回归采用smooth L1 loss
	方向分类就是简单的交叉熵cross_entropy

	
	关于anchor生成需要理解一下，还有就是label assign
	先看下label assign：
		assign_targets完成对一帧点云数据中所有的类别和anchor的正负样本分配，
		assign_targets_single完成对一帧中每个类别的GT和anchor的正负样本分配。
		所以一个Batch样本中anchor与GT的匹配这里是逐帧逐类别进行的。与图像目标检测中稍有不同。图像目标检测就是按照特征图匹配，而不去考虑类别上面的匹配。
		直接看代码的注释，annotate 和我自己的
	关于loss，分类用了相对熵 = 信息熵 + 交叉熵， 且交叉熵是凸函数，求导时能够得到全局最优值，交叉熵损失的一种变形,具体推到参考上面的链接https://zhuanlan.zhihu.com/p/35709485
	box用了smoothL1
	

	
	
	

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★	
second 算法原理   SECOND(Sparsely Embedded CONvolutional Detection)
https://blog.csdn.net/qq_41366026/article/details/123323578?spm=1001.2014.3001.5502
	同时在VoxelNet的基础上改进了中间层的3D卷积，采用稀疏卷积来完成，提升了效率；
	同时解决了VoxelNet中角度预测中，因为物体完全反向和产生很大loss的情况；同时，SECOND还提出了GT_Aug的点云数据增强。
	Pointcloud -> voxel feature and coordinates -> voxel feature extractor -> sparse conv -> rpn -> cls + box + dir_cls
	注：VoxelNet中的点云特征提取VFE模块在作者最新的实现中已经被替换；因为原来的VFE操作速度太慢，并且对显存不友好。具体可以查看这个issue：
		https://github.com/traveller59/second.pytorch/issues/153
		VFE 很慢，需要太多的 gpu 内存。 在 KITTI 数据集中使用高分辨率稀疏卷积会更好一些。   注意这个高分辨率是什么意思
	所以整个second在openpcdet里面的实现可以分为如下几个方面：
		1、MeanVFE （voxel特征编码）
		2、VoxelBackBone8x （中间卷积层，此处为3D稀疏卷积）
		3、HeightCompression （Z轴方向压缩）
		4、BaseBEVBackbone （BEV视角下 2D卷积特征提取）
		5、AnchorHeadSingle （anchor分类和box预测）
	1.VFE：
		Point Cloud Grouping：
			在最先的SECOND中，将点云变成Voxel的方法和VoxelNet中一样，首先创建一个最大存储N个voxel的buffer，并迭代整个点云来分配他们所在的voxel，
			同时存储这个voxel在voxel坐标系中的坐标和每个voxel中有多少个点云数据
			在最新的是实现中，采用了稀疏卷积来进行完成 。      这边和稀疏卷积有什么关系？  
			经过对点云数据进行Grouping操作后得到三份数据：
				1、得到所有的voxel  shape为(N, 5 , 4) ; 5为每个voxel最大的点数，4为每个point的数据     （x，y，z，reflect intensity）
				2、得到每个voxel的位置坐标 shape（N， 3）
				3、得到每个voxel中有多少个非空点 shape （N）
					PCDET：区别：
						1.原文中分别对车、自行车和行人使用了不同的网络结构，PCDet仅使用一种结构训练三个类别。
						2.原论文中的每个voxel的长宽高为0.2，0.2，0.4且每个voxel中采样35个点，在PCDet的实现中每个voxel的长宽0.05米，
							高0.1米且每个voxel采样5个点；同时在Grouping的过程中，一个voxel中点的数量不足5个的话，用0填充至5个。
						3.N为非空voxel的最大个数，训练过程中N取16000，推理时取40000。
	2.mean VFE:
		在得到Voxel和每个Voxel对应的coordinate后，此处的VFE方式稍有变化，原因已写在上面的issue中
		在新的实现中，去掉了原来Stacked Voxel Feature Encoding，直接计算每个voxel内点的平均值，当成这个voxel的特征；
		大幅提高了计算的速度，并且也取得了不错的检测效果。得到voxel特征的维度变换为：(Batch*16000, 5, 4) -->  (Batch*16000, 4)
		16000是非空voxel，5代表点，4代表xyzi
		原来是每一个voxel里面最多5个点，然后做了取均值替代。
		这一步是不是就相当于之前已经做好的特征，（N，35，7）？ 还是（N，1，128）？ voxelnet还做了个index，将数据变成了（128，10，400，352）
		实际上新的做法，把体素弄得更小了，直接均值替代，而像pointpilalr算法还是用的很大的体素，0.16*0.16还得做MLP，pillarVFE
	3.VoxelBackBone8x
		这边的输入我认为就直接是结构化的稀疏的数据了； 可以类比voxelnet，他只是变成了128，10，400，352）走了3D卷积，pillar是（64，216，232）走2D卷积
		这边的输入就是(Batch * 16000, 4)     已经包括了mean的voxel特征编码
		在VoxelNet中，对voxel进行特征提取采取的是3D卷积的操作，但是3D卷积由于计算量太大，并且消耗的计算资源太多；作者对其进行了改进。
		首先稀疏卷积的概念最早由facebook开源且使用在2D手写数字识别上的,因为其特殊的映射规则,其卷积速度比普通的卷积快
		所以,作者在这里想到了用常规稀疏卷积的替代方法
		submanifold卷积将输出位置限制为在且仅当相应的输入位置处于活动状态时才处于活动状态。    关于其作用可以看我之前的笔记。
		这避免了太多的激活位置的产生，从而导致后续卷积层中速度的降低
		作者经过自己的改进,使用了新的稀疏卷积方法，详情可以看这个知乎   https://zhuanlan.zhihu.com/p/356892010
		就是3D卷积的改版，但是细节很多
		输入是[batch_size, 4, [41, 1600, 1408]]输出是[batch_size, 128, [2, 200, 176]]，之前的范围是[41, 1600, 1408]，
		至于这个41的原因也记录在源码了，就是为了最后得到2
	4.HeightCompression （Z轴方向压缩）
		由于前面VoxelBackBone8x得到的tensor是稀疏tensor，数据为：[batch_size, 128, [2, 200, 176]]
		这里需要将原来的稀疏数据转换为密集数据；同时将得到的密集数据在Z轴方向上进行堆叠，因为在KITTI数据集中，没有物体会在Z轴上重合；同时这样做的好处有：
			1.简化了网络检测头的设计难度
			2.增加了高度方向上的感受野
			3.加快了网络的训练、推理速度
		最终得到的BEV特征图为：(batch_size, 128*2, 200, 176) ，这样就可以将图片的检测思路运用进来了。
	5.BaseBEVBackbone
		在获得类图片的特征数据后，需要在对该特征在BEV的视角上进行特征提取。这里采用了和VoxelNet类是的网络结构；
		分别对特征图进行不同尺度的下采样然后再进行上采用后在通道维度进行拼接。
		SECOND中存在两个下采样分支结构，则对应存在两个反卷积结构：
		经过HeightCompression得到的BEV特征图是：(batch_size, 128*2, 200, 176)
		下采样分支一：(batch_size, 128*2, 200, 176) --> (batch,128, 200, 176)
		下采样分支二：(batch_size, 128*2, 200, 176) --> (batch,128, 200, 176)
		反卷积分支一：(batch, 128, 200, 176) --> (batch, 256, 200, 176)
		反卷积分支二：(batch, 256, 100, 88) --> (batch, 256, 200, 176)
		最终将结构在通道维度上进行拼接的特征图维度：(batch, 256 * 2, 200, 176)
	6.AnchorHeadSingle
		经过BaseBEVBackbone后得到的特征图为(batch, 256 * 2, 200, 176)；
		在SECOND中，作者提出了方向分类，将原来VoxelNet的两个预测头上增加了一个方向分类头，
		来解决角度训练过程中一个预测的结果与GTBox的方向相反导致大loss的情况。
		每个头分别采用了1*1的卷积来进行预测。
		其细节主要分为，anchor的生成，get-pred，label assign，loss compute
		anchor：
			由于在3D世界中，每个类别的物体大小相对固定，所以直接使用了基于KITTI数据集上每个类别的平均长宽高作为anchor大小，
			同时每个类别的anchor都有两个方向角为0度和90度。
			每个anchro都有被指定两个个one-hot向量，一个用于方向分类，一个用于类别分类；
			还被指定一个7维的向量用于anchor box的回归，分别是（x, y, z, l, w, h, θ）其中θ为PCDet坐标系下物体的朝向信息。
	7.label assign
		由于预测的时候，将不同类别的anchor堆叠在了一个点进行预测，所有进行Target assignment时候，
		要分类别进行Target assignment操作。这里与2D 的SSD或YOLO的匹配不同。
	8.loss 计算：
		注：使用较小的方向回归损失，可以防止网络在物体的方向分类上摇摆不定。
		分类focal loss
		smoothL1(SIN(Θ))
	后面基本就和pointpilalrs类似了。		
	
	稀疏卷积原理：  https://zhuanlan.zhihu.com/p/356892010
		就是voxelnet的中间卷积层，由conv3d替换为了spconv，
		将稀疏的输入特征通过gather操作获得密集的gather特征；
		然后使用GEMM对密集的gather特征进行卷积操作，获得密集的输出特征；
		通过预先构建的输入-输出索引规则矩阵，将密集的输出特征映射到稀疏的输出特征。
		这个输入-输出索引规则矩阵很明显就是稀疏卷积的关键所在了。
		
		先回顾一下自己之前的理解：
			首先是为什么不用普通的卷积：普通的卷积会从头到尾计算一遍，稀疏数据很多地方都是空的，产生很多的不必要的计算；
			容易提取到失真的特征，从稀疏的数据里面提取到了稠密的特征。
			子流行膨胀：膨胀：稀疏卷积可以让空的地方不参与特征的计算，碰到有值的地方就开始计算了，但是会有空的地方就被提取的稠密特征，
			然后就会在特征图上膨胀了，经过越多的卷积就会越来越膨胀。
			只需要存储特征图中有值的位置的。保存索引，有值的原始的位置需要保存好，不然找不到从哪来的了，
			所以没必要用一块完整的内存无差别的存储特征图中的全部值；
			Nin个体素，卷积核为3*3*3，rule-book就是一个27*Nin*2的一个矩阵，
			27表示卷积核在滑动过程中卷积核的每一个权重都会产生一个input-output的映射关系。（挺难理解）
			Nin表示每一个输入的体素都会计算输出的位置
			2表示，记录输入的体素在输出数据中的索引位置，和输出结果结果在输出数据中的位置。成对出现。
			子流形卷积：稀疏卷积由于子流形膨胀会导致数据丧失稀疏性，子流形卷积就是保持了稀疏性，
			但是只用子流形卷积感受野就会被限制，所以需要配合maxpooling 和 stride=2的稀疏卷积一起用
			保持稀疏性的同时增大感受野。
			总结就是，先把非空的提取出来，然后走3D卷积，最后用rule-book塞回去
			稀疏卷积的stride一般是2，是为了增大感受野降低分辨率用的。在经过子流形卷积的时候他的分辨率是保持不变的。
			也就是说在经过稀疏卷积之前，子流形卷积中的输入输出的位置是保持一致的。经过一次稀疏卷积，rule-book就变了，需要重新计算了。
		
		上面也只是大概理解了稀疏卷积，下面看一个大佬的。				https://zhuanlan.zhihu.com/p/382365889
			
				
				
										




★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
voxelnet 算法原理        https://blog.csdn.net/qq_41366026/article/details/123175074
	三维世界中每一定空间大小划分成一个格子，然后使用pointnet网络对这个小格子的数据进行特征提取；
	并用这个提取出来的特征来代表这个小格子，并放回到3D的空间中。
	这样无序的点云数据就变成了一个个的高维特征数据并且这些数据在三维空间中也变得有序了，之后就可以使用三维卷积来抽取这些三维的voxel数据了，那么图像检测的思路就可以应用在这个特征图上面了。
	Feature Learning Network、Convolutional middle layers、Region proposal network。
	
	详细原理：
		1.Voxel 划分（分配每个voxel到3D空间中对应位置）
			论文中X,Y,Z分别是0 — 70.4；-40 — 40；-3 — 1，Vx，Vy，Vz分别是0.4，0.2，0.2。单位均为米。超出X, Y, Z范围外的点直接裁剪掉，因为点云空间中远处的物体产生的点太过于稀疏，不能保证识别结果的可靠性
		2.Grouping（将每个点分配给对应的Voxel）和Sampling（voxel中点云的采样）
			每个voxel中点的数量都不一样，甚至大部分的voxel中都是没有点存在的。
			同时，一个高精度的激光雷达点云会包含10万以上的点，如果直接对所有的这些点进行处理的话会导致：1、会产生极大的计算和内存消耗，
			还会让模型不能很好的学习，因为点的密度之间的差距太大，可能会导致检测偏差。
			因此在Grouping操作之后，需要对每个非空的voxel中随机采样T个点。（不足补0，超出的仅采样T个点）。
			经过Grouping 后就可以将数据表示为（N，T，C），其中N为非空的voxel了的个数，T为每个voxel中点个个数，C表示点的特征。计算出输入的tensor数据的shape是（N，35，7）。
		3.VFE堆叠
			就是对结构化的voxel如何编码了。
			每一个非空的Voxel都是一个点集，定义为 V = {pi = [xi , yi , zi , ri]T ∈ R^4}i=1...t ; 其中t小于等于35。
			首先，先对voxel中的每个点进行数据增强操作，先计算出每个voxel的所有点的平均值，计为（V_Cx，V_Cy，V_Cz），
			然后将每个点的xi，yi，zi减去对应轴上的平均值，得到每个点到自身voxel中心的偏移量（xi_offset，yi_offset，zi_offset）。
			再将得到的偏移数据和原始的数据拼接在一起得到网络的输入数据V = {pi = [xi , yi , zi , ri, xi_offset，yi_offset，zi_offset]T ∈ R^7}i=1...t。    七维的一个向量
			每一个voxel都是这样的
			（N,35，7）
			接着就是用PointNet提出的方法，将每个voxel中的点通过全连接层转化到高维空间（每个全连接层包含了FC, RELU, BN）。维度也从（N，35，7）变成了（N，35，C1）。
			然后在这个特征中，取出特征值最大的点（Element-wise Maxpool）得到一个voxel的聚合特征（Locally Aggreated Feature），
			可以用这个聚合特征来编码这个voxel包含的表面形状信息。这也是PointNet中所提到的。获得每个voxel的聚合特征后,再用该特征来加强经过FC后的高维特征；
			将聚合特征拼接到每一个高维点云特征中（Point-wise Concatenate）;得到（N，35，2*C1）。   那看来max pooling是在 35 这个维度做的
			一个C1原始特征一个C1是max pooli之后用一个点的特征代表全部的
			作者把上述的这个特征提取的模块称之为VFE（Voxel Feature Encoding），
			这样每个VFE模块都只仅仅包含了一个（C_in，C_out/2）的参数矩阵。
			每个voxel经过VFE输出的特征都包含了voxel内每个点的高维特征和经过聚合的局部特征，那么只需要堆叠VFE模块就可以实现voxel中每个点的信息和局部聚合点信息的交互，
			使得最终得到的特征能够描述这个voxel的形状信息。
			接下来就是要堆叠这个VFE模块得到完整的Stacked Voxel Feature Encoding。
			注：每个VFE模块中FC的参数共享的。原论文的实现中一共堆叠了两个VFE模块，其中第一个VFE模块将维度从输入的7维度升高到了32，第二个VFE模块将数据的维度从32升高到了128。
			经过Stacked Voxel Feature Encoding后，可以得到一个（N，35，128）的特征，
			然后为了得到这个voxel的最终特征表达。需要对这个特征再进行一个FC操作来融合之前点特征和聚合特征，这个FC操作的输入输出保持不变。
			即得到的tensor还是（N，35，128），之后进行Element-wise Maxpool来提取每个voxel中最具体代表性的点，并用这个点来代表这个voxel，即（N，35，128）--> （N，1，128）
		4.parse Tensor Representation（特征提取后稀疏特征的表示）
			在前面的Stacked Voxel Feature Encoding 的处理中，都是对非空的voxel进行处理，代码里面是有mask的
			这些voxel仅仅对应3D空间中很小的一部分空间
			这里需要将得到的N个非空的voxel特征（N,1，128）?重新映射回来源的3D空间中，表示成一个稀疏的4D张量，
			（C，Z'，Y'，X'）--> (128, 10, 400, 352)  也就是在非空的体素里面插入128维的特征
			这种稀疏的表示方法极大的减少了内存消耗和反向传播中的计算消耗。同时也是VoxelNet为了效率而实现的重要步骤。
			前面我写的voxel采样中，如果一个voxel中没有T个点，就直接补0直到点的数量达到35个，如果超出35个点就随机采样35个点。但是在原论文中的具体实现如下。
			原作者为Stacked Voxel Feature Encoding的处理设计了一个高效实现
			由于每个voxel中包含的点的个数都是不一样的，所以这里作者将点云数据转换成了一种密集的数据结构，使得后面的Stacked Voxel Feature Encoding可以在所有的点和voxel的特征上并行处理。
				1、首先创建一个K*T*7的tensor（voxel input feature buffer）用来存储每个点或者中间的voxel特征数据，其中K是最大的非空voxel数量，T是每个voxel中最大的点数，7是每个点的编码特征。所有的点都是被随机处理的。（10000，35，7）
				2、遍历整个点云数据，如果一个点对应的voxel在voxel coordinate buffer中，并且与之对应的voxel input feature buffer中点的数量少于T，直接将这个点插入Voxel Input Feature Buffer中；否则直接抛弃这个点。如果一个点对应的voxel不在voxel coordinate buffer，需要在voxel coordinate buffer中直接使用这个voxel的坐标初始化这个voxel，并存储这个点到Voxel Input Feature Buffer中。这整个操作都是用哈希表完成，因此时间复杂度都是O（1）。整个Voxel Input Feature Buffer和voxel coordinate buffer的创建只需要遍历一次点云数据就可以，时间复杂度只有O（N），同时为了进一步提高内存和计算资源，对voxel中点的数量少于m数量的voxel直接忽略改voxel的创建。  空的直接处理掉，我们只要非空的				（这一步可以看我之前的PPT笔记，实际上就是看看voxel buffer初始化了没，然后塞点进去，满了就不要把，缺了就补0，我的感觉就是和sampling很像，然后再去做VFE）
				3、再创建完Voxel Input Feature Buffer和voxel coordinate buffer后Stacked Voxel Feature Encoding就可以直接在点的基础上或者voxel的基础上进行并行计算。
				再经过VFE模块的concat操作后，就将之前为空的点的特征置0，保证了voxel的特征和点的特征的一致性。
				最后，使用存储在voxel coordinate buffer的内容恢复出稀疏的4D张量数据，完成后续的中间特征提取和RPN层。
				讲白了就很像sactter操作，只不过sactter没有高度。
		5.中间的3D卷积
			在经过了Stacked Voxel Feature Encoding层的特征提取和稀疏张量的表示之后，就可以使用3维卷积来进行整体之间的特征提取了，因为在前的每个VFE中提取反应了每个voxel的信息，
			这里使用3维卷积来聚合voxel之间的局部关系，扩大感受野获取更丰富的形状信息，给后续的RPN层来预测结果。    注意是voxel之间，类似2D卷积，一块ROI的感觉；
			三维卷积可以用ConvMD(cin, cout, k, s, p)来表示，cin和cout是输入和输出的通道数，k表示三维卷积的kernel大小，s表示步长，p表示padding参数。每个三维卷积后都接一个BN层和一个Relu激活函数。
			注：原文中在Convolutional middle layers中分别使用了三个三维卷积，卷积设置分别为  输入第一个卷积torch.Size([1, 128, 10, 400, 352])-》torch.Size([1, 64, 5, 400, 352])
				Conv3D(128, 64, 3, (2,1,1), (1,1,1)),
				Conv3D(64, 64, 3, (1,1,1), (0,1,1))，
				Conv3D(64, 64, 3, (2,1,1), (1,1,1))     这个conv3d的卷积核和padding有空研究研究卷积的计算。
				最终得到的tensor shape是（64，2，400，352）。其中64为通道数。
			经过Convolutional middle layers后，需要将数据整理成RPN网络需要的特整体，直接将Convolutional middle layers得到的tensor 在高度上进行reshape 变成（64 * 2，400，352）
			那么每个维度就变成了 C、Y、X。 （64，2，400，352）->(64*2,400,352)的原因就是这个。
			这样操作的原因是因为KITTI等数据集的检测任务中，物体没有在3D空间中的高度方向进行堆叠，没有出现一个车在另一个车的上方这种情况。同时这样也大大减少了网络后期RPN层设计难度和后期anchor的数量。
		6.RPN的设计：
			实际上可能更加接近了YOLO head了；实际上就是做了不同尺度的特征的融合，这里每一层卷积都是二维的卷积操作，每个卷积后面都接一个BN和RELU层。
			然后最后输出了# torch.Size([1, 768, 200, 176])
		7.anchor的设计：
			VoxelNet中，只使用了一个anchor的尺度，不像FrCNN中的9个anchor。
			每个物体都是有朝向信息的。所以VoxelNet为每个anchor加入了两个朝向信息，分别是0度和90度（激光雷达坐标系）。
			注：由于在原论文中作者分别为车、行人、自行车设计了不同的anchor尺度，并且行人和自行车有自己单独的网络结构（仅仅在Convolutional middle layers的设置上有区别）。
			实际上这个anchor的设计是有问题的，anchor底面中心是-1.78 -0.6，这个-0.6就是使得anchor就飘在天上，虽然bias最后也能学出来，但是就很离谱这个设置。
		8.样本匹配：
			每个类别的先验证只有一种尺度信息；分别是车 [3.9, 1.6, 1.56]，anchor的中心在-1米、人[0.8, 0.6, 1.73]，anchor的中心在-0.6米、自行车[1.76, 0.6, 1.73]，anchor的中心在-0.6米（单位：米）。
			在anchor匹配GT的过程中，使用的是2D IOU匹配方式，直接从生成的特征图也就是BEV视角进行匹配；不需要考虑高度信息。
			1、因为在kitti数据集中所有的物体都是在三维空间的同一个平面中的，没有车在车上面的一个情况。
			2、所有类别物体之间的高度差别不是很大，直接使用SmoothL1回归就可以得到很好的结果。   
				车匹配iou阈值大于等于0.6为正样本，小于0.45为负样本，中间的不计算损失。
				人匹配iou阈值大于等于0.5为正样本，小于0.35为负样本，中间的不计算损失。
				自行车匹配iou阈值大于等于0.5为正样本，小于0.35为负样本，中间的不计算损失。
 		9.损失函数：
			直接看encode的公式，采用除法归一化和log编码。还有底面对角线的长度作为归一化的除法。此时的box的损失还是用最后的回归损失采用了SmoothL1函数。方向角也是直接减去了，没考虑sin cos
			但是总的损失函数还包含了对每个anchor的分类预测，
			仅仅需要对正样本anchor回归计算loss
			同时背景分类和类别分类都采用了BCE损失函数；
			1/Npos和1/Nneg用来normalize各项的分类损失
			α, β为两个平衡系数，在论文中分别是1.5和1。也就是正负样本的分类的损失。
		10.点云数据增强
			1、由于点云的标注数据中，一个GTbox已经标注出来这个box中有哪些点，所以可以同时移动或者旋转这些点来创造大量的变化数据；在移动这些点后需要进行以下碰撞检测，删粗掉经过变换后这个GTbox和其它的GTbox混在一起的，不可能出现在现实中出现的情况。
			2、对所有的GTbox进行放大或者缩小，放大或者缩小的尺度在 [ 0.95, 1.05]之间；引入缩放，可以使得网络在检测不同大小的物体上有更好的泛化性能，这一点在图像中很常见。
			3、对所有的GTbox进行随机的进行旋转操作，角度在从[ -45, 45]均匀分布中抽取，旋转物体的偏航角可以模仿物体在现实中转了个弯的情况。
		实际在评价的时候，kitti使用了不同的IOU的阈值进行的评价，而且分了难度。在nus里面就不是这么苛刻了。
				
				
				


★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
centerpoint 算法原理    https://zhuanlan.zhihu.com/p/357068924 这篇文章是3D视觉工坊写的可以看看
						https://zhuanlan.zhihu.com/p/524608535
	nus数据集的点云为什么5个通道应该是 （z, y, x ,ring_index, intensity）是这样的？0000000
	
	



★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
pointrcnn算法原理    https://blog.csdn.net/qq_41366026/article/details/123214165?spm=1001.2014.3001.5502
	1.前言	
		将点云划分成voxel来进行检测，典型的模型有VoxelNet、SECOND等；作然而本文的作者史博士提出这种方法会出现量化造成的信息损失。
		将点云投影到前视角或者鸟瞰图来来进行检测，包括MV3D、PIXOR、AVOD等检测模型；同时这类模型也会出现量化损失。
		将点云直接生成伪图片，然后使用2D的方式来进行处理，这主要是PointPillar。
		本文PointRCNN提出的方法，是一篇比较新颖的点云检测方法，与此前的检测模型不同，它直接根据点云分割的结果来产生候选框，并根据这些候选框的内部数据和之前的分割特征来完成物体的准确定位和分类。
		第一阶段：
		PointNet2MSG：对原始的点云进行编码解码操作（PointNet++）
		PointHeadBox：对经过PointNet++的点云进行分类操作和box的预测
		第二阶段：
		PointRCNNHead：对每个roi进行精调
	2.网络解析：
		如果直接将2维的检测模型中二阶段直接迁移到3D的检测任务上的话，是比较困难的。
		因为3D检测任务拥有更大的搜索空间，并且点云数据并不是密集数据。
		如果使用这样的方式，像AVOD那样，需要放置20-100K个anchor在3D空间中，这种做法很蠢；
		如果像F-PointNet那样，采用2D的检测框架首先在图片上来生成物体的proposal，然后在这个视锥内来检测3D的物体的话，确实可以极大的减少3D空间中的搜索范围，
		但是也会导致很多只有在3D空间下才能看到的物体因为遮挡等原因无法在图像中得以显示，造成模型的recall不高，同时该方法也极大的依赖于2D检测器的性能。
		因此PointRCNN首次直接在点云数据上分割mask
		这里能直接预测mask的原因主要是3D的标注框中，可以清晰的标注出来一个GTBox中有哪些点云数据，而且在自然的3D世界中，不会出现像图片里面物体重叠的情况，因此每个点云中属于前景的mask就得到了
		然后在第二阶段中对每个proposal中第一阶段学习到的特征和处在proposal中的原始点云数据进行池化操作。
		通过将坐标系转换到canonical coordinate system（CCS）坐标系中来进一步的优化得到box和cls的结果。
	3、自下而上的3D建议框生成：
		3.1. 特征提取网络（PointNet2MSG）：
			在PointRCNN中，首先采用Multi-Scale-Grounping 的PointNet++网络来完成对点云中前背景点的分割和proposal的生成。
			因为标注的数据中每个GTbox就可以清晰的标注出点云中哪些属于的前景点，哪些属于背景点
			显然背景点的数量肯定是比前景点的数量要多得多的。所以作者在这里采用了focal loss来解决类别不平衡的问题。
			原来的PointNet++网络中并没有box的regression，所以作者在这里增加了一个回归头用于前给景点生成proposal。
			其中这里前背景点的分割和前景点的proposal生成是同步完成的，只不过在代码实现中，只取前景点的box
			请注意，这里的proposal的box生成虽然是直接从前景点的特征来生成的，但是背景点也提供了丰富的感受野信息，
			因为Point Cloud Encoder和Point Cloud Decoder网络在已经将点云数据进行了融合（比如PointNet++中，Decoder根据距离权重插值的方式，来反向传播点的信息
			（这边没太理解？）得看pointnet的理解了  pointnet++
			经过该上述部分处理后，得到网络中所有的前景点和前景点生成的proposal。这样就解决了在3D空间中设置大量anchor的愚蠢操作了，也就减少了proposal的搜索范围。
			实际的实现流程：
				其中在实现的过程中，每针点云数据随机采样16384个点（如果一帧点云中点的数量少于16384，则重复采样至16384）
				Point Cloud Network采用PointNet++（MSG），同时作者也表明采用其他网络也是可以的，比如PointShift、PointNet、VoxelNet with sparse conv。
				PointNet++ 参数设置如下：
					Encoder为4个Set-Abstraction层，每层最远点采样的个数分别是4096，1024，256，64。
					每一层采样不同的中心点的个数，FPS，然后基于每个中心点不同的半径进行ball query，做了SA pointnet后，concat
					最远点采样的点数
				    NPOINTS: [4096, 1024, 256, 64]
				    # BallQuery的半径
				    RADIUS: [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
				    # BallQuery内半径内最大采样点数（MSG）
				    NSAMPLE: [[16, 32], [16, 32], [16, 32], [16, 32]]
				    # MLPS的维度变换
				    # 其中[16, 16, 32]表示第一个半径和采样点下的维度变换，
				    [32, 32, 64]表示第二个半径和采样点下的维度变换，以下依次类推
				    MLPS: [[[16, 16, 32], [32, 32, 64]],
				           [[64, 64, 128], [64, 96, 128]],
				           [[128, 196, 256], [128, 196, 256]],
				           [[256, 256, 512], [256, 384, 512]]]
					经过分割后得到了：（1，16384，128），这个16384是怎么来的？貌似是随机采样的。
					插值：ThreeInterpolate， # 知道了三个最近点的索引和当前自己的特征，通过距离插值来计算这三个未知点的特征
		3.2 Bin-based 3D Bbox generation（论文）
			在对点进行box生成的时候，只需要对前景点的box进行回归操作。虽然没有对背景点进行回归操作，但是在PointNet++的反向回传中，背景点也为这些前景点提供了丰富的感受野信息。
			用bin分类：那么这设置的话到GT的偏移就变成了由第N个bin的分类问题和第N个bin中residual regression的回归问题。论文中作者说这种方式要比直接使用L1损失函数计算的回归进度更高。
			对于高度上的预测，直接使用了SmoothL1来回归之间的残差，因为高度上的变化没有这么大。
			同时，对于角度的预测，也采用了基于bin的方式，将2pi分成n个bins，并分类GT在哪个bin中，和对应分类bin中的residual regression回归。
			只有前景点才去做预测
			*****在推理阶段，这个bin-base的预测中，只需要找到拥有最高预测置信度的bin，并加上残差的预测结果就可以。其它的参数只需要将预测的数值加上初始数值就可以。
			代码中所有的box回归（第一阶段的box生成和第二阶段的ROI优化）都采用了bin-based的方法。该方法的内容实现与原论文代码仓库中的内容相同，但是在OpenPCDet中，史帅的实现并没有采用基于bin-base的方法，而是直接使用了smoothL1来进行预测；同时角度的预测也从bin-based的方式变成了residual-cos-based的方法。
			就是原始的论文里面，xy+yaw是多个bin的，lwhz 是直接回归
			PCdet的代码实现：
				经过PointNet++后，网络得到的point feature输出为（batch * 16384， 128），接下来就需要对每个点都进行分类和回归操作，前景点分割
				取出（batch * 16384， 128）做MLP BN1d relu 让 128->256->3  或者 到8box reg
				然后就是label assign：
					对每个点进行分类后可以得到 point_cls_preds_max，维度是(batch * 16384, num_class)，并取出每个类别的最大值并进行sigmoid激活，得到 point_cls_scores，维度是(batch * 16384, )。然后进行target assignment。
					target assignment的时候需要对每个GT的长宽高都延长0.2米，并将这些处于边缘0.2米的点的target设置为-1，不计算分类损失。原因是为了提高点云分割的健壮性，因为，3D的GTBox会有小的变化。
					要注意是不计算分类的损失。
					GT在label assign的时候，被预处理成x, y, z, l, w, h, heading, class_id 也就是8个维度
					enlarge box：以防人工标注漏掉的点。点云不像图像遮挡很难区分，正常一个物体的周围是有一些空间的，略微的放大一些box也是没问题的
					# 在训练的过程中，需要忽略掉点云中离GTBox较近的点，因为3D的GTBox也会有扰动，
					# 所以这里通过将每一个GT_box都在x，y，z方向上扩大0.2米，
					# 来检测哪些点是属于扩大后才有的点，增强点云分割的健壮性
					assign_stack_targets函数完成了一批数据中所有点的前背景分配，并为每个前景点分配了对应的类别和box的7个回归参数，xyzlwhθ
					函数用于完成一批数据中点云的前背景分配和前景点云的每个类别的平均anchor大小和GTbox编码操作
					# 得到一批数据中batch_size的大小，是以方便逐帧完成target assign
					前景点的box 怎么匹配assign？又没有IOU之类的。
					是不是因为现在是对（16384，3）操作的，所以和GT的关系可以很清楚的知道。这样子类别的信息也可以赋予前景点了
					就是此时应该已经对每个点进行了分类了。
					讲白了就是分类过的点，结合位置信息对每个前景点背景点分配cls，就是得到了mask 表明该帧中的哪些点属于前景点，哪些点属于背景点;得到属于前景点的mask
					enlarge的设置为-1，不考虑loss的计算。
					取出前景点
					# 并为这些点分配对应的GT_box shape (num_of_gt_match_by_points, 8)
            		# 8个维度分别是x, y, z, l, w, h, heading, class_id
					为什么准确到具体的类别？
					反正就是对前景点赋予类别信息，然后赋予BOX，然后此处编码为(Δx, Δy, Δz, dx, dy, dz, cos(heading), sin(heading)) 8个 box参数
					gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
					这句代码就是赋值
					赋值完了就是编码了。
					★★★★★★★此处我有个疑问，这个类别到底是具体的类别还是前景背景的区分，如果是后者那么就是mask的。
					这句话给了解答：            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
					其中GTbox和anchor的编码操作中使用了数据集上每个类别的平均长宽高为anchor的大小来完成编码。
					这句话的意思就是需要编码需要残差作为target
					同时，角度也不是当年那个什么bin based了，这里对角度的编码采用了residual-cos-based，用角度的 torch.cos(rg), torch.sin(rg)来编码GT的heading数值。
					# 每个gt_box的长宽高不得小于 1*10^-5，这里限制了一下
					根据每个点的类别索引，来为每个点生成对应类别的anchor大小 这个anchor来自于数据集中该类别的平均长宽高
			一阶段的proposal的生成：
				根据point_coords，point_cls_preds和point_box_preds来生成前景点的proposal。
				此时是根据pred的结果操作。还做了解码。因为训练的label做了编码。
				然后第一阶段生成的预测结果放入前向传播字典
		4.点云区域池化（根据proposal选取ROI）：
			 在获得了3DBbox的proposal之后，重点变聚焦在如何优化box的位置和朝向。为了学习到更加细致的proposal特征，PointRCNN中通过池化的方式，将每个proposal中的点的特征和内部的点进行池化操作。
			4.1. ROI获取
				训练阶段保留类别置信度最高的512个proposal，NMS_thresh为0.8；
        		测试阶段保留类别置信度最高的100个proposal，NMS_thresh为0.85
				注意：
				1、NMS都是无类别NMS，不考虑不同类别的物体会出现在3维空间中同一个地方
				2、原论文在训练时是300个proposal  代码好像是512？
				3、原论文中使用的是基于BEV视角的oriented IOU，这里是3D IOU  
				代码：pcdet/models/roi_heads/roi_head_template.py
					用0初始化所有的rois          shape : （batch, 512, 7） 训练时为512个roi，测试时为100个roi
					用0初始化所有的roi_score     shape : （batch, 512） 训练时为512个roi，测试时为100个roi
					用0初始化所有的roi_labels    shape : （batch, 512） 训练时为512个roi，测试时为100个roi
					逐帧计算rois
					得到当前帧点的box预测结果和对应的cls预测结果 box ：（16384, 7）; cls ：（16384, 3）
					取出每个点类别预测的最大数值和最大数值所对应的索引 cur_roi_scores: (16384, )
					进行无类别的nms操作  selected为经过NMS操作后被留下来的box的索引，selected_scores为被留下来box的最大类别预测分数
					# 从所有预测结果中选取经过nms操作后得到的box存入roi中
				    rois[index, :len(selected), :] = box_preds[selected]
				    # 从所有预测结果中选取经过nms操作后得到的box对应类别分数存入roi中
				    roi_scores[index, :len(selected)] = cur_roi_scores[selected]
				    # 从所有预测结果中选取经过nms操作后得到的box对应类别存入roi中
				    roi_labels[index, :len(selected)] = cur_roi_labels[selected]
					可以看出这种ROI的操作都是从pred里面取值的。
					将生成的proposal放入字典中  shape (batch, num_of_roi, 7)
					感觉貌似很么pooling的操作，而是就是nms得到了一些3Dbox而已
					最终得到的结果是：
						rois：shape (batch, num_of_roi, 7)包含了每个roi在3D空间的位置和大小
						roi_scores：shape (batch, num_of_roi)包含了每个roi置信度分数
						roi_labels：shape (batch, num_of_roi)包含了每个roi对应的类别
				4.2. ROI与GT的target assignment
					4.2.1. ROI采样（样本均衡）
						给PointRCNNHead网络进行学习时候，需要完成roi之间的类别均衡。前面提出了512个roi。但是只会采样128个ROI给PointRCNNHead网络进行学习。其中一共要采样3种不同类型的ROI
						简单的前景样本：ROI与GT的3D IOU大于0.55（采样64个，不够有多少个用多少个）
						简单的背景样本：ROI与GT的3D IOU小于0.1
						困难的背景样本：ROI与GT的3D IOU大于0.1小于0.55
						注：前景如果采样64个，背景也采样64个；如果前景没有64个，那么采样背景到总共128个ROI。
							同时难背景的在所有背景中的采样比为 0.8。 不够用简单背景补充。
						pcdet/models/roi_heads/target_assigner/proposal_target_layer.py
						batch_gt_of_rois （batch， 128, 8） GTBox为7个box的参数和1个类别
						batch_gt_of_rois （batch， 128）  ROI和GT的最大iou
						batch_roi_scores （batch， 128） ROI预测类别置信度
						batch_roi_labels （batch， 128） ROI预测的类别
						从GT中取出结果，因为之前GT中以一个batch中最多的GT数量为准，
						其他不足的帧中，在最后填充0数据。这里消除填充的0数据，就是为了一个batch能一起矩阵加速计算。？
						进行iou匹配的时候，只有roi的预测类别与GT相同时才会匹配该区域的roi到该区域的GT上
							self.get_max_iou_with_same_class()函数完成了相同类别直接的最大 IOU匹配。
							max_overlaps, gt_assignment = self.get_max_iou_with_same_class
							其中max_overlaps包含了每个roi和GT的最大iou数值，gt_assignment得到了每个roi对应的GT索引
							max_overlaps(512, ) gt_assignment(512,)
							一般都roi和GT最大的IOU值和索引成对出现。GT的索引
							计算指定类别的roi和GT之间的3d_iou shape ： （num_of_class_specified_roi, num_of_class_specified_GT）
							                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)  # (M, N)
							取出每个roi与当前GT最大的iou数值和最大iou数值对应的GT索引
							将该类别最大iou的数值填充进max_overlaps中
							将该类别roi与GT拥有最大iou的GT索引填充入gt_assignment中
					self.subsample_rois()函数完成了从512个roi中采样出128个roi。
					pcdet/models/roi_heads/target_assigner/proposal_target_layer.py
					"""此处的背景采样不是意义上的背景，而是那些iou与GT小于0.55的roi，对这些roi进行采样"""
					将所有属于前景点的roi打乱，使用np.random.permutation()函数
					直接取前N个roi为前景，得到被选取的前景roi在所有roi中的索引
					背景采样，其中前景采样了64个，背景也采样64个，保持样本均衡，如果不够用负样本填充
					assignment完成可以得到采样后的128个roi和对应的参数，分别是：
					roi的box预测参数             batch_rois：（batch， 128, 7）
					roi对应GT的box               batch_gt_of_rois：（batch， 128, 8） 8为7个回归参数和1个类别
					roi和GT的最大iou            batch_roi_ious： （batch， 128）
					roi的类别预测置信度        batch_roi_scores：（batch， 128）
					roi的预测类别                   batch_roi_labels：（batch， 128）
				4.2.2. loss计算中分类和回归的mask生成
					在进行roi和GT的匹配过程中，需要将ROI中与GT的3D IOU大于0.6的roi认为是正样本，3D IOU在0.45到0.6之间的不计算损失，小于0.45的为负样本，供给后面的PointRCNNHead分类头学习。
					同时PointRCNNHead中的回归头只学习ROI和GT的3DIOU大于0.55的那部分ROI。
					得到需要计算回归损失的的roi的mask，其中iou大于0.55也就是在self.sample_rois_for_rcnn（）中定义为真正属于前景的roi
					对iou大于0.6的roi进行分类，忽略iou属于0.45到0.6之前的roi
					将iou属于0.45到0.6之前的roi的类别置-1，不计算loss
					'rois': batch_rois,                 roi的box的7个参数                    shape（batch， 128, 7）
					'gt_of_rois': batch_gt_of_rois,     roi对应的GTbox的8个参数，包含类别      shape（batch， 128, 8）
					'gt_iou_of_rois': batch_roi_ious,   roi个对应GTbox的最大iou数值           shape（batch， 128）
					'roi_scores': batch_roi_scores,     roi box的类别预测分数                shape（batch， 128）
					'roi_labels': batch_roi_labels,     roi box的类别预测结果              shape（batch， 128）
					'reg_valid_mask': reg_valid_mask,   需要计算回归损失的roi            shape（batch， 128）
					'rcnn_cls_labels': batch_cls_labels 需要计算分类损失的roi             shape（batch， 128）
					targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                        'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                        'reg_valid_mask': reg_valid_mask,
                        'rcnn_cls_labels': batch_cls_labels}
				4.2.3. target转换到CCS坐标系（Canonical Coordinate System）
					PointRCNNHead对proposal的box精调是在CCS坐标系中进行的，而刚刚被选中的的ROI对应的GT target是在OpenPCDet的点云坐标系中，因此需要将GT Box变换到每个ROI自身的CCS坐标系中，坐标系变换包括了坐标系的平移和旋转。
					同时这里在变换坐标系时候，需要注意，如果一个ROI和对应的GT的IOU大于0.55，那么这两个box的角度偏差只会在正负45度内。
					从计算图中拿出来gt_of_rois并放入targets_dict
					进行canonical transformation变换，需要roi的xyz在点云中的坐标位置转换到以自身中心为原点
					将heading的数值，由-pi-pi转到0-2pi中  弧度
					计算在经过canonical transformation变换后GT相对于以roi中心的x，y，z偏移量
					计算GT和roi的heading偏移量
					上面完成了点云坐标系中的GT到roi坐标系的xyz方向的偏移，下面完成点云坐标系中的GT到roi坐标系的角度旋转，
					其中点云坐标系，x向前，y向右，z向上；roi坐标系中，x朝车头方向，y与x垂直，z向上		
					在3D的iou计算中，如果两个box的iou大于0.55，那么他们的角度偏差只会在-45度到45度之间
					最后也是做成了label了，计算了偏移。
					平移就直接加减就可以，旋转是调用这个common_utils.rotate_points_along_z
				5. Canonical 3D建议框精调网络（PointRCNNHead）：
					根据前面提出的proposal，为了学习细致的学习每个proposal的特征信息，并更好的优化box的回归精确度；
					PointRCNN将每个proposal中的3D点和这些点在经过PointNet++后对应的特征进行池化操作后并转换到CCS坐标系下进行学习。
					ROI POOL前融合：
						1、在对proposal进行池化的过程中，由于转换到了以每个roi自己中心做原点，那么不可避免的就损失了该点在点云中的深度信息，因此在池化之前，先为每个点加上了自己的深度信息（文中用该点在点云中的欧氏距离来完成）
						2、同时还为每个点的特征加上了该点最大的类别预测置信度。来组成每个点自身的新特征
						因此得到的点的特征维度是（batch , 16384, 128+Depth+CLS_SCORE）128为每个点经过PointNet++输出的特征维度
						ROI POOL融合：
						1、ROI的融合过程中还在点的特征上拼接点的坐标位置：（batch , 16384, xyz+128+Depth+CLS_SCORE）
						2、为每个点的特征融合该点自身的雷达反射率（reflection intensity）
						3、扩大每个proposal，并将扩大后才有的点也一起进行ROI POOL操作
						注：1、上述的2、3操作在OpenPCDet的实现中都没有，此处不再关注这点。
							2、对于那些内部没有点的proposal，特征置0。
						ROI POOL的采样点为512个，最终生成的特征为：(bacth * num_of_roi, num_sampled_points, xyz+Depth+CLS_SCORE+128)
						pcdet/models/roi_heads/pointrcnn_head.py
						# pooled_features为(batch, num_of_roi, num_sample_points, 3 + C)
						# pooled_empty_flag:(B, num_rois)反映哪些proposal中没有点在其中
						pooled_features, pooled_empty_flag = self.roipoint_pool3d_layer(
						    batch_points, batch_point_features, rois
						)
						池化后，proposal转换到以自身ROI中心为坐标系 
					5.2. ROI精调网络
						得到的pooled_features的特征为：
						(bacth * num_of_roi, num_sampled_points, xyz+Depth+CLS_SCORE+128)
						其中xyz，Depth，CLS_SCORE这几个都属于Local Spatial Point，128是PointNet++输出的点云语义特征。
						将Local Spatial Point特征进行Canonical Transformation后进行两层的MLP操作提升维度到128维（MLP维度变换 5->128,128->128），后再与点的语义特征进行拼接，得到的特征维度变为:
						(bacth * num_of_roi, 256, num_sampled_points, 1)。
						之后要让拼接后的特征输入的维度信息与语义特征信息一致。再使用一个MLP操作变换维度到128即可。
					5.2.2. proposal refine network
						对每个ROI中点的特征进行优化，直接使用了一个SSG的PointNet++网络。其中包含3个SA层。每层的group size为 128， 32， 1。并最终用最高维特征来优化该ROI的confidence classification和proposal location refinement。
						
				关于loss：
					OpenPCDet的损失实现已与原论文和原代码仓库不同，网络构建时候已经叙述过此问题。同时原来的实现中，PointRCNN是分阶段训练，先训练第一阶段之后在训练第二阶段网络。但是在OpenPCDet已经变成联合训练。
					1、第一阶段loss计算
					第一阶段的损失包含了两部分：
					1. 对该帧中所有的点云计算前背景分类loss
					2. 对属于前景的点云计算box的回归loss
					由于在一帧点云中属于前背景点的数量差异较大，作者在此处使用了Focal Loss：
					此处直接使用了SmoothL1损失计算前景点与GT直接的loss。对角度的编码使用了residual-cos-based的方法。所以这里的8个回归参数分别是：(x,y,z,l,w,h,cos(theta),sin(theta))
					第一阶段的损失也包含了两部分：
					1. 对ROI与GT的3D IOU大于0.6的ROI计算分类loss
					2. 对ROI与GT的3D IOU大于0.55的ROI计算回归loss
					直接使用BCE损失计算置信度分数。
					box loss：
						这里需要ROI于GT的3D IOU大于0.55的ROI计算回归loss。在OpenPCDet中，PointRCNN的第二阶段的回归loss由两部分组成；其中第一部分为前景ROI与GT的每个参数的SmoothL1 Loss，第二部分为前景ROI与GT的Corner Loss。
					1 SmoothL1 Loss
				        直接对前景roi的微调结果和GT计算Loss，这里的角度残差计算直接使用SmoothL1函数计算，原因是因为被认为属于前景的ROI其与GT的3D IOU大于0.55,所以两个box之间的角度偏差在正负45度以内。
					2 CORNER LOSS REGULARIZATION
						Corner Loss来源于F-PointNet，用于联合优化box的7个预测参数；在F-PointNet中指出，直接使用SmoothL1来回归box的参数，是直接对box的中心点，box的长宽高，box的朝向分别进行优化的。这样的优化可能会出现，box的中心点和长宽高已经可以十分准确的回归时，角度的预测却出现了偏差，导致3D IOU的降低的主要原因由角度预测错误引起。因此提出需要在(IOU metric)的度量方式下联合优化3D Box。为了解决这个问题，提出了一个正则化损失即Corner Loss
						Corner Loss是GTBox和预测Box的8个顶点的差值的和，因为一个box的顶点会被box的中心、box的长宽高、box的朝向所决定；因此 Corner Loss 可以作为这个多任务优化参数的正则项。
						

				推理：https://blog.csdn.net/qq_41366026/article/details/123275480?spm=1001.2014.3001.5502
					预测结果生成
        			看回第二阶段中roi精调的代码，在预测阶段，需要根据前面提出的roi和第二阶段的精调结果生成最终的预测结果；分别是
					batch_cls_preds (1,100,1) 每个ROI Box的置信度得分
					batch_box_preds (1,100,7) 每个ROI Box的7个参数 (x，y，z，l，w，h，theta)
					注：在推理阶段，batch size 默认为1，ROI的个数是100个。
					decode_torch完成预测结果和原ROI Box解码。 这里的对角度的解码直接将refine的预测结果与ROI的角度相加，因为他们的误差在正负45度以内。
					后处理完成了最终100个ROI的NMS操作；同时需要注意的是，每个box的最终分类结果是由第一阶段得出，第二阶段的分类结果得到的是该类别属于前景或背景的置信度得分；此处实现与FRCNN不同，需注意。
					可能由于其做了分割。类别本身就很准了。
				消融实验：
					1、如果送入refinement网络中没有将坐标系转换到CCS坐标系下，检测的结果十分糟糕，这说明了将该该proposal的点转换到CCS坐标系下可以极大程度的消除旋转和定位的的变化，降低网络学习的难度，同时也提高了refinement网络学习的效果。
					2、如果将来自第阶段的每个点经过PointNet++的特征进行移除的话，网络在中等难度的目标检测精度上下降了2.71%mAP，这说明了在第一阶段中每个点经过分割得到的特征是有效的。
					3、对于每个点加上自身相对于相机的深度信息和CLS_SCORE点的分割特征对最终的结果影响较轻微，但是需要注意的是，加上来自相机的深度信息可以补全因为转换到CCS后每个点的深度信息丢失的情况，同时CLS_SCORE也可以在点的池化时候指明哪些是前景点。
					感知点云池化
        			这里与PCDet的代码实现不同，论文中的实现是在点云池化的过程中，将池化的box范围进行了一定范围的扩展，结果也显示延长1米对最终的结果是最好的，因为每个proposal内就能拥有更多的周围环境信息，可以得到更准确的置信度估计和提高位置回归的准确度。但是如果延长的的太长，又会在ROI池化的过程中引入其它物体景点变成噪声，影响结果。
        			注：PCDet的实现中，yaml文件中在此处并没有enlarge每个proposal，同时在实现中，因为作者已经将池化操作进行了编译，所以这里暂不做详细讨论。
					在原论文中，作者对每个阶段的box回归和角度预测都是用的bin-based的方式进行的，详情可以看训练的文章内容。在PCDet的实现中，第一阶段和第二阶段box回归变成了smoothL1，角度回归变成了residual-cos-based，第二阶段的box由于是基于前面的proposal，每个proposal的3D IOU与GT Box大于0.55，它们之间的角度差仅在正负45度内，所以也直接使用了smoothL1作为损失函数；为了联合优化box和GT，还在第二阶段的box回归上加入了Corner loss进行正则化约束。
						
					

						
			

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
pvrcnn 算法原理    https://blog.csdn.net/qq_41366026/article/details/123349889?spm=1001.2014.3001.5502
	motivation：将pointbase 和voxelbase结合
		pointbase具有灵活的感受野的范围，精确的位置信息，而point-based的方法因为通过Set Abstraction操作拥有可变的感受野，使特征拥有良好的定位信息，然而基于Point-based的方法不可避免的会带来更大的计算量。
		而voxel就是高效率高效果。但是由于进行voxelize的量化操作，不可避免的导致信息丢失，使得网络的定位精度下降；
	2、PV-RCNN网络解析
		作者认为使用3D CNN backbone with anchor based的方法可以取得相比于point-based方法更高的proposal召回率，所以PV-RCNN使用了该方法作为第一阶段的提议网络；但同时因为1）特征经过了8x的下采样，这使得难以对物体进行精确定位，2）尽管可以将特征图进行上采样获得高分辨率的特征图，但是这样会使得特征图更加的稀疏，让ROI Pooling、ROI Align中的插值操作产生更多的0值，造成计算和内存的浪费。
		另一方面，set abstraction（SA）操作在PointNet的变种网络中展现了很强的领域编码能力，因此，作者提出将3D CNN和SA操作进行结合并实现第二阶段的的精确的proposal refinement。
		PV-RCNN结合了voxel-based（grid-based）的高效编码多尺度特征层来产生3D的proposal，同时又采用了point-based的灵活感受野实现精确的位置定位；将这两种方式有效的进行结合成了PV-RCNN的关键任务。
		但是如果直接在每个3D-proposal中均匀的采样几个grid-point，然后对这些grid-point进行SA（Set Abstraction）操作来进行proposal的优化的话，会造成很高的内存消耗；因为要达到不错的检测性能，voxels的数量和grid-point的数量都会相当大。
		所以，为了更好的将两种方法进行结合，作者提出了两个模块来完成这一融合。
		1、voxel-to-keypoint scene encoding
			首先，将原始的点云数据就voxelization，然后使用3D稀疏卷积来进行高效的特征提取。同时为了减少直接对庞大的voxel进行场景编码，导致内存消耗较大的问题，这里先使用FPS最远点采样方法，选取N个点后并根据这N个点来对Voxel的特征进行操作来概括整个场景信息。相当于PointNet++对每一层的voxel进行SA操作。在进行3D卷积时，会得到多层不同尺度的voxel特征层，分别在不同尺度的voxel特征层上进行基于Voxel的Grouping、SA操作，这样就可以获得不同尺度的点云信息。整个点云场景就可以被一小部分拥有多尺度信息关键点高效的编码。
			这段话的意思就是先还是走second的，然后有很多层的voxel feature，就可以做SA了。
		2、keypoint-to-grid ROI feature abstraction
        	再通过AnchorHeadSingle获得3D proposal后，为了取得准确的置信度分数预测和精确的box微调，每个proposal结合他grid-point的位置，使用ROI-grid pooling方法将多尺度的关键点特征和每一个grid point进行融合来给每个grid point丰富的感受野信息。
	
	使用3D的稀疏卷积来完成对非空voxel的特征提取，大大减少传统3D卷积的计算量。
    此处分别对voxel wise feature进行了1x, 2x, 4x, 8x的下采样操作，VoxelSetAbstraction模块会在每一层的特征图上进行VSA操作。
	
	比second多的部分：
		其中第四点（VSA模块）与第一阶段的区域提议是分开的两条计算线；先完成对voxel的Set Abstraction操作后，再在第二阶段的box refinement中融合不同的特征来更好的定位bbox。
		VoxelSetAbstraction（VSA模块，对不同voxel特征层完成SA）
			PV-RCNN在3D卷积的多层的voxel特整层上对voxel进行set abstraction操作,用一小部分关键点来编码整个场景，用于后续的proposal refinement network。
			为了让采样的一小部分关键点均匀分布在稀疏的点云中，首先对**原始的**点云数据进行了最远点采样（Furthest Point Sampling），其中KITTI数据集采样2048个关键点，Waymo数据集采样4096个关键点。
			采样得到：得到选取的关键点 shape : (batch*2048， 4)  , 4-->batch_idx, x, y, z
		 3D CNN VSA
			PV-RCNN中使用了PointNet++中提出的SA操作来对不同尺度上的voxel特征进行聚合。
			在VoxelBackBone8x中，分别得到了1x，2x， 4x， 8x的voxel-wise feature volumes，VSA操作会分别在这些尺度的voxel-wise feature volumes上进行，得到4个尺度的voxel编码特征。
			对于上一步得到的每一个关键点，首先确定他在第K层上，半径为R_k邻域内的非空voxel，将这些非空voxel组成voxel-wsie的特征向量集合；
			然后将不同尺度上相同关键点获取的voxel-wsie的特征向量拼接在一起，并使用一个简单的PointNet网络来融合该关键点不同尺度的特征，
			具体有：
				经过3D CNN的第K层的voxel特征的集合
				voxel在第K层中的3D坐标，Nk是第K层中非空的voxel
				每个voxel 特征在对应半径内关键点的相对位置信息
			对每一个关键点，首先确定他在第K层上，半径为R_k邻域内的非空voxel，将这些非空voxel组成voxel-wsie的特征向量集合；然后将不同尺度上相同关键点获取的voxel-wsie的特征向量拼接在一起，并使用一个简单的PointNet网络来融合该关键点不同尺度的特征，
			注意是不同尺度上的同一个关键点获取的voxel wise特征向量。
			K层中，固定半径内voxel特征集合中的随机采样操作。
			在实现中，每个集合中最大采样16或32个voxel-wise feature，节省计算资源
			同时，每一层的R_k设置如下（单位:米），用于聚合不同的感受野信息：

			1x : [0.4, 0.8] ，采样数[16, 16]，MLP维度[[16, 16], [16, 16]]

			2x : [0.8, 1.2]，采样数[16, 32]，MLP维度[[32, 32], [32, 32]]

			3x : [1.2, 2.4]，采样数[16, 32]，MLP维度[[64, 64]], [64, 64]]

			4x : [2.4, 4.8]，采样数[16, 32]，MLP维度[[64, 64], [64, 64]]
			最终学习到的特征结合了基于3DCNN学习到的特征和基于PointNet从voxel-wise SA中学习到的特征。
		 Extended VSA：
			在对每层3D卷积的输出进行VSA操作后，为了能够是学习到的特征更加丰富，作者扩展了VSA模块；
			在原来VSA模块的特征上加上了来自原点的SA特征和来自堆叠后BEV视角的双线性插值特征，
			原始点的sa特征比较好理解，但BEV的双线性插值就很难理解了。
			加入Extended VSA的好处：
				1、来自原点的SA操作可以弥补因为voxelization导致的量化损失
				2、来自BEV视角的插值SA操作拥有更大的Z轴（高度）感受野
			[
			(2048 * batch, 256) BEV视角下点特征数据
			(2048 * batch, 32)  原始点云下特征数据
			(2048 * batch, 32)  x_conv1 第一次稀疏卷积后特征数据
			(2048 * batch, 64)  x_conv2 第二次稀疏卷积后特征数据
			(2048 * batch, 128) x_conv3 第三次稀疏卷积后特征数据
			(2048 * batch, 128) x_conv4 第四次稀疏卷积后特征数据
			]
			
			代码都在：pcdet/models/backbones_3d/pfe/voxel_set_abstraction.py    BEV视角插值也在
				
		PointHeadSimple Predicted Keypoint Weighting                                     pcdet/models/dense_heads/point_head_simple.py
			在将不同尺度的场景都编码到N个关键点后，将会在后面的精调阶段使用到这些关键点的特征，
			但是这些被最远点采样（FPS）算法选取出来的关键点是均匀的分布在点云中的，
			这就意味着有一部分的关键点并没有落在GT_box内，他们就代表了背景；
			作者在这里认为，属于前景的关键定应该主导box的精调，
			所以作者在这里加入了PKW模块用于预测该关键点属于前景点还是背景点。
			PKW模块用于调整前背景点的权重方式来实现，
			其中对于前背景点的分割GT值，由于在自动驾驶场景的数据集中所有的3D物体都是独立的，不会像图片中物体overlap的情况，
			可以直接判断一个点是否在3Dbox内即可得到前背景的类别，权重调整公式如下：
			是一个三层的多层感知机，最终接上一个sigmoid函数来判断该点的属于前景的置信度。
			由于3D场景中前背景点的数量过于不均衡，PKW模块使用Focal Loss进行训练，Focal Loss的alpha，gamma参数设置与RetinaNet一直，alpha为0.25，gamma为2
			注：对于点前背景分割，PV-RCNN与PointRCNN中设置一致，对每个GTBox扩大0.2m，判断是否有关键点落在GTBox边沿，并将这个处于边沿的GTBox点不进行loss计算。
		
		关键点的target assignment：pcdet/models/dense_heads/point_head_template.py
			
		PVRCNNHead（二阶proposal精调）：
			在第VSA模块中，已经一帧点云场景编码到一小部分拥有多尺度语义信息的关键点特征中，
			同时，也由BEV视角下生成了很多3D的proposal（ROI）；
			在第二阶段的refinement过程中，需要将来自ROI的特征融合关键点的特征，提升最终box预测的准确度和泛化性。
			作者在这里提出了基于SA操作的keypoint-to-grid ROI feature abstraction，用于多尺度的ROI特征的编码。
			对于一个3D ROI，PV-RCNN中提出了ROI-grid pooling操作，
			在每个3D proposal中，均匀的采样6*6*6个grid point点     这个6*6*6的点是直接采样得到的，可能还不是像最远点采样直接采样的就是原始的点，6*6*6可能原始的点还不存在
			并采用和PointNet++中和SA一样的操作，并在这些grid point点的多个尺度半径内分别聚合来自VSA中2048或4096个关键点特征的信息。     又做了SA？  聚合2048啥意思？
			最后将不同半径大小的grid point特征拼接在一起来获取更加丰富的多尺度语义信息
			操作取每个keypoint feature set中特征最大的特征作为grid point特征。
			在获得所有的grid point特征后，使用一个两层的MLP网络将特征转换到最终的256维度，用以代表该proposal。
			ROI grid Pooling操作相比于Point RCNN、Part A2、STD模型的ROI池化操作，拥有更好的感受野信息，因为在ROI grid Pooling的时候，gridpoint包含的关键点特征可能超出了proposal本身，获取到了3D ROI边缘的特征信息；而之前模型的ROI Pooling操作仅仅池化proposal内部的点的特征（Point RCNN），或者池化很多无意义零点的特征（Part A2、STD）来作为ROI feature。
			最终的refinement网络由两个分支构成，一个分支用于confidence预测，另外一个分支用于回归残差预测，
			同时confidence的预测变成了quality-aware 3D Intersection-over-Union (IoU)，公式如下：
			并使用cross-entropy损失函数来优化：
			
		proposal生成：
			根据前面一阶段得到的anchor信息，生成二阶段refinement需要的proposal；其中在训练阶段需要生成512个proposal，推理阶段生成100个proposal。得到的结果如下：
			rois                （batch size， 512， 7）  7-->（x, y, z, l, w, h, θ）
			roi_scores     （batch size， 512）        每个proposal的类别预测置信度分数
			roi_labels      （batch size， 512）        每个proposal的预测得到的类别
		proposal的target assignment：
			在选取出proposal后，需要对选取proposal对应的GT，用于计算loss。
			注意：在训练阶段的proposal refinement中，只需要从512个proposal选取出128个样本进行refinement，同时正负样本比例为1：1（正样本不够则用负样本填充）；如果一个proposal与GT的3D IOU大于0.55，则认为这个proposal为正样本，需要进行box refinement回归，否则，认为他是负样本。
			和prcnn的类似
			实现流程：
				1、使用self.get_max_iou_with_same_class()函数计算512个proposal与对应类别GT的3D IOU大小。
				2、self.subsample_rois（）函数完成128个proposal采样和proposal的正负样本分配。
				3、类ProposalTargetLayer  forward函数选取出来需要计算回归loss 的roi mask，在PV-RCNN中使用了带感知的IOU当成置信度预测分数，并使用cross-entropy损失来进行优化。
				4、并将选取出来的roi的GTBox转换到和PointRCNN一样的CCS坐标系下。
		proposal的grid point融合关键点特征：
			就是之前说的什么SA之后在和之前的特征进行聚合了？
		
		LOSS：
			这篇文章将根据前面生成的target assignment结果进行loss计算和网络的推理以及PV-RCNN中的消融实验。
			PV-RCNN是端到端训练的，一共需要计算三部分的损失：
				1、anchor头的分类和回归损失 
				2、Predicted Keypoint Weighting （PKW）分割损失
				3、Refinement网络损失
				总体的训练损失为三者相加，且三者权重相等。				
			注：Loss计算的内容均在之前的博客中详细介绍过，这里再简单叙述一下；如果了解SECOND和PointRCNN的LOSS计算，不需要再看Loss计算部分。
			1、anchor头的分类和回归损失
				其中Lcls为每个acnhor的分类损失，采用focal loss进行计算（alpha和gamma与Retina一样，分别为0.25, 2），并采用SmoothL1函数优化anhcor box residual regression。
				而且有每个框的方向的分类
			2、Predicted Keypoint Weighting （PKW）分割损失
				由于在一帧点云的关键点中属于前背景点的数量差异较大，作者在此处使用了Focal Loss
				注：在计算前背景点的分类loss时，对每个GT enlarge 0.2米后才包括的点，类别置为-1,不计算这些点的分类loss，来提高网络的泛化性，网络构建已经有提到过。
			3、Refinement网络损失
				为预测的box残差
				quality-aware confidence prediction
					在PV-RCNN中作者将二阶段的置信度预测改成了quality-aware confidence预测的形式，计算公式如下：
					 yk = min (1, max (0, 2IoUk − 0.5))
					在target assignment的时候，已经完成了该计算。
				box refinement loss
					在 box refinement loss的计算过程中，这里需要ROI于GT的3D IOU大于0.55的ROI计算回归loss。在OpenPCDet中，PVRCNN的第二阶段的回归loss由两部分组成；其中第一部分为前景ROI与GT的每个参数的SmoothL1 Loss，第二部分为前景ROI与GT的Corner Loss。
					SmoothL1 Loss
						直接对前景roi的微调结果和GT计算Loss，这里的角度残差计算直接使用SmoothL1函数计算，原因是因为被认为属于前景的ROI其与GT的3D IOU大于0.55,所以两个box之间的角度偏差在正负45度以内。
					CORNER LOSS REGULARIZATION
						见prcnn的loss。
						
			推理：
				看回PV-RCNN head第二阶段中roi精调的代码，在预测阶段，需要根据前面提出的roi和第二阶段的精调结果生成最终的预测结果；分别是：
				batch_cls_preds (100,1) 每个ROI Box的置信度得分
				batch_box_preds (100,7) 每个ROI Box的7个参数 (x，y，z，l，w，h，theta)
				注：在推理阶段，第一阶段的anchor得到的ROI的个数是100个且NMS阈值是0.7。
				生成最终预测box的函数为generate_predicted_boxes
				注：这里没有方向分类，因为角度已经在正负45度以内的，原因在训练的target assignment中已经说过。
			后处理：
				后处理完成了对最终100个ROI预测结果的NMS操作；同时需要注意的是，每个box的最终分类结果是由第一阶段得出，第二阶段的分类结果得到的是IOU预测；
				此处实现与FRCNN不同（Frcnn第一阶段完成前背景选取，第二阶段进行分类），需注意。
				后处理中的NMS阈值是0.01，不考虑任何overlap在3D世界中。
			消融实验：
				RPN Baseline是直接SECOND的网络形式，
        		Pool from Encoder是前面提到过的，直接将多个尺度的特征聚合在一个ROI grid（个人认为，此处ROI应该是指BEV操作前的voxel特征层）中进行调优的结果以KITTI为例，在经过4x下采样的一般场景中，还会存在18K个voxel，如果每个voxel中采用3x3x3的grid point进行聚合的话，那么需要计算2700*18000的成对距离和特征聚合（该方式会占用很多的计算资源和内存）
				PV-RCNN则是本文采用的方式，只将场景编码成一小部分关键点特征，并采用Keypoint-to-grid RoI Feature Abstraction for Proposal Refifinement将的方式来对proposal进行精调
				主要是因为在proposal中融入关键点编码的可以扩大感受野并且使用带有监督的关键点分割可以学习到更好的关键点特征。使用一小部分关键点来作为中间的特征表达相比于Pool from Encoder的方式可以有效的减小资源消耗。
				VSA中使用来自不同层的特征的影响				
					从图中可以在处在VSA模块中融入不同尺度的特征对最终结果的影响如何，这里主要说一点，如果只采用来自原始点云中的特征的话，效果会降低很多，说明来自不同3D卷积层的语义信息对于box的定位是有帮助的。		
				PKW模块主要用于调整属于前背景关键点特征的权重，可以看1和4行结果，如果没有使用PKW模块对关键点属于前背景权重的权重进行调整，网络的定位精度下降了。说明在proposal中前景点的多尺度特征需要给与更多的关注。
				这里的2、3行主要比对了使用ROI-aware Pooling（来自史帅自己的Part A2 Net）和使用本篇文章提出的ROI-grid Pooling对结果的影响，验证了ROI-grid Pooling相比于ROI-aware Pooling拥有更加丰富的感受野，原因在实现第一章就已经说明了，主要还是基于ROI-grid Pooling因为有SA操作的原因，对proposal外部边沿的点都有聚合的作用。

		        3、4行的比对试验表明了基于quality-aware confifidence prediction strategy对最终结果的影响，大家自行看一下。
						
											


		
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
			


voxelrcnn 算法原理            https://blog.csdn.net/qq_41366026/article/details/123520480
	point的就是位置很准，但是本文认为认为精确的原始点云信息对于高性能的3D点云检测器不是必不可少的，同时提出使用粗粒度的voxel（coarse voxel granularity）信息同样可以得到不错的检测精度。
	Voxel-RCNN的6个模块
		1、MeanVFE
		2、VoxelBackBone8x
		3、HeightCompression
		4、BaseBEVBackbone
		5、AnchorHeadSingle
		6、VoxelRCNNHead
		注1：其中黑色部分均与SECOND中相同。
		注2：第6点内容为Voxel-RCNN的二阶段调优实现，代码解析会直接从这里开始，前面的内容均与SECOND一致。
	2、Voxel-RCNN设计思考：
        Voxel-RCNN的设计初衷，是为了能够达到和point-based一样的精度同时要和Voxel-based的方法一样的快。因此，作者还是采用了基于Voxel-based的方法，并试图对他进行改造提升他的精度。所以，文中对比了PV-RCNN、SECOND两个经典的网络来进行分析。
		注：在PV-RCNN中，也是通过SECOND的网络结构作为第一阶段的特征提取，但是同时也采取了关键点编码场景特征的方式来聚合不同尺度下3D卷积层特征；并在精调阶段，将对应的关键点特征融入proposal中，提高了网络的性能。
		对比上图，可以发现，如果直接对SECOND加入二阶网络的box refinement操作，精度的提升也是十分有限， 
		说明BEV特征的表达能力受限，主要原因是现存的voxel-based的方法检测精度受限的主要原因来源于直接将经过3D的卷积的voxel feature直接在Z轴堆叠之后之后在BEV视角上进行检测，
		没有在中间的过程中恢复特征原有的3D结构信息。
		就是HeightCompression压缩了位置结构信息。
		秉持这个观点，在二阶网络的精调过程中融合来自3D卷积层中的3D结构信息。
		文章中对这部分的实现主要是使用voxel ROI pooling来提取proposal中相邻的voxel特征，
		并设计了一个local feature aggregation模块来进一步提升计算速度，因为作者比对了PV-RCNN各个模块的耗时，结果如下
		可以看到最不同尺度的voxel进行关键点编码是十分耗时的。
		总的来说总结如下：
		1、3D的结构信息对于3D检测器十分重要，单纯使用来自BEV的特征表达对于精确的3D BBOX定位是有限的。
		2、如果直接使用point-voxel的方式来生成编码特征是十分耗时的，影响了检测器的效率。
		3、Voxel ROI pooling
			3.1 Voxel Volumes as Points
				为了直接从经过3D卷积后的3D特征层上聚合空间信息，直接将每层3D特征层认为是一个的非空的Voxel的集合，和所有对应非空voxel的特征向量
				同时每个voxel的3D中心坐标根据3D卷积的indice、voxel size（KITTI：[0.05,0.05,0.1],WAYMO:[0.1,0.1,0.2]）、选取的点云的范围计算得来。
			3.2 Voxel query        
				得到非空voxel的坐标和特征后，使用Voxel query来寻找周围特征信息。
				原来的Ball Query操作是在一个没有组织的数据上进行的，然而voxel是特征则是规则有序的；所以作者在这里使用了曼哈顿距离来计算一个Query Point相邻的voxel有哪些。
				（为什么用曼哈顿距离）
				每个voxel query最多采样K个voxels，K为16。Voxel Query的时间复杂度是O（K）（K为邻域voxel的数量），而Ball Query的时间复杂度是O（N） （N为非空voxel的数量）
			3.3 Voxel ROI Pooling
				每一个ROI内部均匀划分成6x6x6个个小块，每个小块的中心可以看做成一个grid point点，然后根据grid point点在哪个voxel，并以这个voxel为query point执行voxel query操作，
				得到的每一个grid point点集合的特征，然后使用一个简单的PointNet模块来聚合领域voxel的 特征，
				注：如果已经了解PV-RCNN Keypoint-to-grid RoI Feature Abstraction操作的的话，那么此处的操作就是将grid point从编码的关键点中融合信息变成了从每层3D卷积的非空voxel中直接获取一个范围内的点进行group操作。只不过原来的group操作重BallQuery换成了此处Voxel Query（从原来的基于半径的group变成了基于曼哈顿距离的group操作）
			3.4 加速实现（Accelerated Local Aggregation）
				原来的Pointnet中，将点的特征和点的坐标拼接在一起再念FC操作；这里的加速实现则是将点的坐标和点的特征分别进行FC操作，并得到两组对应的特征后，将特征对应相加，然后再取沿着通道维度取Max得到这个grid point点的聚合特征。
				
		后面都是和pvrcnn一致的，直接看消融实验了：
			D.H. 代表二阶检测头，用于boxrefinement
			V.Q. 代表voxel querry操作
			A.P. 代表加速实现的PointNet操作
			1、方法(a)是对BEV特征进行检测的单阶段基线。它以40.8FPS的速度运行，但AP不令人满意，这表明仅凭BEV表示不足以精确检测3D空间中的对象。 
			2、方法（b）在a的基础上扩展了一个检测头用于box refinement，提升了检测精度，这也证明了来自三维体素的空间的上下文信息（3D Voxels context）可以为精确的目标检测提供了足够的信息，但是使用的是ball query来提取中间卷积层的特征。
			3、方法（c）在b的基础上使用了voxel query，这替换了方法b中的ball query操作，加速了检测速度
			4、方法（d）使用了加速实现的PointNet模块，进一步将FPS从17.4提升到21.4。
			5、方法（e）就是本文提出的Voxel RCNN，结果如上图所示。
				
				
				

SST 算法原理



bevformer 算法原理



rangedet 算法原理



pvrcnn++ 算法原理
	PV-RCNN++在PV-RCNN的基础上进行了主要的两点改进：
	1、将原来的FPS(最远点采样)换成了sectorized proposal-centric keypoint sampling strategy（分区域的提议中心关键点采样 ？），
	使得有限的关键点可以更加的聚集在proposal区域范围内，来更多的编码有效前景点特征用于后面的proposal refinement。
	同时sectorized farthest point sampling在不同sectors（分块区域 ）的关键点采样是平行进行的，这样不仅保证了分块区域中采样的关键点在该分块点集中的均匀分布，还相比于vailla FPS（普通的最远点采样）算法减少了两倍的复杂度。
	同时作者再次强调，局部位置中点与点之间的相对位置信息对于描述局部的空间几何信息是十分有效的。
	PV-RCNN++中的8个模块（其中两个改进点都集中在了VoxelSetAbstraction和二阶预测头中）
		1、MeanVFE        （voxel feature encoding）
		2、VoxelBackBone8x    （3D backbone）        
		3、HeightCompression （Z轴方向堆叠）
		4、VoxelSetAbstraction （VSA模块）
		5、BaseBEVBackbone   （2D backbone for RPN）
		6、AnchorHeadSingle        （一阶预测头）
		7、PointHeadSimple Predicted Keypoint Weighting  （PKW模块）
		8、PVRCNNHead        （二阶预测头） （grid voxel中特征聚合也采用vector pool的方式）
	3、sectorized proposal-centric keypoint sampling strategy
		在PV-RCNN中关键点采样是非常重要的，使用聚合的关键点特征补齐了点的体素表达，提升了最终proposal refinement的效果。但是在之前PV-RCNN中的使用的关键点采样算法是FPS（Farthest Point Sampling），该算法有两个主要的缺陷
		1：该算法的时间复杂度是O（n^2），这会严重拖慢网络的训练和推理的效率，尤其是在大场景的点云检测中。
		2：该算法直接在大范围的点云中进行关键点采样，事实上只会有很小一部分采样得到的关键点属于前景点，大部分关键点属于背景点；然而在proposal refinement阶段中，背景点对优化是无意义的，因为在进行6*6*6 的ROI-grid Pooling的时候只会采用每个grid point周围的关键点进行融合。
		所以作者为了解决这个问题，在PV-RCNN++中提出了更为有效的关键点采样策略-->Sectorized Proposal-Centric (SPC) Keypoint Sampling。
		2（Proposal-Centric）：既然大部分背景点都是没有用的，那么不妨就直接在第一阶段提出的proposal中附近进行采样，
		注1：原始点云为P，每个3Dproposal的中心点和尺度大小分别为C和D；其中dxj , dyj , dzj为proposal的长宽高；r^(s)为proposal中心点向外扩张的最小半径（最小的原因是要先取（dxj , dyj , dzj）一半的最大值），实现中该超参数被设置为1.6米。P'为经过选取后保留下来的在proposal附近的点。
		 1（Sectorized）：同时为了解决FPS时间复杂度为O（n^2），致使对大场景采样慢的问题；最直接的想法就是将原始的点云分成多个子集，并在每个子集上分别进行采样，因此作者直接基于每帧点云的中心点，采用公式
			将点云分成多个子集。 其中在每个子集点云中继续采用FPS算法进行关键点采样，因为FPS算法可以保证采样的关键点均匀的分布在原始点云场景中；同时由于得到的每个子集点云都是独立的，所以每个子集点云都可以并行运行在gpu上，进一步加速了点云的关键点采样过程。
		每个扇区执行最远点采样
	4、VectorPool aggregation
		PV-RCNN中作何就提出了从局部聚合有用的特征来提升refinement的效果，所以在PV-RCNN中作者就采用了SA（Set  Abstraction）操作来分别在每个关键点特征集合的特征和ROI grid pooling中进行使用。但是SA操作在大型的点云数据中消耗的资源和也是庞大的，这样使得网络难以在端侧运行，所以作者提出了Local Vector Representaion for Structure-Preserved Local Feature Learning。这行专业术简单点说就是带有空间结构信息的vector特征。
		
			
	



fcos3d 算法原理



SSN/RSN



3DSSD



smoke



lidarRCNN



BEVFormer



SASSD



CIASSD



partA2



Bevfusion（阿里）



BEVfusion（MIT HAN）



CIA-SSD



SA-SSD



IA-SSD



PillarNet



pointnet && pointnet++
	有个问题，pointnet全部都是MLP？
	pointnet++里面就是SA模块，FPS+Grouping+pointnet
	FPS完了后，对每个中心点找它的邻域点,形成子点集。注意子点集之间可以存在overlap,并且每个子点集的点数目不一定相等。有两种方案，一个是ball query 一个是K近邻
	分割部分就是得知道怎么插值的，找到在原始的点云的位置，然后找到最近的几个点，加权求和几个特征，权重和距离成反相关，插值。那么附近的几个点的特征从哪里来，特征图？原始点云空间肯定没有
	距离的倒数。  反正就是上采样。
	不同的感受野？：	
		对当前层的每个中心点,取不同radius的query ball,可以得到多个不同大小的同心球,
		也就是得到了多个相同中心但规模不同的局部邻域,分别对这些局部邻域表征,并将所有表征拼接。



AI视觉网奇整理的3D检测：
	https://blog.csdn.net/jacke121/article/details/125072778


简单总结下关于yaw的问题：
	一开始在voxelnet里面，是直接回归的yaw角，但是实际就会发现当前6个值回归的都很好的时候，yaw角相反，造成loss很大，优化效果不佳，那么想到了用sin函数进行约束
	这样相反的yaw就得到了sin（-yaw）= -sin（yaw），自然损失函数就大了，那么当 [公式] 的时候，该损失趋向于0，这样更利于模型训练。
	而且你看sin（0）和sin（180）也是一样的
	但是这样的话，实际上在sin（150）或者sin（30）的时候，yaw角的损失实际是一样的，我们现在给yaw角分两个bin表示前后
	就是预测是正还是反；这样子如果预测的结果是正那么yaw就是yaw本身， 如果预测的结果是反，即dir分类为1，那么就yaw+180，就完成了预测。
	但是实际上在分配target的yaw的cls时候，由于大部分的车都存在于和自身的车90°，0°这样的值，
	使得其正好在分类的边界，这样就会导致这个值不稳定，所以我们提前对GT的yaw角减去45°（舍弃了原本45°的车的分类，即正在转弯的？）
	这样分配出来的正样本标签就可以分出前后了。所以在预测的时候，由于你的真值是根据减去45度预测的cls，
	那么你实际在做pred_yaw + cls * 180这个操作的时候需要把pred_yaw-45
	因为pred_yaw就类比GT，是减去了45再去加上分类才是真的带了方向分类的yaw，
	而pred_yaw本身确实回归按照GT原本的回归的，所以减去45之后加上cls，还得再加上45恢复到原本的值，也就是之前减去45完全为了cls

	在kitti的标注里面，yaw角是障碍物车前进方向和自车相机x轴的夹角，实际在训练的时候肯定需要把这个角度转换为：
	障碍物车前进的方向和激光雷达的x轴的夹角
	openpcdet 是把角度限制在了0-2pi，原始的角度在-pi，pi；
	比如x朝前，你加上180°，反向了，但是和你迎面而来以及和你同方向的可能就是不好去做label assign；

    大部分的目标都在，0度和180度，270度和90度，
    应该是0和180度的很容易分错掉，减去了45度，0 和180就很好分辨了
    但是如果你的bin过多，在边界的车就越不容易分出来，来回跳，洒老师讲的，而且我之前也遇到过的，却是这样







'''
