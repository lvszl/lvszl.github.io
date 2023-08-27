---
tag: AI
category: 系统论文
---

又要读论文啦！！

![image-20230826202227357](https://raw.githubusercontent.com/lvszl/figure/master/image-20230826202227357.png)

**目的**：联合利用多个GPU来训练图神经网络。

**方式**：用一种特殊的数据布局来利用GPU之间的`NVLink`连接， 并且GPU（显存）中存放了图形拓扑和popular node features。

**措施**：

- 引入集体采样源语：collective sampling primitive
- 设计了基于消费者生产者的管道：允许不同小批量任务并行。

##  先抨击了已有的系统：`Quiver`和`DGL-UVA`

系统`Quiver`与`DGL-UVA` ：将图形拓扑存放在内存中，而节点特征存放在GPU的显存中，每个GPU独立进行图采样，并使用UVA（统一虚拟寻址）通过PCle（一种高速串行计算机扩展总线标准，支持显卡）访问图拓扑，但有两个问题：

1. 高沟通成本：

![image-20230826230847222](https://raw.githubusercontent.com/lvszl/figure/master/image-20230826230847222.png)

![image-20230826231006622](https://raw.githubusercontent.com/lvszl/figure/master/image-20230826231006622.png)

图1和表1说明：

- UVA等系统读入了太多无用的数据量
- 采用的PCIe接口远远慢于NVLink，但UVA与Quiver对NVLink的支持并不好，UVA采样是在PCIe上的

2. GPU利用率低

![image-20230826232715665](https://raw.githubusercontent.com/lvszl/figure/master/image-20230826232715665.png)

图为“更改物理线程时候Quiver的图形采样和功能加载的执行速度”，且这个GPU最多开5120个线程。

图中可以发现，均存在某个点，在那个点之后，就算再开线程，但速度就不变了基本。因此无法充分利用GPU。



## 提出自己的模型

DSP：

- 将图结构划分为不同的图块，这些图块都是联通（well-connected）的，并把每个图块放在一个GPU中，这样图形采样就能通过NVLink，而不是用PCIe了，而且由于不用PCIe了，所以读取的数据量就变少了。

  >  为什么呢？因为看前面，前两个系统把图结构存放在内存中，而NVLink是适用于GPU之间通信的，因此采样需要通过PCIe，但如果把图信息放到GPU中，那采样就是读取GPU，就可以通过NVLink了。

  每个GPU的剩余显存，则用来存放不同的节电功能，之后，所有的GPU形成一个通过NVLink的大型聚合缓存，来减少节点通过PCIe对CPU内存的访问。

- 定义了采样器，装载器和训练器：

  - 采样器：就是采样——集体采样源语：CSP
  - 装载器：加载图样本的节点特征
  - 训练器：用来训练

- CSP（集体采样源语：将每个图节点上的采样任务推送到其驻留的GPU上，而不是拉出其邻接列表（图1）大幅度减少通信数据量：因为通常只对节点邻居节点进行采样。

- 利用生产者消费者管道，来重叠"使用GPU"的任务



## 基于采样的GNN训练

基于文中的假设，太长了不写了，直接复制，狠狠地复制！

**不基于采样的GNN训练**

![](https://raw.githubusercontent.com/lvszl/figure/master/20230828004830.png)

然后我们知道GNN的聚合公式：

![](https://raw.githubusercontent.com/lvszl/figure/master/20230828005601.png)

不知道？不知道~~滚~~回去看文章中的解释。

有个关键点就是：

用第一层训练节点$v$ 的时候，要聚合$v$的邻居节点的信息，同时其邻居节点也聚合了他们自己邻居节点的信息，因此当在第二层训练节点$v$时，表面上节点$v$是再次聚合了它邻居节点的信息，实则同时聚合了邻居节点以及邻居节点的邻居节点的信息。故，GNN有k层，训练一个节点$v$就需要聚合其k-邻居节点的信息，并且貌似重复聚合了很多次。所以在稠密图中复杂度贼高。

**基于采样的GNN训练**

