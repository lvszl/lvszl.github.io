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

- 将图结构划分为不同的图块，这些图块都是联通（well-connected）的，并把每个图块放在一个GPU中，这样图形采样就能通过NVLink，而不是用PCIe了。

  >  为什么呢？因为看前面，前两个系统把图结构存放在内存中，而NVLink是适用于GPU之间通信的，因此采样需要通过PCIe，但如果把图信息放到GPU中，那采样就是读取GPU，就可以通过NVLink了。

- 

