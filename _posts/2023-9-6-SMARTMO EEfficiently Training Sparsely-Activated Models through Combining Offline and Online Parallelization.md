---
tag: AI
category: 系统论文
---

# 2 Background and Motivation

MoE：Misture-of-Experts

<img src="https://raw.githubusercontent.com/lvszl/figure/master/20230907153413.png"/>

FFN为MoE模型中的专家，多个FFN和一个Gating组成了基本的MoE模型。

## 2.2 混合和自动并行化

训练密集型深度学习网络的常见的三种并行方式：

### Data Parallelism（DP）

每个worker均存储一个完整的参数副本，分配给每个worker的训练样本都不同，并且前向传播和反向传播是在每个worker上独立完成的, then gradients on different workers will be aggregated before being used in the optimization of the model.

disadvantages: 由于参数需要在每个workder中同步和复制，因此会造成内存浪费和communication overhead.

### Pipeline Model Parallelism(PP)

参见之前的DSP的文章，就用的PP这个思路：将训练过程分为多个阶段：Sampling，loading and training。

然后每个worker负责一个，会导致communication过程的时间花销变大。

### Tensor Model Parallelism

张量模型并行（Tensor Model Parallelism，TP）是一种人工智能领域的技术。模型的单个运算符被分割成多个工作节点（workers）。每个工作节点存储运算符参数的一部分，并执行其中一部分的计算，例如矩阵的一个瓦片（tile）。不同运算符的TP需要由专家专门设计，分割方法对分布式训练性能至关重要。Megatron [35] 提供了在Transformer模型上使用TP的最佳实践。其他研究[36, 38]探讨了TP的统一表示和最高效分割的自动生成。



最后，总结了任何自动并行化训练系统的三个关键挑战：

1. Space of Hybrid Parallelism（混合并行空间）

不同策略的混合使用可能会带来适应的问题

2. Performance Modeling 性能建模

性能建模有助于有效探索巨大的混合并行空间。

3. Searching Algorithm 搜索算法

因为搜索空间巨大，因此要找一个好的搜索算法。



## 2.3 Challenges of Automatic Parallelization for MoE Models

用下图解释自动并行MoE模型的挑战性：

<img src="https://raw.githubusercontent.com/lvszl/figure/master/20230909164732.png"/>

1. 更大的混合并行空间。

因为模型要额外考虑不同expert之间的组合差异，如E0与E1组合还是E0与E3组合。（一个MoE layer 有四个expert）

2. 工作负载感知性能建模

传统的性能建模方式仅使用模型结构和硬件信息来估计性能，缺乏对工作负载的考虑。因为在上图中，上下两个工作负载使得同一个方案出现了效率上的差异。

3. 自适应动态并行化

由于MoE训练过程会出现不同工作负载，因此我们需要自适应自动并行化，该方法采用运行时执行方案搜索和切换来保持训练过程高效率。训练系统可以在每次迭代时更新执行计划，以实现最终的高性能。

# 3 Overview——SmartMoE



