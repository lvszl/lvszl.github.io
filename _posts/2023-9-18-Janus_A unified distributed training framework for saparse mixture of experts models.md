---
tag: AI system
category: 系统论文
---

![](https://raw.githubusercontent.com/lvszl/figure/master/20230918152630.png)

# abstruct

**all-to-all communication:** (expert-centric) 让专家位于原地，数据在专家之间进行交换。

作者提出了一种”data-centric“的范式：让数据位于原地，在GPU之间移动专家。（因为专家的规模小于数据）。——Janus

**主要适用于**

the size of expert is small while the amout of input data is large(e.g., large batch size)

**特点**

1. 支持“fine-grained”（细粒度的）异步通信。——使得计算和通信重叠

同时，实现了分层通信，通过在**同一台机器上共享获取的专家**来进一步减少交叉节点流量。

2. 其次，在调度“抓取专家”请求时，Janus实现了拓扑感知的优先级策略，以有效地利用节点内和节点间的链路
3. 最后，Janus允许**预取**专家，这允许下游计算在前一步完成后立即开始。

# introduction

------

**MoE示意图**

![](https://raw.githubusercontent.com/lvszl/figure/master/20230919111212.png)

**expert parallelism示意图** 

是为了解决MoE模型超过了单个GPU的内存限制而提出的。示例如下：

![](https://raw.githubusercontent.com/lvszl/figure/master/20230919112413.png)

Expert被放在了不同的GPU（worker）上，因此，在模型计算过程中，必须在专家层前后添加all -to- all通信，以便在各gpu之间交换中间结果。

> 在NLP中，中间数据也叫作 **token**

------

## data-centric的优点：

1. 实现expert之间的异步通信，而不像expert-centric一样，必须等每轮迭代中GPU全部完成后，再进行通信（同步）。同时，同一个机器上的expert可以被该机器上不同的GPU重用。
2. 由于**专家权重（expert weights)**在单次迭代中是确定的。因此预取专家是可能的，并且它允许我们进一步改善计算和通信之间的重叠

> "expert weights" 指的是模型中的参数，具体来说是用来调整模型中不同专家（或子模型）的重要性或贡献程度的权重参数。
>
> 这里的权重不是用来决定数据分配给哪个专家的权重，而是用来决定每个专家模型对最终预测结果的贡献有多大的权重
>
> 因此，其在一次迭代过程中是确定的。

## Janus提升模型性能的三个视角

1. Janus将获取每个专家的请求作为单独的任务。
