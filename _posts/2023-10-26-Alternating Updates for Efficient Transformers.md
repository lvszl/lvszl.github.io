---
tag: AI
---



![](https://raw.githubusercontent.com/lvszl/figure/master/20231026132158.png)

交替更新，来在不增加计算负担的情况下增加模型容量。

AltUp通过在每一层扩展表示的子块上工作，并使用预测和纠正机制来更新未激活的块来实现这一点。

具体实现：

# 3 Alternating Updates:

## 3.1背景

有L层的Transformer

对于一个长度为$N$的输入序列，初始token为$x_1\in R^d$ ， 是embedding得到的。

之后每通过一层Transformer，就能计算出一个：

$x_{i+1} = L_i(x_i)$ 。

![](https://raw.githubusercontent.com/lvszl/figure/master/20231031215403.png)

输入是d维的embedding，然后进入Transformer后，变为了Kd维的representation。

算法具体流程如下：（如何训练出这个预测模型）

加入Altup层的输入是d维的连续子块的串联：$x_{old}=concat(x_{old}^1,x_{old}^2,\dots,x_{old}^K)\in R^{dK}$ 

然后AltUp先 生成每个子块$i$的预测结果：$\hat x_i$ （利用等式1）

$\hat x_i=\sum_{j=1}^Kp_{i,j}x^j_{old}$ ，$p_{i,j}$是用来学习的参数。

 然后，选择一个子块，然后在这个子块上用原始的d维的Transformer进行计算：

![](https://raw.githubusercontent.com/lvszl/figure/master/20231031220700.png)

最后，该计算结果来对预测的子块进行校正，然后生成每个子块的更新表示。

![](https://raw.githubusercontent.com/lvszl/figure/master/20231031220826.png)



思考：

1.预测模型需要由数据集预训练出来，然后才能针对IID的数据进行预测。相当于训练出来了一个大模型。（Transformer是预先训练好的）



这篇文章中与MoE的结合：

![](https://raw.githubusercontent.com/lvszl/figure/master/20231031221837.png)



1. **研究目的**：研究人员试图确定是否可以将"AltUp"技术与"MoE"技术相结合，以获得T5模型在预训练阶段的性能提升。
2. **部分专家设置**：研究采用了类似于先前研究的"部分专家设置"，在每一层中，除了该层的模块之外，还将输入数据路由到一个较小的专家模块，并将主要模块和辅助模块的输出组合作为下一层的输入。
3. **MoE层**：MoE（Mixture of Experts）层用于将输入标记 x 路由到 n 个专家中的 k 个，其中每个专家本身是一个参数化的子网络，例如一个全连接层。权重矩阵 W 被应用于输入 x，以获得 logits h(x) = Wx。查找概率通过对 h(x) 进行 softmax 计算而得。标记 x 被路由到具有前 k 个概率 p(x) 的专家集 T ⊂ [n]。为了使梯度能够传播回路由参数，输出 y 被计算为专家输出的概率加权组合。
4. **MoE实现**：研究中使用了 [12] 的简化实现，采用了 top-1 softmax 路由，每个编码器和解码器层使用了 128 个专家，每个专家表示一个具有隐藏维度 16 的两层全连接神经网络。未采用诸如负载平衡损失（load balancing loss）或路由器 z 损失（router z loss）等负载平衡机制。噪声采用从 [1 - ε, 1 + ε] 的均匀分布中采样的乘法抖动噪声，其中 ε = 0.01。路由器矩阵 W 初始化为均值为零、标准差为 2 × 10^(-2) 的正态分布中采样的值。



# 参数的增加

假如要把原本参数变为K倍，

