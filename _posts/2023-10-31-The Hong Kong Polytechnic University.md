---

---

![](https://raw.githubusercontent.com/lvszl/figure/master/20231031172634.png)

解决的问题：时尚领域中数据稀疏和用户偏好多样化的挑战

关注的点：

1. 隐式的用户-物品交互
2. 颜色、风格和品牌对用户选择的影响

提出：Attentional Factor Field Interaction Graph (AFFIG)

具体而言：该方法模型化了用户因素交互和字段间交叉因素交互，来预测特定字段的推荐概率。

影响用户与物品之间交互的因素：颜色，风格，品牌，价格等。



AFFIG模型还构建了因子字段图FFG，来对跨字段的因子之间的影响进行建模。FFG是通过注意力机制来实现的。



# Problem Formulation（问题表述）

目的：是对字段级用户偏好和上下文因素吸引力进行建模，以预测特定因素字段（例如颜色和风格）的推荐概率，并进一步获得给定项目和用户的整体预测分数。

User集合：$U=\{u_t\}_{t=1}^{N_u}$ 

物品集合：$I=\{i_t\}_{t=1}^{N_i}$

 factor field set:$F = \{f_t\}_{t=1}^{N_f}$

并且使用$u，i，j$ 来表示user，item，$f~or~g$则表示定义了一个field。

用户和商品之间的交互集定义为$R = \{(u, i)\}$，描述了用户的历史购物行为。

对于每个因子字段$ f $，都有一个因子集 $A_f $，其中包含属于该字段的所有因子。

每一项 $i $都与一个因子列表 $a_i = [a^f_i for f ∈ F ] $相关联，列表中的每个因子都属于一个因子字段$ a^f_i ∈ A_f$ 。

输入：用户集$ U$ 、项目因子集 $\{a_i\}_{i∈I} $以及用户-项目交互$ R$

输出：一种预测模型，不仅输出给定用户-项目对 $(u, i)$ 的整体交互得分 $y_{ui}$，还输出时尚产品多个影响因素领域的特定交互得分 ${y^f_{ui}}$。



# 具体模型

![](https://raw.githubusercontent.com/lvszl/figure/master/20231031190632.png)

包含三个主要部分：(1) 用于初始化因子嵌入的因子嵌入层(Factor embedding layer)，(2) 用于建模交互模式的注意因子级交互层(attentional factor interaction layer)，以及 (3) 预测因子级的层(prediction layer)和整体互动得分。



注意在Attentional factor interaction layer上：

![](https://raw.githubusercontent.com/lvszl/figure/master/20231031191505.png)

该项目的跨字段因素之间的相互作用以类似的方式建模，如下所示：

![](https://raw.githubusercontent.com/lvszl/figure/master/20231031191714.png)

然后通过注意力机制聚合每个因素的所有跨领域因素之间的相互作用：

![](https://raw.githubusercontent.com/lvszl/figure/master/20231031191815.png)

最后，将两个交互部分组合在一起，得到场 f 的最终表示：

![](https://raw.githubusercontent.com/lvszl/figure/master/20231031191953.png)



在prediction layer上：

特定领域偏好：给定每个因素领域与用户交互后的最终表示以及跨领域因素，可以预测特定领域用户的时尚偏好。对于字段 f ，预测为：

![](https://raw.githubusercontent.com/lvszl/figure/master/20231031192047.png)

其中，$h_f$为待确定参数。

整体偏好：获得因子领域特定的推荐预测后，可以进一步结合领域级别的预测获得整体推荐得分。在我们的实现中，简单地总结并获得最终的概率得分：

![](https://raw.githubusercontent.com/lvszl/figure/master/20231031192137.png)



# 训练

损失函数采用BPR（成对贝叶斯个性化排名损失）

![](https://raw.githubusercontent.com/lvszl/figure/master/20231031192224.png)

