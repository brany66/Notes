---
typora-copy-images-to: picture
---

#  机器学习中的偏差和方差

在机器学习领域我们总是希望使自己的模型尽可能准确地描述数据背后的真实规律。一般来说准确，其实就是误差小。在领域中，排除人为失误，人们一般会遇到三种误差来源：**随机误差、偏差和方差**。偏差和方差又与**欠拟合**及**过拟合**紧紧联系在一起。因为随机误差是不可消除的，所以此篇我们讨论在偏差和方差之间的权衡（Bias-Variance Tradeoff）。

> 准确是两个概念。准是 bias 小，确是 variance 小。准确是相对概念，因为 bias-variance tradeoff

## 数学定义

首先需要说明的是**随机误差**。随机误差是数据本身的噪音带来的，这种误差是不可避免的。一般认为随机误差服从高斯分布，记作$\epsilon\sim\mathcal N(0, \sigma_\epsilon)$，因此若有变量$y$作为预测值，以及$X$作为自变量（协变量），则将数据背后的真实规律$f$记作：
$$
y = f(X) + \epsilon
$$

- **偏差（bias）**描述的是通过学习拟合出来的结果之期望，与真实规律之间的差距，**刻画了学习算法本身的拟合能力** .

  记作$\text{Bias}(X) = E[\hat f(X)] - f(X)$

- **方差（variance）**即是统计学中的定义，描述的是通过学习拟合出来的结果自身的不稳定性，**刻画了数据扰动所造成的影响** ，记作：$\text{Var}(X) = E\Bigl[\hat f(X) - E[\hat f(X)]\Bigr]$

- 噪声：噪声表达了在当前任务上任何学习算法所能达到的期望泛化误差的下界, 即 **刻画了学习问题本身的难度** . 巧妇难为无米之炊, 给一堆很差的食材, 要想做出一顿美味, 肯定是很有难度的



以均方误差为例，公式$1$：
$$
\begin{equation}
\begin{aligned}
\text{Err}(X) &{}= E\Bigl[\bigl(y - \hat f(X)\bigr)^2\Bigr] \\
              &{}= E\Bigl[\bigl(f(X) + \epsilon - \hat f(X)\bigr)^2\Bigr] \\
              &{}= \left(E[\hat{f}(X)]-f(X)\right)^2 + E\left[\left(\hat{f}(X)-E[\hat{f}(X)]\right)^2\right] +\sigma_\epsilon^2 \\
              &{}= \text{Bias}^2 + \text{Variance} + \text{Random Error}.
\end{aligned}
\end{equation}
$$

其中：
$$
\sigma_\epsilon^2 =E[(y-\hat{y})^2]
$$
##  偏差和方差图示

下图将机器学习任务描述为一个**打靶**的活动：根据相同算法、不同数据集训练出的模型，对同一个样本进行预测；每个模型作出的预测相当于是一次打靶。**左上角**的示例是理想状况：**偏差和方差都非常小**。如果有无穷的训练数据，以及完美的模型算法，则可以实现这种情况。然而，现实中的工程问题通常数据量是有限的，而模型也是不完美的。因此，这只是一个理想状况。**右上角**的示例表**示偏差小而方差大**。落点都集中分布在红心周围，它们的期望落在红心之内，因此偏差较小。另外一方面，落点虽然集中在红心周围，但是比较分散，这是方差大的表现。**左下角**的示例表示**偏差大而方差小**。显而易见落点非常集中，说明方差小。但是落点集中的位置距离红心很远，这是偏差大的表现。**右下角**的示例则是最糟糕的情况，**偏差和方差都非常大**。

![bias-and-variance](D:\work\Notes\base\picture\bias-and-variance.png)

## 示例

首先生成了两组$array$，分别作为训练集和验证集。这里$x$与$y$是接近线性相关的，而在$y$上加入了随机噪声，用以模拟真实问题中的情况。然后选用最小平方误差作为损失函数，尝试用多项式函数去拟合这些数据。这里对于$prop$，采用了一阶的多项式函数（线性模型）去拟合数据；对于$overf$，我们采用 15 阶的多项式函数（多项式模型）去拟合数据。如此可以把拟合效果绘制成图。

![bias-and-varicance-test](D:\work\Notes\base\picture\bias-and-varicance-test.png)

```
#!/usr/bin/python
# -*- coding: utf-8 -*-
# create by YWJ, 2017.9.23

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42) 
realNum = lambda x : x + x ** 0.1

x_train = np.linspace(0, 15, 100)
y_train = list(map(realNum, x_train))
y_noise = 2 * np.random.normal(size=x_train.size)
y_train = y_train + y_noise

x_valid = np.linspace(0, 15, 50)
y_valid = list(map(realNum, x_valid))
y_noise = 2 * np.random.normal(size=x_valid.size)
y_valid = y_valid + y_noise

prop = np.polyfit(x_train, y_train, 1)
prop_ = np.poly1d(prop)
overf = np.polyfit(x_train, y_train, 15)
overf_ = np.poly1d(overf)

_ = plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
prop_e = np.mean((y_train - np.polyval(prop, x_train)) ** 2)
overf_e = np.mean((y_train - np.polyval(overf, x_train)) ** 2)
xp = np.linspace(-2, 17, 200)
plt.plot(x_train, y_train, '.')
plt.plot(xp, prop_(xp), '-', label='proper, err: %.3f' % (prop_e))
plt.plot(xp, overf_(xp), '--', label='overfit, err: %.3f' % (overf_e))
plt.ylim(-5, 20)
plt.legend()
plt.title('train set')

plt.subplot(1, 2, 2)
prop_e = np.mean((y_valid - np.polyval(prop,  x_valid)) ** 2)
overf_e = np.mean((y_valid - np.polyval(overf, x_valid)) ** 2)
xp = np.linspace(-2, 17, 200)
plt.plot(x_valid, y_valid, '.')
plt.plot(xp, prop_(xp), '-', label='proper, err: %.3f' % (prop_e))
plt.plot(xp, overf_(xp), '--', label='overfit, err: %.3f' % (overf_e))
plt.ylim(-5, 20)
plt.legend()
plt.title('validation set')
```

以训练集上的结果来说，**线性模型的误差要明显高于多项式模型**。站在观察者的角度来说：`数据是围绕一个近似线性的函数附近抖动的，那么用简单的线性模型自然就无法准确地拟合数据；但是高阶的多项式函数可以进行各种扭曲，以便将训练集的数据拟合得更好`

这种情况即是线性模型在训练集上欠拟合$(underfitting)$，并且它的偏差$bias$要高于多项式模型的偏差。但并不是说线性模型在这个问题里，要弱于多项式模型。在验证集上，**线性模型的误差要小于多项式模型的误差**，并且线性模型在训练集和验证集上的误差相对接近，而多项式模型在两个数据集上的误差很大。这种情况即是**多项式模型在训练集上过拟合**$\,(overfitting)$，并且它的方差$\,(variance)$要高于线性模型的偏差。此外因为线性模型在两个集合上的误差较为接近，因此说线性模型在训练过程中未见的数据上，**泛化能力**更好。因为在真实情况下，都需要使用有限的训练集去拟合模型，而这些真实样本对于模型训练过程都是不可见的。所以模型的泛化能力，是非常重要的指标。考虑到两个模型在验证集上的表现，在这个任务上，**线性模型**表现得较好。

## 如何权衡

对于很多人来说，不可避免地会有这样的想法：希望训练误差降至$0$。首先，对于误差，在公式$1$中，我们得知误差中至少有「随机误差」是无论如何不可避免的。因此，哪怕有一个模型在训练集上的表现非常优秀，它的误差是 0，这也不能说明这个模型完美无缺。因为，训练集本身存在的误差将会被带入到模型之中，也就是说这个模型天然地就和真实情况存在误差，于是它不是完美的。其次由于训练样本无法完美地反应真实情况（样本容量有限、抽样不均匀），以及由于模型本身的学习能力存在上限，也意味着模型不可能是完美的。因此不要刻意追求训练误差为0，反而去追求在给定数据集和模型算法的前提下的，逼近最优结果。

## 最佳平衡点

实际任务中，模型选择的方法：

- 选一个算法
- 调整算法超参数
- 以某种指标选择最合适的超参数组合

也就是说在整个过程中，在固定训练样本，改变模型的描述能力（模型复杂度）。因此随着模型复杂度的增加，其描述能力也就会增加；此时模型在测试集上的表现**，偏差会倾向于减小而方差会倾向于增大**。而随着模型复杂度的降低，其描述能力也就会降低；此时模型在验证集上的表现，**偏差会倾向于增大而方差会倾向于减小**。考虑到，模型误差是偏差与方差的加和，因此可以绘制出下图:

![bias-variance-tend](D:\work\Notes\base\picture\bias-variance-tend.png)

图中的最右位置，实际上是$total error$曲线的拐点。则连续函数的拐点意味着此处一阶导数的值为$0$。考虑到$total error$是偏差与方差的加和，所在拐点处：
$$
\begin{equation}
\newcommand{\dif}{\mathop{}\!\mathrm{d}}
\frac{\dif\text{Bias}}{\dif\text{Complexity}} = - \frac{\dif\text{Variance}}{\dif\text{Complexity}}
\end{equation}
$$
公式$2$给出了寻找最优平衡点的数学描述。若模型复杂度大于平衡点，则模型的方差会偏高，模型倾向于过拟合；若模型复杂度小于平衡点，则模型的偏差会偏高，模型倾向于过拟合。

## 过拟合和欠拟合的外在表现

尽管有了上述数学表述，但是在现实环境中有时候很难计算模型的偏差与方差。因此需要通过外在表现，判断模型的拟合状态：是欠拟合还是过拟合。同样地，在有限的训练数据集中，不断增加模型的复杂度，意味着模型会尽可能多地降低在训练集上的误差。因此在训练集上，不断增加模型的复杂度，训练集上的误差会一直下降。因此可以绘制出这样的图像

- 当模型处于欠拟合状态时，训练集和验证集上的误差都很高；
- 当模型处于过拟合状态时，训练集上的误差低，而验证集上的误差会非常高

![error-curve](D:\work\Notes\base\picture\error-curve.png)

## 如何处理过拟合与欠拟合

有了以上分析，就能比较容易地判断模型所处的拟合状态。接下来就可以参考$Andrew Ng$提供的处理模型欠拟合/过拟合的一般方法了。

![Machine-Learning-Workflow](D:\work\Notes\base\picture\Machine-Learning-Workflow.png)

### 欠拟合

当模型处于欠拟合状态时，根本的办法是增加模型复杂度。有以下办法：

- 增加模型的迭代次数
- 更换描述能力更强的模型
- 生成更多特征供训练使用
- 降低正则化水平

### 过拟合

当模型处于过拟合状态时，根本的办法是降低模型复杂度。有以下办法：

- 扩增训练集
- 减少训练使用的特征的数量
- 提高正则化水平


## 交叉验证

上述内容并没涉及到**模型选择**的问题，我们用训练集训练出来一个预测模型，就拿到测试集去计算预测误差。如果有一系列模型需要选择的话，当然不能用测试集去做选择，因为这相当提前偷看了测试集的数据，得到的测试误差r会低估真实的预测误差。**所以需要在测试集以外，另外寻找一个统计量作为预测误差的估计值，以这个统计量来作为模型选择的标准。** 其中验证误差需要另外找一个独立的**验证集**，因此在样本量不足,分割出一个验证集并不现实的情况就有$Cross Validation$的方法。

对于一系列模型$F(\hat{f},\,\theta)$, **使用Cross Validation的目的是获得预测误差的无偏估计量CV**，**从而可以用来选择一个最优的Theta\*, 使得CV最小。**假设$K-folds cross validation$，CV统计量定义为每个子集中误差的平均值，**而K的大小和CV平均值的bias和variance是有关**的：
$$
CV\,=\,\frac{1}{K}\sum_{k=1}^{K}\,\frac{1}{m}\,\sum_{i=1}^{m}\,(\hat{f}^k\,-\,y_i)^2
$$
其中，$m=\frac{N}{K}$代表每个subset的大小， N是总的训练样本量，K是subsets的数目。当K较大时，m较小，模型建立在较大的N-m上，因此CV与Testing Error的差距较小，所以说**CV对Testing Error估计的Bias较小。**同时每个Subsets重合的部分较大，相关性较高，**如果在大量不同样本中进行模拟，CV统计量本身的变异较大，所以说Variance高，反之亦然**

![CV-test](D:\work\Notes\base\picture\CV-test.png)

**(图中蓝色的线是10-folds Cross validation的CV统计量，放这个图不是为了讨论subset Size与bias的关系，而是想说，图中绿色的点和置信区间是Cross validation里的bias和variance，是针对Prediction Error估计的)**

## 参考文献以及链接

[1. Understanding the Bias-Variance Tradeoff ](http://scott.fortmann-roe.com/docs/BiasVariance.html)

[2. 机器学习中的Bias(偏差)，Error(误差)，和Variance(方差)有什么区别和联系？](https://www.zhihu.com/question/27068705)

[3. 偏差与方差](http://liuchengxu.org/blog-cn/posts/bias-variance/)

