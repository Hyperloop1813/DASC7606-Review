# DASC7606 Final - summarized by hyperloop

# Section 0 Scope

![Scope](./Scope.png)

# Section 1 DeepLearning Basis

## Q1 2024 May ABC

![Q3](./2024MayABC/7.png)

## Answer

### (1) 训练误差低但测试误差高：原因？哪些动作能改善？
**原因：过拟合（overfitting）**  
模型容量/训练轮数过大，学习到了训练集的噪声或偶然模式，泛化能力差 → 测试误差高。

对选项逐条判断：
1. **increase the size of the network（增大网络）**：❌ 通常更容易过拟合，测试误差可能更高。
2. **use data augmentation（数据增强）**：✅ 增加“有效数据量”，让模型见到更多变化，提升泛化。
3. **stop training earlier（提前停止/early stopping）**：✅ 防止后期开始记忆噪声，降低过拟合。
4. **use ReLU as the activation function（用ReLU）**：⚠️ 不直接针对过拟合；可能改善优化/梯度消失，但不一定降低测试误差（需配合正则化等）。
5. **dropout is applied differently during training and testing**：✅ 正确做法本来就是训练时启用dropout、测试时关闭并做缩放（等效集成），能抑制过拟合。
6. **change the cost function, explain how（改变损失函数）**：✅ 若“改变”为加入正则项则能改善，例如：
   - **L2 weight decay**：`L' = L + λ||W||^2`（抑制大权重，降低模型复杂度）
   - **L1 正则**：`L' = L + λ||W||_1`（促稀疏）
   - 分类还可用 **label smoothing** 等降低过度自信、改善泛化

### (2) 测试误差显著低于训练误差：可能原因？
可能原因（任意写出几点即可）：
- **训练阶段启用了正则化/噪声**：如 dropout、数据增强、BN 训练模式带噪声 → 训练更难；测试时关闭/更稳定 → 测试误差更低。
- **训练误差计算口径不同**：训练用 mini-batch 即时loss、含正则项；测试用全量平均、且不含正则项 → 看起来测试更低。
- **训练集更“脏/更难”**：训练集含噪声标签、强增强；测试集更干净或更容易。
- **小样本波动**：测试集很小，偶然抽到更容易的样本 → 指标偏乐观。
- **数据泄漏/划分问题**：测试集中混入与训练高度相似/重复样本，也会导致测试异常低（需要警惕）。


Any of the following:
- **Regularization/noise during training** (dropout, data augmentation, BatchNorm in train mode) makes training harder; at test time these are disabled/stable → lower test error.
- **Different measurement protocol**: training loss reported per mini-batch, possibly includes regularization term; test loss is averaged over the full set and may exclude regularization → test looks lower.
- **Training set is harder/noisier**: label noise or aggressive augmentation in training; cleaner/easier test set.
- **Small-sample variance**: test set is small and happens to be easier → optimistic test metric.
- **Data leakage / split issues**: overlap or near-duplicates between train and test can make test error unrealistically low.


### (3) 小 batch 时：batch 越大，达到目标loss所需迭代次数越少，为什么？

当 batch 很小，**梯度估计方差很大（噪声大）**，每次更新方向不稳定，需要更多迭代才能收敛。  
增大 batch 会让梯度更接近“全数据真实梯度”，**更新更稳定、有效步长更大** → 达到目标loss所需迭代次数下降。
With very small batches, the gradient estimate has **high variance (noisy)**, leading to unstable updates and slower convergence (more iterations).  
Increasing batch size reduces gradient noise, making updates closer to the true gradient → **more stable and effective steps** → fewer iterations to reach target loss.

### (4) 大 batch 时：继续增大 batch，迭代次数变化不大，为什么？

batch 足够大后，梯度估计已经很接近全量梯度，**方差进一步下降的收益递减**；  
优化速度更多受 **学习率、曲率/条件数、模型与数据本身**限制，而不是梯度噪声限制，  
因此继续增大 batch 对“需要的迭代次数”提升很小（边际收益很低）。

Once the batch is large enough, the gradient is already close to the full-batch gradient, so **variance reduction has diminishing returns**.  
Convergence is then limited mainly by **learning rate, curvature/conditioning, and problem structure**, not by gradient noise, so further increasing batch size yields little change in required iterations.

---

## Q2 2024 Dec C

![Q6](./2024DecC/1.png)

### Answer

### (a) 零中心激活函数的优势 (不考)

如果输出是零中心的，梯度下降会被限制在“好”的方向，从而收敛更快
→ 不完全准确，零中心主要使梯度方向更合理，但不是“限制”方向，而是避免 zig-zag 更新（当所有输入为正时，权重梯度同号，更新路径差）。所以这个说法有点模糊，但意图是指零中心帮助优化。

如果输出是零中心，梯度通常取更广的值，通常导致更好收敛
→ 合理，零中心使梯度有正有负，更新更灵活。

因为这解决了梯度消失问题
→ 错误，零中心不能直接解决梯度消失（这更多与导数幅度、权重初始化、网络结构有关）。

如果激活函数输出全正，且激活函数是线性的，那么梯度要么全非正要么全非负
→ 正确，这是零中心重要性的一个关键原因：若输入全正，则某权重的梯度符号取决于上一层误差项，可能所有样本对该权重的梯度同号，导致权重更新只能沿特定方向曲折前进，收敛慢。

因为这保证没有过拟合
→ 错误，零中心与过拟合无直接必然联系。

正确选项：1, 2, 4

### (b) 反向传播相关说法 (不考)

反向传播算法能高效计算神经网络中的梯度
→ 正确，这正是 BP 的核心作用。

反向传播包含两个阶段：第一阶段计算梯度，第二阶段通常运行一步梯度下降，这通常导致后者更好收敛
→ 错误，两个阶段是前向传播（算输出）和反向传播（算梯度），梯度下降是优化步骤，不是 BP 的一部分。

反向传播可以高效并行化：此时运行时间不依赖于网络层数，而只依赖于 GPU 数量
→ 错误，即使并行，每层依赖前一层的梯度，所以时间仍与层数有关，不能完全并行所有层。

正确选项：1

### (c) ResNet 相关说法 (可能考)

ResNet 的主要目标是任意逼近恒等函数。因为大多数现实问题可由恒等函数近似，ResNet 结果很好
→ 错误，ResNet 确实让网络容易学习恒等映射，但这不是因为现实问题近似恒等，而是为了缓解深层网络退化问题。

更深的网络应提供更好的训练和测试误差，但它们更难训练
→ 正确，这是 ResNet 提出的动机：理论上更深网络表达能力更强，但普通网络加深时训练误差反而上升（退化问题）。

ResNet 缓解了深层网络学习恒等函数的问题，同时允许学习更复杂函数
→ 正确，残差块让网络能轻松学习恒等映射（通过跳过连接），如果需要变化就学习残差部分。

正确选项：2, 3

---
## Q3 2024 May D

![Q8](./2024MayD/1.png)

### Answer

### 1.1 为什么测试集最好只用一次？

**题目分析（中文）**： 
如果多次在测试集上评估模型，并根据测试集结果调整模型或超参数，那么测试集会间接影响训练过程，相当于把测试集信息“泄漏”到了模型中。这样测试集就不再能提供对模型泛化能力的无偏估计，而是会给出过于乐观的结果。

**中文答案**：  
因为测试集用于最终一次的泛化性能估计。多次使用测试集来比较模型或调参会造成数据泄漏/选择偏差，使模型对测试集过拟合，从而得到偏高且不可信的测试精度。应使用训练集+验证集（或交叉验证）做选择与调参，测试集留到最后一次评估。

**English answer**：  
Because the test set should provide an **unbiased final estimate** of generalization. Repeatedly evaluating on the test set and using it for model selection/hyperparameter tuning causes **test leakage** and **overfitting to the test set**, yielding an **optimistically biased and unreliable** estimate. Use a validation set (or cross-validation) for tuning and keep the test set for a one-time final evaluation.

### 1.2 一种缓解病态损失函数问题的梯度下降变体 （可能不考）

**题目分析（中文）**：  
ill-conditioned 常表现为“狭长谷地/各方向曲率差异大”，普通 GD 容易在陡峭方向震荡、在平缓方向很慢。能缓解的算法通常通过**动量**或**自适应缩放/预条件化**来加速与稳定。

**中文答案（示例）**：  
可以用 **Momentum（动量法）**（或 **Nesterov Momentum**）。动量累计历史梯度形成速度项，能减少高曲率方向的来回震荡，并在低曲率方向加速前进，从而改善病态条件数导致的收敛慢问题。  
（也可答 **Adam / RMSProp / AdaGrad**：对不同参数维度自适应缩放梯度，具有类似预条件化效果。）

**English answer (example)**：  
Use **Momentum (heavy-ball)** or **Nesterov momentum**. Accumulating past gradients into a velocity term **damps oscillations** in high-curvature directions and **accelerates** movement in low-curvature directions, which helps for **ill-conditioned** objectives.  
(Alternatively: **Adam/RMSProp/AdaGrad**—adaptive per-parameter scaling acts like **preconditioning**.)

课程中常见的标准答案是 Momentum 或 Adam。

### 1.3 LSTM 的三个主要门 (不考)


**中文答案**：  
**输入门（Input gate）**、**遗忘门（Forget gate）**、**输出门（Output gate）**。

**English answer**：  
**Input gate**, **Forget gate**, **Output gate**.

![Q8](./2024MayD/6.png)
![Q8](./2024MayD/7.png)
![Q8](./2024MayD/8.png)
![Q8](./2024MayD/9.png)
![Q8](./2024MayD/10.png)

---

## Q4 2023 May ABC

![Q9](./2023MayABC/1.png)


### (a) (9 pts) 选择 Precision 或 Recall，并解释

### 中文题目分析
核心看你更怕哪种错误：  
- **更怕漏报（False Negative, FN）** → 选 **Recall（召回率）**  
- **更怕误报（False Positive, FP）** → 选 **Precision（精确率）**

### (i) Detect driver fatigue to prevent any car accidents.
**中文答案：Recall（召回率）更重要。**  
漏检疲劳驾驶（FN）可能导致事故，代价极高；误报（FP）通常只是多提醒一次。

**English answer: Recall.**  
False negatives (missing a fatigued driver) can be catastrophic; false positives are usually just extra warnings.


### (ii) Screen thousands of job applications for interview.
**中文答案：Precision（精确率）更重要。**  
面试资源有限，误选大量不合格者（FP）会浪费面试时间与成本；被筛入的人应尽量“真的合格”。  
（实际中常用两阶段：先高Recall粗筛，再高Precision精筛。）

**English answer: Precision.**  
Interviews are limited/costly, so shortlisted applicants should be truly qualified (reduce false positives).  
(In practice, a two-stage pipeline often uses high recall first, then high precision.)


### (iii) Detect and raise an alarm for fire.
**中文答案：Recall（召回率）更重要。**  
漏报火灾（FN）后果严重；误报（FP）虽然影响体验但代价通常更低。

**English answer: Recall.**  
Missing a real fire (false negative) is far more dangerous than a false alarm.


### (b) (9 pts) 训练现象分析：你会怎么做？

### (i) Training converges, but the training error is high/very high.
**中文分析：欠拟合（Underfitting）/高偏差（High bias）。**  
**中文做法：**
- 增大模型容量：更深/更宽/更强架构
- 减弱正则化：降低 weight decay / dropout
- 改善数据与特征：更好预处理、修正标签噪声、处理类别不平衡
- 调整优化：换优化器（Adam/SGD+momentum）、调学习率/训练更久（若并非真正充分训练）

**English answer: Likely underfitting / high bias.**  
Actions: increase capacity, reduce regularization, improve data/features and labels, handle imbalance, tune optimizer and learning rate (and train longer if appropriate).

### (ii) Training converges and training error is low, but testing error is high/very high.
**中文分析：过拟合（Overfitting）/高方差（High variance）。**  
**中文做法：**
- 更多数据 / 数据增强（augmentation）
- 加强正则化：dropout、L2 weight decay、label smoothing
- early stopping（提前停止）
- 降低模型复杂度：更小网络、剪枝
- 检查数据泄漏/分布偏移：划分是否正确、是否有重复样本、train/test 分布是否不同

**English answer: Likely overfitting / high variance.**  
Actions: more data/augmentation, stronger regularization, early stopping, reduce model size, and check data leakage or distribution shift.

### (iii) Training error does not converge but increases.
**中文分析：训练发散/数值不稳定/实现或数据问题。**  
常见原因：学习率过大、梯度爆炸、输入未归一化、标签/损失实现错误、train/eval 模式错误等。  
**中文做法：**
- 降低学习率（最优先），或使用学习率衰减
- 使用更稳定优化器（如 Adam），调整 momentum
- 梯度裁剪（gradient clipping）防止 exploding gradients
- 检查数据与实现：输入归一化、标签是否错位、loss 是否正确、BN/Dropout 模式是否正确
- 使用合理初始化（He/Xavier）

**English answer: Likely divergence/instability or a bug/data issue.**  
Actions: lower learning rate, use a stable optimizer, apply gradient clipping, verify normalization and labels/loss implementation, check train/eval modes, and use proper initialization.


![Q9](./2023MayABC/2.png)

### (c) (6 pts) 题目分析（中文）
这题考的是：**反向传播不更新网络权重，而是把“图像像素”当作可优化变量**，对像素做梯度上升/下降来达到某种目标。  
要求：用文字描述（不需要公式）——写出两种典型场景分别在优化什么“函数/目标”。

常见两类代表性应用：
1) **对抗样本（Adversarial example）**：改像素让模型把图像分错/指定成某类。  
2) **特征可视化 / DeepDream / 最大化某个神经元或类别得分**：改像素让某层/某个滤波器/某个类别的激活更大。  
（也可答：Neural Style Transfer，优化内容损失+风格损失+平滑正则。）

### Answer (i) / 答案 (i)
### 中文
**对抗样本生成**：把“像素”当变量，优化一个目标，使得网络对这张图的预测发生改变（例如让它被分类成某个指定类别，或让原本正确分类变成错误分类），同时通常会限制像素改动要很小、肉眼几乎看不出来。

### English
**Adversarial example generation**: treat the image pixels as variables and optimize an objective that changes the network’s prediction (e.g., force a targeted class or cause misclassification), usually while constraining the perturbation to be small and visually imperceptible.


### Answer (ii) / 答案 (ii)
### 中文
**特征/类别激活可视化（DeepDream / feature visualization）**：把“像素”当变量，优化一个目标来**最大化**网络中某个层的某个神经元/通道（滤波器）的激活，或最大化某个类别的得分，从而生成一张能强烈触发该特征/类别的图像，用于理解网络学到了什么。

### English
**Activation / feature visualization (DeepDream)**: treat pixels as variables and optimize an objective to **maximize** the activation of a chosen neuron/channel/layer (or maximize a class score), producing an image that strongly excites that feature/class to interpret what the network has learned.


---


## Q5 2023 May D

![Q10](./2023MayD/1.png)

## Q6 2022 May 1

![Q11](./2022May1/1.png)

## Q7 2022 May 1

![Q12](./2022May1/2.png)

---

# Section 2 NLP

## Q1 2023 May ABC

![Q2](./2023MayABC/5.png)
![Q2](./2023MayABC/6.png)
![Q2](./2023MayABC/7.png)

---


# Section 3 Convolutional Neural Networks

## Q1 2025 May

![Q1](./2025May/9.png)
![Q1](./2025May/10.png)

### Answer

### (a) 为什么三层 3×3 卷积的有效感受野相当于一层 7×7 卷积

第一层 3×3 卷积：每个输出神经元看到输入 3×3 区域。

第二层 3×3 卷积：每个神经元看到第一层输出的 3×3 区域，而第一层的每个神经元对应输入 3×3，所以第二层输出神经元看到的输入区域是 3 + 3 - 1 = 5×5（两个 3×3 卷积堆叠，未经 padding 缩小的情况）。

第三层 3×3 卷积：同理，看到第二层输出的 3×3 区域，第二层每个神经元对应输入 5×5，所以第三层输出神经元看到的输入区域是 5 + 3 - 1 = 7×7。

因此三层 3×3 卷积的有效感受野 = 7×7。

答 (a)：
每个 3×3 卷积增加 2 到感受野尺寸，三层堆叠：1→3→5→7

### (b) 为什么三层 3×3 比一层 7×7 参数少

假设输入输出通道数均为 C。

一层 7×7 卷积参数：7×7×C×C = 49C²

三层 3×3 卷积参数：每层 3×3×C×C = 9C²，三层共 3×9C² = 27C²

显然 27C² < 49C²。

答 (b)：
27C² < 49C²

### (c) 保持尺寸的 padding 和 stride

输入 28×28×192，输出 28×28×128，用 3×3 卷积。
要保持空间尺寸 28×28，需要 padding = 1

（因为输出尺寸公式 H_out = H_in + 2P - K / S + 1，当 H_in = 28，K = 3，S = 1，要 H_out = 28，则 28 = 28 + 2P - 3 + 1 ⇒ 2P - 2 = 0 ⇒ P = 1）。

通道数由 192 变 128 是靠使用 128 个滤波器。

答 (c)：
padding = 1, stride = 1

### (d) 乘法次数（直接 3×3 卷积）

输入：28×28×192

卷积核：3×3×192，共 128 个滤波器

每个输出位置：3×3×192 次乘法

输出位置数：28×28×128

乘法次数：
(3×3×192)×(28×28×128) = 1728×100352 = 173,408,256

答 (d)：
173,408,256

### (e) 先用 1×1 卷积降维再 3×3 卷积

第一步：1×1 卷积，输入 192 通道 → 64 通道，
每个输出位置乘法：1×1×192，
输出位置数：28×28×64

乘法次数：192×(28×28×64) = 192×50176 = 9,633,792

第二步：3×3 卷积，输入 64 通道 → 128 通道，
每个输出位置乘法：3×3×64，
输出位置数：28×28×128

乘法次数：576×100352 = 57,802,752

总乘法次数：9,633,792 + 57,802,752 = 67,436,544

技术名称：瓶颈层（Bottleneck Layer）或 1×1 卷积降维（来自 GoogleNet/Inception 的思想）。

答 (e)：
67,436,544, Bottleneck

### (f) 采用 (e) 的方法节省了多少成本

(d) 直接 3×3 卷积：

173,408,256 次乘法
(e) 1×1 瓶颈层 + 3×3 卷积：
67,436,544 次乘法

节省：
173,408,256 - 67,436,544 = 105,971,712

节省比例：
105,971,712 / 173,408,256 ≈ 0.611（约 61.1%）

答 (f)：
105,971,712 次乘法节省（约 61%）

### (g) ResNet 中训练极深网络的技术

ResNet 的核心创新是残差连接（skip connection / residual connection）。

它让网络层学习残差映射 F(x) = H(x) - x，而不是直接学习 H(x)。这样即使深层网络恒等映射是最优时，也可以让 F(x) → 0 来轻松实现，避免了梯度消失和网络退化问题。

答 (g)：
残差连接（skip connection / residual connection）

### (h) GoogleNet 中训练深度网络的技术

GoogleNet (Inception v1) 的主要技术是 Inception 模块，它使用并行多尺度卷积（1×1, 3×3, 5×5 和池化）并利用 1×1 卷积降维（bottleneck）来控制计算量，使网络既宽又深而不至于计算量爆炸。

此外，GoogleNet 还使用了辅助分类器（auxiliary classifiers）在中间层加入损失，帮助梯度传播缓解消失问题。

题目问“什么技术使深度网络没有显著性能损失”，主要答案是 Inception 模块（含 bottleneck）。

答 (h)：
Inception 模块（包含 1×1 卷积降维）

---

## Q2 2024 May ABC

![Q2](./2024MayABC/1.jpg)
![Q2](./2024MayABC/2.jpg)

## Q3 2024 Dec ABD

![Q3](./2024DecABD/3.png)

### Answer

### (a)

After applying filter F1After applying filter F2

$\begin{pmatrix} 3 & 3 & 3 & 3 & 3 \\ 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 \\ -3 & -3 & -3 & -3 & -3 \end{pmatrix}$$\begin{pmatrix} -3 & -3 & -3 & -3 & -3 \\ 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 \\ 3 & 3 & 3 & 3 & 3 \end{pmatrix}$

### (b)

F1Horizontal Edge Detector: (Top $0 \to 1$ and Bottom $1 \to 0$ with $\mathbf{+3}$ and $\mathbf{-3}$ response).

It detects changes in intensity along the vertical axis (a horizontal edge). It highlights the region where the patch below is brighter than the patch above it.

F2Horizontal Edge Detector (Top $0 \to 1$ and Bottom $1 \to 0$ with $\mathbf{-3}$ and $\mathbf{+3}$ response).

It detects the same horizontal edges but with the opposite polarity/sign compared to F1. F1 and F2 together provide two ways to represent the same horizontal features.

### (c)

在将 F1 和 F2 的结果相加之前，常用 ReLU 作为激活函数，因为：

边缘检测结果可能有正有负，但负值可能表示反向边缘，有时我们只关心边缘强度而不关心方向（或者两个方向都保留为正）。

如果使用 ReLU，可以去除负响应，只保留正边缘，避免正负抵消在相加时丢失信息。

另一种选择是 绝对值，但标准 CNN 用 ReLU 更常见。

---

### Common Convolutional Filters

### 1. Vertical Edge Detector (e.g., Sobel or Prewitt X-axis)

This is the counterpart to the horizontal edge filter (F1/F2) you just analyzed.Vertical Edge Filter

$$
\begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix}
$$

Detects vertical edges by calculating the difference in intensity between the columns. A large positive output indicates a light-to-dark transition moving from left to right, while a large negative output indicates the reverse.

### 2. Gaussian Blur Filter (Smoothing)

Gaussian Blur Filter

$$
\frac{1}{16} \begin{pmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{pmatrix}
$$

Smooths or blurs an image by calculating a weighted average of pixels. The weights are highest in the center (4) and decrease toward the edges (1), following a Gaussian distribution. This is used to reduce noise and detail before feature extraction, making the edge detection more robust.

### 3. Sharpening FilterSharpening Filter

$$
\begin{pmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{pmatrix}
$$

Enhances the edges and fine details of an image. It works by subtracting the blurred version of the image from the original image (or an approximation of this). The high positive weight (5) in the center emphasizes the central pixel relative to its neighbors.

### 4. $1 \times 1$ Convolution Filter (Bottleneck/Channel Mixer)

1×1 Filter Concept: A single $1 \times 1 \times C_{in}$ kernel, where $C_{in}$ is the number of input channels.

This filter is primarily used for two purposes: Dimensionality Reduction (the "bottleneck" technique) and Channel Mixing. It projects the input channels into a lower-dimensional space, significantly reducing computation cost and parameter count without losing spatial resolution. It can also act as a simple channel-wise fully connected layer to learn complex relationships between feature maps.

---

## Q4 2024 Dec ABD

![Q4](./2024DecABD/5.jpg)
![Q4](./2024DecABD/6.jpg)

## Q5 2024 Dec C

![Q5](./2024DecC/5.png)

## Q6 2024 May D

![Q6](./2024MayD/2.png)
![Q6](./2024MayD/3.png)

## Q7 2023 May ABC

![Q7](./2023MayABC/3.png)
![Q7](./2023MayABC/4.png)

## Q8 2022 May 1

![Q8](./2022May1/3.png)

---

# Section 4 Large Language Model

## Q1 2025 May

![Q2](./2025May/7.jpg)
![Q2](./2025May/8.jpg)

### (c) self-attention 后 dog 的嵌入如何变化

self-attention 会让每个位置的输出嵌入变成其他所有位置输入的加权组合。

这里 “dog” 在位置 6，它的新嵌入会融合 “Bill hate #s big black dog #s” 中其他词的信息。
例如：

“big” 和 “black” 是形容词，可能修饰 “dog”，因此 “dog” 的嵌入会获得这些形容词的特征。

“Bill hate #s” 可能表达情感，也可能影响 “dog” 的上下文表示。

所以 “dog” 的嵌入会从独立的 (-2.0, 0.5, 0.0, 0.5) 变成一个包含全局上下文信息的向量，更少依赖于原始词嵌入，更多依赖于它在句子中的语义角色。

---

## Q2 2024 May ABC

![Q3](./2024MayABC/3.jpg)
![Q3](./2024MayABC/4.jpg)
![Q3](./2024MayABC/5.png)

---

## Q3 2024 Dec ABD

![Q4](./2024DecABD/7.png)
![Q4](./2024DecABD/8.png)

### Answer by DS

### (a) (6 pts) Transformer架构优于RNN/LSTM的三个主要优点

答案：

直接的全局长程交互

RNN/LSTM是顺序处理，信息在序列中逐步传递，距离较远的token间交互困难，存在长程依赖问题。

Transformer通过自注意力机制，允许序列中任意两个位置直接交互，无论距离远近，都能建立直接依赖关系。

高度并行化计算

RNN/LSTM必须按时间步顺序计算，无法充分利用现代硬件的并行计算能力。

Transformer的自注意力层和前馈网络可以对序列中的所有元素同时计算，极大提升了训练和推理效率。

灵活且更大的嵌入维度

RNN/LSTM受限于计算效率，通常使用相对较小的隐藏状态维度。

Transformer的并行架构使其能够支持更大的嵌入维度和隐藏层维度，从而学习更丰富、更复杂的表示，而不会显著增加训练时间。

![Q4](./2024DecABD/9.png)

### (b) (4 pts) Transformer如何解决(a)中提出的问题？

答案：

针对问题1（全局依赖性）：通过自注意力机制 解决。在计算某个位置的表示时，它会直接关注并聚合序列中所有其他位置的信息，从而立即建立全局依赖关系，无需像RNN那样通过多个时间步传递。

针对问题2（并行化）：通过摒弃循环结构并完全依赖前馈网络和矩阵乘法 解决。整个输入序列被作为一个整体输入模型，所有注意力权重和层内变换都可以通过高效的矩阵运算并行完成。

### (c) (6 pts) 注意力机制中K, Q, V的解释

(i) 超市找物品场景：

查询：每个购物者心中的物品名称。它代表了“我需要什么”。

键：超市货架上每个商品的标签名称。它代表了“我是什么”。

值：货架上的商品本身。一旦通过匹配Q和K找到了正确的物品，你最终拿取的就是这个V。

过程：购物者（Q）将自己的需求与货架上所有商品的标签（K）进行比对。找到匹配度最高的标签后，就取走该标签对应的实际商品（V）。

(ii) 分配最近Uber车辆场景：

查询：乘客的乘车请求和位置。它代表了“我需要一辆车在这里”。

键：所有可用Uber车辆的位置和状态。它代表了“我是一辆可用的车，我在这里”。

值：可用的Uber车辆本身。它是最终被分配给乘客的实体。

过程：系统将乘客的请求（Q）与所有可用车辆的信息（K）进行匹配，计算出一个“距离分数”。选择分数最高（即最近/最合适）的车辆（V）分配给该乘客。

### (d) (3 pts) 编码器-解码器注意力机制的实现及原因

答案：

实现方式：

查询 来自解码器的前一层输出。

键和值 均来自编码器的最终输出。

原因：
这种实现允许解码器在生成每一个目标语言词汇时，有选择地关注输入序列中最相关的部分。它让解码器能够“查阅”完整的编码源信息，而不是仅仅依赖解码器自身的隐藏状态。这模仿了人类在翻译时的行为：在写出下一个词之前，会回看整个源句子的意思以确定最佳表达。

### (e) (4 pts) Transformer架构的缺点

答案：

计算复杂度高：自注意力机制的计算复杂度与序列长度的平方成正比（O(n²)），这使得处理非常长的文档或高分辨率图像变得非常昂贵和困难。

位置信息需要显式注入：由于自注意力本身是置换不变的（不关心顺序），模型必须依赖额外添加的位置编码来理解序列中元素的顺序，这不如RNN/LSTM内生的顺序处理机制直接。

巨大的内存和计算资源消耗：庞大的模型参数量、大的激活值和注意力矩阵需要大量的GPU内存，限制了其在资源受限环境中的应用。

缺乏归纳偏置：虽然其通用性是个优点，但这也意味着它需要比RNN更多的数据来学习一些固有的序列特性（如局部性），可能导致在小型数据集上过拟合。

---

### Answer by GPT

### (a) (6 pts) Three main advantages that make Transformer superior to RNN/LSTM

Advantages:

Parallel computation –

Transformers process all tokens simultaneously instead of sequentially (no recurrence).

→ Much faster training, especially on GPUs.

Long-range dependency modeling –

Self-attention lets every token directly attend to every other token.

→ Captures global context better than RNNs/LSTMs, which suffer from vanishing gradients over long sequences.

Fixed embedding dimension (independent of sequence length) –

Word embeddings and positional encodings have a constant size.

→ Enables stable representations regardless of input length.

### (b) (4 pts) How Transformer solves RNN/LSTM problems

Removes sequential dependency: self-attention computes relationships among tokens in parallel, avoiding step-by-step recurrence.

Handles long context: attention weights connect distant words directly (no gradient decay through time).

Stable embeddings: constant-dimension positional + word embeddings replace hidden states whose size and quality degrade over long sequences.

### (c) (6 pts) Interpretation of K, Q, V (Keys, Queries, Values)

Scenario	Analogy	Explanation
(i) Supermarket example	- Each person = a Query (Q) (they are “asking” for something).

- Each shelf label = a Key (K) (what items are available).
- Each item on the shelf = a Value (V) (the actual content retrieved).	The person (Q) scans all shelves (K) to find the best match and then picks the corresponding item (V).
  (ii) Uber example	- Each passenger request = a Query (Q).
- Each car’s location = a Key (K).
- Each car’s details (driver info, ETA, etc.) = a Value (V).	The system computes attention (similarity) between passenger Q and car K to assign the closest car (retrieve its V).

### (d) (3 pts) Encoder-decoder attention mechanism

In encoder–decoder attention, the decoder’s queries (Q) attend to the encoder’s output keys (K) and values (V).

This lets each decoder token focus on the most relevant encoder tokens when generating output (e.g., aligning target words with source words in translation).

Reason: enables information flow from the input sentence (encoder) to the output sentence (decoder) for accurate sequence generation.

### (e) (4 pts) Shortcomings of Transformer architecture

High computational and memory cost:

Self-attention scales as O(n²) with sequence length.

Requires large datasets to train effectively.

Limited inductive bias for sequential order:

Positional encoding is less intuitive than recurrence for time-dependent data.

Interpretability and efficiency issues:

Many attention heads are redundant; not easily interpretable.

---

## Q5 2025 May

![Q5](./2025May/12.png)

### Answer

### (a) Transformer 中的 Masking

在 Transformer 中，Masking 主要用于 Decoder 的自注意力层，目的是防止在训练时“偷看”未来的信息。

在自回归生成任务（如机器翻译）中，Decoder 在预测第 t 个位置时，只能使用前 t 个位置的信息。

通过 注意力掩码（一个下三角为 0、上三角为 -inf 的矩阵），将未来位置的注意力权重设为 0，确保模型只能关注已生成的部分。

这样，训练时即使一次性输入整个目标序列，也不会信息泄漏。

### (b) BERT 中的 Masking

BERT 使用 Masking 作为 预训练任务（Masked Language Model, MLM）。

在输入序列中，随机遮盖（Mask）一部分 token（如 15%）。

模型的任务是基于上下文（双向信息）预测被遮盖的原始 token。

这使得 BERT 能学习深层的双向语言表征，而不是像从左到右的语言模型那样只看到上文。

注意：BERT 在预训练时使用 [MASK] 符号，但在微调时所有输入 token 都是真实词，不存在 [MASK]，这带来一定的预训练-微调差异，BERT 通过替换部分遮盖词为随机词或原词来缓解。

### (c) 视觉中的遮挡敏感性

遮挡敏感性是一种 可解释性/分析技术，用于理解 CNN 等模型依赖图像的哪些区域进行预测。

方法：在输入图像上用一个灰色方块或模糊区域遮挡一小块区域，然后观察模型预测概率的变化。

通过系统性地滑动遮挡块并记录预测得分，可以生成一个“敏感性图”：如果遮挡某区域导致预测概率大幅下降，说明该区域对模型决策很重要。

这类似于 NLP 中遮盖一个词看句子概率的变化，但在视觉中用于定位关键图像区域。

---

# Section 5 Generative Adversarial Networks

## Q1 2025 May

![Q1](./2025May/11.png)

### Answer

### (a)

GAN 的原始论文中，生成器和判别器的优化目标可以写成 二元交叉熵（Binary cross-entropy） 的形式，也就是 Binary logistic loss。

A 交叉熵损失（Cross-entropy loss）一般指多分类交叉熵，虽然二元交叉熵是它的特例，但通常不直接这么说。

B 均方误差损失（MSE）不是 GAN 原始版本常用的，尽管有些变体（如 LSGAN）用了 MSE。

C 二元逻辑损失（Binary logistic loss）就是二元交叉熵，这是原始 GAN 用的。

D Softmax 损失用于多分类，不是 GAN 判别器二分类的标配。

因此答案是 C。

### (b)

**DeepSeek**

A 生成器的输入一般是随机噪z，不同的z产生不同样本，因此与样本多样性有关。

B 调整生成器的权重和偏差会影响生成样本的分布，因此也影响多样性。

C 模式崩溃（Mode collapse）是 GAN 训练失败的一种情况，导致生成样本缺乏多样性，因此与多样性有关（负面相关）。

D 条件 GAN（Conditional GAN）通过附加条件信息控制生成样本的类别，可以增加多样性（例如按类别生成不同样本）。

因此 A、B、C、D 都相关。

**Gemini**

Sample variations refer to the ability of the GAN to produce diverse outputs.

A. Input in the generator component of a GAN: The generator's input is a random noise vector ($z$), which is the primary source of variation in the generated output. Different noise vectors lead to different generated images.

B. Adjusting the weights and biases of the generator: This relates to training the generator to learn the data distribution, which is how it enables variation, but the input noise is the immediate cause of variation for a trained generator.

C. Mode collapse in GANs: This is a problem where the generator fails to produce diverse outputs (i.e., it only generates a small subset of the true data distribution's "modes"). It is fundamentally related to sample variations, but as a failure condition. If the question implies things that enable or govern variation, A is better. If it asks what involves variation, C is relevant. Given the options, A and D are direct enablers.

D. Conditional GAN: By conditioning the GAN on external information (e.g., a class label), you explicitly control and vary the type of sample generated, leading to planned variations.

Selected Answer(s): A. Input in the generator component of a GAN and D. Conditional GAN.

**GPT**
A: The random noise vector (z) fed into the generator is the source of sample variation.

D: Conditional GANs (cGANs) control variation by conditioning on labels or attributes (e.g., class, text).

C: Mode collapse reduces variation — it’s a problem related to variation (lack thereof).

B: Adjusting weights/biases affects learning but not the source of variation directly.

So the best set: A, C, D

### (c)

**DeepSeek**

* A 错，因为训练时是交替训练生成器和判别器，不是只训练生成器。
* B 对，训练时是交替训练（either … or …）。
* C 错，目标不是让判别器准确率固定在 0.5，而是希望生成样本分布与真实分布不可分时，判别器自然趋近 0.5，但训练过程不是直接为此优化。
* D 对，训练生成器时希望判别器对生成样本输出 1（即骗过判别器），训练判别器时希望真实样本标 1、生成样本标 0。

因此答案是  **B 和 D** 。

**Gemini&GPT**

The GAN training is an iterative two-step process:

Train the Discriminator ($D$):

Goal: Maximize $\log D(x)$ for real samples $x$ (score close to 1).

Goal: Maximize $\log(1 - D(G(z)))$ for fake samples $G(z)$ (score close to 0).

Train the Generator ($G$):

Goal: Minimize $\log(1 - D(G(z)))$, which is equivalent to maximizing $\log D(G(z))$ for the generated samples (score close to 1, to fool the discriminator).

When a sample is generated by $G(z)$:

The discriminator is trained to score it 0 (or low) because it's fake.

The generator is trained to get the discriminator to score it 1 (or high) because its goal is to fool the discriminator.

Selected Answer(s): D. The generator is trained for score 1 and the discriminator for score 0 (when processing a generated/fake sample).

---

## Q2 2024 May ABC

![Q2](./2024MayABC/8.png)

### Answer

### (a)

A GAN consists of two competing neural networks: the Generator ($G$) and the Discriminator ($D$). They are trained simultaneously in an adversarial process.

![Q2](./2024MayABC/10.png)
![Q2](./2024MayABC/9.png)

### (b)

Mode collapse – the generator produces limited or identical outputs (poor diversity).(PPT)

Non-convergence / instability – adversarial training may oscillate instead of converging. (PPT)

Vanishing gradients – if the discriminator becomes too strong, the generator receives almost no gradient signal.

Sensitive hyperparameters – training depends heavily on learning rates, architectures, and batch normalization.

Evaluation difficulty – it’s hard to quantify sample quality objectively.

Distorted figures (PPT 不算训练中的问题？)

### (c)

The standard GAN uses Binary Cross-Entropy (BCE) loss. When the Discriminator ($D$) outputs a score of $\mathbf{0.2}$ on a generated instance $G(z)$, the training is conducted in two steps:

1. Training the Discriminator ($D$)  The Discriminator's goal is to correctly classify the generated sample as Fake (target label $y=0$).
   Discriminator Loss on Fake Sample:

   $$
   L_D(\text{fake}) = -\log(1 - D(G(z)))
   $$

   Loss Value:

   $$
   L_D(\text{fake}) = -\log(1 - 0.2) = -\log(0.8) \approx \mathbf{0.223}
   $$

   Training Action: The discriminator's weights are adjusted to reduce this loss, which pushes its output $D(G(z))$ closer to 0.
2. Training the Generator ($G$)  The Generator's goal is to make the Discriminator classify the generated sample as Real (target label $y=1$) to fool it.
   Generator Loss (Non-saturating):

   $$
   L_G = -\log(D(G(z)))
   $$

   Loss Value:

   $$
   L_G = -\log(0.2) \approx \mathbf{1.609}
   $$

   Training Action: The generator's weights are adjusted to reduce this loss, which pushes the Discriminator's output $D(G(z))$ closer to 1.

Summary: The Discriminator gets a relatively small gradient and updates to be slightly better at classifying the sample as fake. The Generator gets a large gradient and updates aggressively to make the Discriminator score the sample higher.

### (d)

A Conditional Generative Adversarial Network (cGAN) using Convolutional Neural Networks (CNNs).

Reason:

Input: grayscale (black-and-white) image.

Condition: this grayscale image acts as the conditional input to guide generation.

Generator (CNN-based, often U-Net) learns to output corresponding color channels.

Discriminator checks whether the colored image is realistic given the grayscale input.
→ CNNs handle spatial features effectively, and conditioning ensures consistent colorization.

| Concept           | What It Is                                                                                                                                                                                       | Level                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------- |
| **U-Net**   | A**neural network architecture** — an *encoder–decoder* with skip connections, originally designed for biomedical image segmentation (Ronneberger et al., 2015).                       | **Architecture (building block)**       |
| **Pix2Pix** | A**complete framework / model** — a *conditional GAN* (cGAN) for image-to-image translation that **uses a U-Net as its generator** and a **PatchGAN as its discriminator**. | **Model framework (uses U-Net inside)** |

![Q2](./2024MayABC/11.png)
![Q2](./2024MayABC/12.png)

### (e)

图灵测试：判断者与一个人和一个机器通过文本交互，如果判断者无法区分机器和人，则机器通过测试。

GAN 中的类比：

判别器 ↔ 判断者

生成器 ↔ 机器

真实数据 ↔ 人的回答

生成数据 ↔ 机器的回答

目标都是让机器（生成器）产生与真人（真实数据）无法区分的结果，从而骗过判断者（判别器）。

The Turing Test evaluates whether a machine can generate outputs indistinguishable from a human’s.

Similarly, in a GAN, the discriminator acts like the human judge, and the generator acts like the machine trying to fool it.

The goal: the generator produces data so realistic that the discriminator can’t tell it apart from real data (ideally 50/50 confusion).
→ Thus, GAN training is an automated version of the Turing test in the data domain.

## Q3 2022 May 2

![Q3](./2022May2/4.png)
![Q3](./2022May2/5.png)
![Q3](./2022May2/6.png)

### Answer

### (a) Discriminator evaluates real data and gives answer 0.4.

The Discriminator ($D$) is trained to classify real data ($\mathbf{x}$) with a target label of 1 (or high probability). The standard loss function is Binary Cross-Entropy (BCE).

Which part(s) of GAN can be trained?Only the Discriminator ($D$) is trained. The Generator ($G$) is not involved in processing real data.

What is the loss value?The Discriminator's loss on real data is $L_D(\text{real}) = -\log(D(\mathbf{x}))$.With $D(\mathbf{x}) = 0.4$:

$$
L_D(\text{real}) = -\log(0.4) \approx \mathbf{0.916}
$$

Since the score of 0.4 is far from the target of 1, this loss is relatively high, and the Discriminator's weights will be adjusted to increase its score for real data towards 1.

### (b) Discriminator evaluates generated data and gives answer 0.7.

When processing generated data $G(\mathbf{z})$, both the Discriminator and the Generator are involved, but they have opposite objectives.

1. Training the Discriminator ($D$)Target Label:

The Discriminator aims to classify the generated sample as Fake (target label 0).

Which part(s) of GAN can be trained?The Discriminator ($D$) is trained.
Loss Value:$L_D(\text{fake}) = -\log(1 - D(G(\mathbf{z})))$.

With $D(G(\mathbf{z})) = 0.7$:

$$
L_D(\text{fake}) = -\log(1 - 0.7) = -\log(0.3) \approx \mathbf{1.204}
$$

This is a high loss because $0.7$ is closer to $1$ than to the target of $0$. The Discriminator's weights will be adjusted to decrease its score for the generated data towards 0.

2. Training the Generator ($G$)

Target Label: The Generator aims to fool the Discriminator, making it output 1 (Real). The common non-saturating loss is used.Which part(s) of GAN can be trained?The Generator ($G$) is trained.Loss Value:$L_G = -\log(D(G(\mathbf{z})))$.

With $D(G(\mathbf{z})) = 0.7$:

$$
L_G = -\log(0.7) \approx \mathbf{0.357}
$$

This is a relatively low loss for the Generator, as the score of $0.7$ is already quite close to its target of $1$. The Generator's weights will be adjusted to make the output score slightly higher than 0.7.

### (c) Can both parts of the network be trained simultaneously by backpropagation? Justify your answer.

No, both parts of the network (Generator $G$ and Discriminator $D$) are typically not trained simultaneously in a single backpropagation pass, especially in the standard GAN formulation.

Justification:

Adversarial Objectives: $G$ and $D$ have conflicting objectives. $D$ tries to maximize $\log D(\mathbf{x}) + \log(1 - D(G(\mathbf{z})))$, while $G$ tries to minimize $\log(1 - D(G(\mathbf{z})))$ (or maximize $\log D(G(\mathbf{z}))$). Training them simultaneously on a single combined loss would lead to an unstable gradient with undefined behavior.

Separate Forward Passes: The typical training procedure is done in alternating steps:Step 1: Run a forward pass and backpropagate the loss to update only $D$'s weights, keeping $G$'s weights fixed.Step 2: Run a forward pass and backpropagate the loss to update only $G$'s weights, keeping $D$'s weights fixed.

Need for Frozen Network: When training $G$, the goal is to make $D$ score the fake data high. If $D$'s weights were also changing during this step, the target would be constantly moving, making the optimization impossible. Therefore, for the $G$ update, the $D$ network is frozen.

### (d) L1 Loss calculation

What is the formula for L1?

The L1 loss (also known as the Mean Absolute Error (MAE)) measures the average magnitude of the errors between the ground truth and the generated result. For two matrices (or vectors) $\mathbf{T}$ (ground truth) and $\mathbf{G}$ (generated result), the L1 loss is:

$$
L_1(\mathbf{T}, \mathbf{G}) = \frac{1}{N} \sum_{i=1}^{N} |\mathbf{T}_i - \mathbf{G}_i|
$$

Where $N$ is the total number of elements, and $\mathbf{T}_i$ and $\mathbf{G}_i$ are the $i$-th elements of the ground truth and generated result, respectively.

What is the L1 loss for the above two matrices?

The ground truth $\mathbf{T}_1$ and generated result $\mathbf{G}_1$ are:

$$
\mathbf{T}_1 = \begin{bmatrix} 0.5 & 0.6 & 0.8 \\ 0.2 & 0.4 & 0.3 \\ 0.1 & 0.1 & 0.1 \end{bmatrix}, \quad \mathbf{G}_1 = \begin{bmatrix} 0.4 & 0.7 & 0.7 \\ 0.4 & 0.5 & 0.4 \\ 0.3 & 0.2 & 0.1 \end{bmatrix}
$$

The total number of elements is $N = 3 \times 3 = 9$.Calculate the absolute difference for each element:

$$
\begin{bmatrix} |0.5 - 0.4| & |0.6 - 0.7| & |0.8 - 0.7| \\ |0.2 - 0.4| & |0.4 - 0.5| & |0.3 - 0.4| \\ |0.1 - 0.3| & |0.1 - 0.2| & |0.1 - 0.1| \end{bmatrix} = \begin{bmatrix} 0.1 & 0.1 & 0.1 \\ 0.2 & 0.1 & 0.1 \\ 0.2 & 0.1 & 0.0 \end{bmatrix}
$$

Calculate the sum of all absolute differences:

$$
\text{Sum} = 0.1 + 0.1 + 0.1 + 0.2 + 0.1 + 0.1 + 0.2 + 0.1 + 0.0 = 1.0
$$

The L1 loss is the average of the absolute differences:

$$
L_1 = \frac{\text{Sum}}{N} = \frac{1.0}{9} \approx \mathbf{0.111}
$$

