# ACl-2019-Cross-Domain NER using Cross-Domain Language Modeling


ACl-2019-Cross-Domain NER using Cross-Domain Language Modeling 论文解读。

<!--more-->

### 题目

Cross-Domain NER using Cross-Domain Language Modeling [[ACL 2019\]](https://www.aclweb.org/anthology/P19-1236) [[Code\]](https://github.com/jiachenwestlake/Cross-Domain_NER)





### 摘要

由于标签资源的限制，跨域命名实体识别（Cross-Domain NER）一直是一项具有挑战性的任务。大多数现有的工作都是在监督下进行的，即利用源域和目标域的标记数据。这类方法的一个缺点是，它们不能对没有NER数据的domain进行训练。为了解决这个问题，我们考虑使用跨域的语言模型(LMs)作为NER领域适应的桥梁，通过设计一个新的参数生成网络进行跨域和跨任务的知识转移。结果表明，我们的方法可以有效地从跨域LMs对比中提取域的差异，允许无监督的域适应，同时也给出了最先进的结果。



### 贡献



### 模型

模型的整体结构如图Fig-1所示。底部展示了两个领域和两个任务的组合。首先给定一个输入句子，通过一个共享的嵌入层计算单词表征，然后通过一个新的参数生成网络计算出一组特定任务和领域的BiLSTM参数，用于编码输入序列，最后不同的输出层被用于不同的任务和领域。

<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="https://i.bmp.ovh/imgs/2022/03/8fc9a7474d79931e.jpg"
         alt="无法显示图片"
         style="zoom:90%"/>
    <br>		<!--换行-->
    Fig-1.Model architecture	<!--标题-->
    </center>
</div>

#### Input Layer

按照Yang等人（2018）的说法,给定一个输入$\mathbf{x}=[x_1,x_2,\cdots,x_n]$，来自以下4个数据集

- 源域NER训练集$S_{ner}=\\{\\{x_i,y_i\\}\\}_{i=1}^m$

- 目标域NER训练集$T_{ner}=\\{\\{x_i,y_i\\}\\}_{i=1}^n$
- 源域原始文本集$S_{lm}=\\{\\{x_i\\}\\}_{i=1}^p$
- 目标域原始文本集$T_{lm}=\\{\\{x_i\\}\\}_{i=1}^p$

每个词$x_i$被表示为其词嵌入和字符级CNN输出的连接：
$$
\mathbf{v}_i =[\mathbf{e}^w(x_i)\oplus \text{CNN}(\mathbf{e}^c(x_i))]
$$
其中$\mathbf{e}^w$代表一个共享的词嵌入查询表，$\mathbf{e}^c$代表一个共享的字符嵌入查询表。$\text{CNN}(\cdot)$代表一个标准的$\text{CNN}$，作用于一个词$x_i$的字符嵌入序列$\mathbf{e}^c(x_i)$，$\oplus$表示矢量连接。



#### Parameter Generation Network

将$\mathbf{v}$送入一个双向的LSTM层，为了实现跨领域和跨任务的知识转移，使用一个参数生成网络$f(\cdot,\cdot,\cdot)$动态地生成$\text{BiLSTM}$的参数，由此产生的参数被表示为$\theta_{\text{LSTM}}^{d,t}$，其中$d \in {src,tgt}$，$t\in {ner,lm}$ 分别代表领域标签和任务标签
$$
\theta_{\text{LSTM}}^{d,t} = \mathbf{W} \otimes \mathbf{I}_d^D \otimes \mathbf{I}_t^T
$$
参数解释：

- $\mathbf{v}=[{\mathbf{v}_1},{\mathbf{v}_2},\dots,{\mathbf{v}_n}]$表示输入词嵌入

- $ \mathbf{W}\in \mathbb{R}^{P^{(LSTM)}\times V \times U}$代表一组以三阶张量形式存在的元参数

- $\mathbf{I}_d^D\in \mathbb{R}^U$代表领域词嵌入

- $\mathbf{I}_d^D\in \mathbb{R}^V$代表任务词嵌入

- $U$、 $V$分别代表领域和任务词嵌入的大小

- $P^{(LSTM)}$是$\text{BiLSTM}$参数的数量

- $\otimes$ 指张量收缩

给定输入$v$和参数$\theta$，一个任务和特定领域$\text{BiLSTM}$单元的隐藏输出可以统一写成:

 <img src="https://i.bmp.ovh/imgs/2022/03/2f1c264d513dcacc.png" style="zoom:50%;" />

```tex
\begin{aligned}
\overrightarrow{\mathbf{h}}_i^{d,t}=\text{LSTM}(\overrightarrow{\mathbf{h}}_{i-1}^{d,t},\mathbf{v}_i,\overrightarrow{\theta}_{\text{LSTM}}^{d,t})\\
\overleftarrow{\mathbf{h}}_{i}^{d,t}=\text{LSTM}({\overleftarrow{\mathbf{h}}}_{i-1}^{d,t},\mathbf{v}_i,\overleftarrow{\theta}_{\text{LSTM}}^{d,t})
\end{aligned}
```

$\overrightarrow{\mathbf{h}}_i^{d,t}$，$\overleftarrow{\mathbf{h}}_i^{d,t}$分别为前向和后向。



#### Output Layers

标准CRFs被用作NER的输出层，在输入句子$\mathbf{x}$上产生的标签序列$\mathbf{y}=l_1,l_2,\dots,l_i$的输出概率$p(\mathbf{y}\vert \mathbf{x})$是



<img src="https://i.bmp.ovh/imgs/2022/03/e2126ea6072d8faa.png" style="zoom:50%;" />

```tex
$$
p(\boldsymbol{y} \mid \boldsymbol{x})=\frac{\exp \left\{\sum_{i}\left(\mathbf{w}_{\mathrm{CRF}}^{l_{i}} \cdot \mathbf{h}_{i}+b_{\mathrm{CRF}}^{\left(l_{i-1}, l_{i}\right)}\right)\right\}}{\sum_{\boldsymbol{y}^{\prime}} \exp \left\{\sum_{i}\left(\mathbf{w}_{i}^{l_{\mathrm{CRF}}^{\prime}} \cdot \mathbf{h}_{i}+b_{\mathrm{CRF}}^{\left(l_{i-1}^{\prime}, l_{i}^{\prime}\right)}\right)\right\}}
$$
```



参数解释：

- $\mathbf{h}=[\overrightarrow{\mathbf{h}}_1 \otimes \overleftarrow{\mathbf{h}}_1,\dots,\overrightarrow{\mathbf{h}}_n \otimes \overleftarrow{\mathbf{h}}_n]$代表前向和后向的组合特征

- $y'$代表一个任意的标签序列

- $\mathbf{w}^{li}_{CRF}$是$l_i$特有的模型参数

- ${b_{CRF}^{(l_{i-1},l_i)}}$ 是 $l_{i-1}$ 和 $l_i$特有的偏置

考虑到不同领域的NER标签集可能不同，在Fig-1中分别用$\text{CRF(S)}$和$\text{CRF(T)}$来表示源域和目标域的$\text{CRFs}$，使用一阶Viterbi算法来寻找高分的标签序列。



#### Language modeling

前向$\text{LM(LMf)}$ 使用前向LSTM隐藏状态$\overrightarrow{\mathbf{h}}=[\overrightarrow{\mathbf{h}}_1,\dots,\overrightarrow{\mathbf{h}}_n]$：

- 在给定$x_{1:i}$情况下来计算下一个词$x_{i+1}$的概率，表示为$p^f (x_{i+1}\vert x_{1:i})$

后向$\text{LM(LMb)}$ 使用后向LSTM隐藏状态$\overleftarrow{\mathbf{h}}=[\overleftarrow{\mathbf{h}}_1,\dots,\overleftarrow{\mathbf{h}}_n]$：

- 在给定$x_{i:n}$情况下来计算上一个词$x_{i-1}$的概率，表示为$p^f (x_{i-1}\vert x_{i:n})$

考虑到计算效率，采用负采样Softmax（NSSoftmax）来计算前向和后向概率，具体如下：

 <img src="https://i.bmp.ovh/imgs/2022/03/89ee46cafad38527.png" style="zoom:50%;" />

```tex
$$
\begin{aligned}
&p^{f}\left(x_{i+1} \mid x_{1: i}\right)=\frac{1}{Z} \exp \left\{\mathbf{w}_{\# x_{i+1}}^{\top} \overrightarrow{\mathbf{h}}_{i}+b_{\# x_{i+1}}\right\} \\
&p^{b}\left(x_{i-1} \mid x_{i: n}\right)=\frac{1}{Z} \exp \left\{\mathbf{w}_{\# x_{i-1}}^{\top} \overleftarrow{\mathbf{h}}_{i}+b_{\# x_{i-1}}\right\}
\end{aligned}
$$
```

其中

- $\\\# x$代表目标词$x$的词汇索引

- $\mathbf{w}_{\\\#x}$和$b_{\\\#x}$分别为目标词向量和目标词bias

- $Z$是归一化项目，计算公式为:

  <img src="https://i.bmp.ovh/imgs/2022/03/780747805e0cf3d3.png" style="zoom:50%;" />

  其中$\mathcal{N}_x$代表目标词$x$的nagative样本集，该集的每个元素都是1到跨域词汇量的随机数，$\bar{\mathbf{h}}i$分别代表LMf中的$\overrightarrow{\mathbf{h}}_i$和LMb中的$\overleftarrow{\mathbf{h}}_i$。

### 实验结果与讨论

待补充



### 结论

通过从原始文本中提取领域差异的知识来进行NER领域适应。为了实现这一目标，作者通过一个新的参数生成网络进行跨领域语言建模，该网络将领域和任务知识分解为两组嵌入向量。在三个数据集上的实验表明，方法在有监督的领域适应方法中是非常有效的，同时允许在无监督的领域适应中进行zero-shot学习。

### 代码

https://github.com/jiachenwestlake/Cross-Domain_NER


