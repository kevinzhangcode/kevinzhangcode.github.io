# ACl-2021-Crowdsourcing Learning as Domain Adaptation: A Case Study on Named Entity Recognition


ACl-2021-Crowdsourcing Learning as Domain Adaptation: A Case Study on Named Entity Recognition 论文解读。

<!--more-->

### 题目

Crowdsourcing Learning as Domain Adaptation: A Case Study on Named Entity Recognition [[ACL 2021 Long](https://arxiv.org/pdf/2105.14980v1.pdf)] [[**Code**](https://github.com/izhx/CLasDA)]



### 摘要

众包被认为是有效监督学习的一个前瞻性解决方案，旨在通过群体劳动建立大规模的注释训练数据。以前的研究集中在减少众包注解的噪音对监督模式的影响。在这项工作中，我们采取了不同的观点，将所有众包注释重新视为与个别数据标注师有关的黄金标准。通过这种方式，我们发现众包可以与领域适应性（domain adaptation）高度相似，那么最近的跨领域方法的进展几乎可以直接应用于众包。在这里，我们以命名实体识别（NER）为研究案例，提出了一个`Annotator-aware`的表示学习模型，该模型受到领域适应方法的启发，试图捕捉有效的`Domain-aware`的特征。我们研究了无监督和有监督的众包学习，假设没有或只有小规模的专家注释可用，在一个基准的众包NER数据集上的实验结果表明，我们的方法是非常有效的，表现了一个新的最先进的性能。此外，在有监督的情况下，我们只需要很小规模的专家注释就可以获得令人印象深刻的性能提升。

### 贡献

- 对众包学习提出了不同的看法，并建议将众包学习转化为领域适应问题（domain adaptation），这自然而然地将NLP的两个重要主题联系起来。

- 提出了一种新型的众包学习方法。尽管该方法在领域适应方面的新颖性有限，但它是第一项关于众包学习的工作，并能在NER上取得最先进的性能。

- 首次引入了有监督的众包学习，这是从领域适应性中借来的，将是NLP任务的一个前瞻性解决方案。


### 模型架构

包括四个部分：

(1) word representation

(2) annotator switcher

(3) BiLSTM Encoding

(4) CRF inference and training.

<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="https://i.bmp.ovh/imgs/2022/03/db2ee517bcc61dcb.png"
         alt="无法显示图片"
         style="zoom:90%"/>
    <br>		<!--换行-->
    Fig-1.模型结构	<!--标题-->
    </center>
</div>

#### 词表示层（word representation）

假设存在一个包含$n$个单词的句子$w_1 \dots w_n$,我们首先通过[$\texttt{Adapter} \circ \texttt{BERT}$](http://proceedings.mlr.press/v97/houlsby19a.html)将其转换为矢量表征.

$$
e_1 \dots e_n = \texttt{Adapter} \circ \texttt{BERT}(w_1 \dots w_n)
$$

注意：值得注意的是，[$\texttt{Adapter} \circ \texttt{BERT}$](http://proceedings.mlr.press/v97/houlsby19a.html)方法不再需要对庞大的BERT参数进行微调，而是通过调整轻得多的适配器参数来获得相当的性能。因此通过这种方式，可以很容易地将词的表示法扩展为`annotator-aware`的表示法。



#### 注释者切换器层（annotator switcher）

作者目标是有效地学习不同数据标注师意识到的词汇特征，这可以被视为对个别注释者的上下文理解。因此，引入了一个注释者切换器，以支持带有注释者输入的$\texttt{Adapter} \circ \texttt{BERT}$，其灵感来自于[Parameter Generation Network (PGN)](https://aclanthology.org/D18-1039/)，其关键思想是使用参数生成网络（PGN），通过输入annotators动态地产生适配器参数。通过这种方式，模型可以在不同的annotators之间灵活地切换。

具体来说，假设$V$是所有适配器参数的矢量形式，通过打包操作，也可以解包恢复所有的适配器参数，PGN模块就是根据annotators的输入动态生成[$\texttt{Adapter} \circ \texttt{BERT}$](http://proceedings.mlr.press/v97/houlsby19a.html)的$V$，如模型图中右边的橙色部分所示，切换器switcher可以被形式化为：



$$
\begin{align}
\textbf{x} &=\textbf{r}_1' \dots \textbf{n}_1'\\\\
   &=\textbf{PGN} \circ \textbf{Adapter} \circ \textbf{BERT}(x,a)\\\\
   &=\textbf{Adapter} \circ \textbf{BERT}(x,\textbf{V}=\mathbf{\Theta} \times \textbf{e}^a)
\end{align}
$$

其中$\mathbf{\Theta} \in \mathcal{R}^{\vert \textbf{V} \vert \times\textbf{e}^a}$，$\textbf{x} =\textbf{r}_1' \dots \textbf{n}_1'$是注释者$a$对$x=w_1 \dots w_n$的`annotator-aware`的表示，$\textbf{e}^a$是annotator的embedding。

#### BiLSTM编码层（BiLSTM Encoding）

[$\texttt{Adapter} \circ \texttt{BERT}$](http://proceedings.mlr.press/v97/houlsby19a.html)需要一个额外的面向任务的模块来进行高级特征提取。在这里利用单一的BiLSTM层来实现：$h_1 \dots h_n = \texttt{BiLSTM}(x)$，用于下一步的推理和训练。

#### CRF层（CRF inference and training）

最后使用`CRF`来计算候选顺序输出$y = l_1 \dots l_n$的全局得分。


$$
\begin{align}
{\textbf{o}_i} &=\textbf{W}^{crf} {\textbf{h}_i}+\textbf{b}^{crf} \\
\end{align}
$$

$$
\begin{align}
{\sum_{i=1}^n} (\textbf{T}[l_{i-1},l_i]+{\textbf{o}_i}[l_i])
\end{align}
$$



其中$\textbf{W}^{crf}、 \textbf{b}^{crf}、 \textbf{T}$是模型的参数。给定一个输入$(x，a)$，通过维特比算法进行推理,对于训练，定义了一个句子级别的交叉熵目标。
$$
\begin{align}
p(y^a \vert x,a) &=\frac{\exp{\texttt{score}(y^a \vert x,a)}}{\sum_y \exp{\texttt{score}(y \vert x,a)}}\\\\
\mathcal{L} &=-\log{(y^a \vert x,a)}
\end{align}
$$
其中$y^a$是$a$对$x$的黄金标准输出，$y$属于所有可能的候选人，$p(y^a|x, a)$表示句子级的概率。





### 结果与讨论

<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="https://i.bmp.ovh/imgs/2022/03/9098737e2db2df88.png"
         alt="无法显示图片"
         style="zoom:80%"/>
    <br>		<!--换行-->
    Fig-1.无监督学习的实验结果	<!--标题-->
    </center>
</div>

Fig-1显示了无监督情况下的测试结果。从整体上看，我们可以看到表征学习模型通过借用了`Domain Adaptation`，可以达到最好的性能，F1得分达到77.95，明显优于第二好的模型`LC-cat`。 



这一结果表明作者提出的方法比其他模型更有优势。通过深入研究结果，可以发现，annotator-aware模型明显优于annotator-agnostic模型，表明注释者信息对众包学习有很大帮助，这个观察结果进一步显示了将不同注释者类比不同领域的合理性，因为领域信息对于领域适应也是有用的。此外，作者提出的表征学习方法在annotator-aware模型中的表现更好，表明模型可以更有效地捕捉注释者感知的信息。



<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="https://i.bmp.ovh/imgs/2022/03/1a7468db109474d7.png"
         alt="无法显示图片"
         style="zoom:90%"/>
    <br>		<!--换行-->
    Fig-2.有监督学习的实验结果	<!--标题-->
    </center>
</div>



为了研究有监督情况，我们假设所有众包句子的专家注释是可用的。除了探索完整的专家注释，我们还研究了另外三种不同的情况，即在无监督环境下逐步增加专家注释，目的是研究我们的模型在小规模专家注释下的有效性。具体来说，我们`假设专家注释的比例为1%、5%、25%和100%`。



Fig-2显示了所有的结果，包括四个baselines和一个只基于专家注释的`gold`模型进行比较。总的来说，作者提出的表征学习模型可以为所有的场景带来最好的表现，证明它在监督学习中也是有效的。



接下来，通过比较annotator-agnostic模型和annotator-aware模型，我们可以看到annotator-aware模型效果更好，这与无监督的设置是一致的。



更有趣的是，结果显示在专家注释规模很小的情况下（1%和5%），`All`比`gold`好，而只有在有足够的专家注释时（25%和100%），这一趋势才会逆转。



该观察表明，当黄金注释(gold annotations)不足时，众包注释(crowdsourced annotations)总是有帮助的，同时，我们很容易理解`MV`比`gold`差，因为后者的训练语料质量更高。



此外，可以发现即使是`增加annotator-aware机制的LC和LC-cat模型`也无法获得与`gold annotations`类似的效果，这表明从众包注释中提炼更加优秀的数据标注可能不是最有希望的解决方案，而表征学习模型可以持续给出比`gold annotations`更好的结果，表明众包注释对我们的方法总是有帮助。

### 代码解读

待复现

