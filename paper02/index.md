# ACl-2021-Learning Domain-Specialised Representations for Cross-Lingual Biomedical Entity Linking




ACl-2021-Learning Domain-Specialised Representations for Cross-Lingual Biomedical Entity Linking 论文解读。

<!--more-->

### 题目

Learning Domain-Specialised Representations for Cross-Lingual Biomedical Entity Linking [[**ACL 2021 Short**](https://arxiv.org/pdf/2105.14398.pdf)] [[**Code**](https://github.com/cambridgeltl/sapbert)]





### 摘要

将外部特定领域的知识（如UMLS）注入预训练的语言模型（LM）中，可以提高其处理特殊领域内任务的能力，如生物医学实体链接任务（BEL），然而，这种丰富的专家知识只适用于少数语言（如英语）。在这项工作中，作者通过提出一个新的跨语言生物医学实体连接任务(XL-BEL)，并建立一个新的`XL-BEL`基准，跨越10种不同的语言，作者首先研究了标准的知识诊断以及知识增强的单语言和多语言LMs在标准的单语言英语BEL任务之外的能力。这些评分显示了与英语表现的巨大差距。然后，作者解决了将特定领域的知识从资源丰富的语言转移到资源贫乏的语言的挑战。为此，作者为`XL-BEL`任务提出并评估了一系列跨语言的转移方法，并证明一般领域的抓字眼有助于将现有的英语知识传播给几乎没有in-domain数据的语言。值得注意的是，作者提出的特定领域的转移方法在所有目标语言中产生了一致的收益，有时甚至达到了20%@1点，而目标语言中没有任何领域内的知识，也没有任何领域内的平行数据。



### 贡献

1. 强调了学习（生物医学）专业领域的跨语言表征的挑战。

2) 提出了一个新颖的多语种`XL-BEL`任务，并有10种语言的综合评估基准。
3) 对`XL-BEL`任务中现有的知识诊断和知识增强的单语和多语LMs进行了系统性的评估。
4) 在生物医学领域提出了一个新的SotA多语言编码器，它在XL-BEL中产生了巨大的收益，特别是在资源贫乏的语言上，并提供了强大的基准测试结果来指导未来的工作。

### 模型



### Language-Agnostic SAP

让$(x,y)\in \mathcal{X}\times \mathcal{Y}$表示一个名字和其分类标签的元组。当从UMLS同义词学习时，$\mathcal{X}\times \mathcal{Y}$是所有`(name，CUI)`对的集合，例`(vaccination, C0042196)`。虽然[Liu](https://aclanthology.org/2021.naacl-main.334/)等人只使用英文名称，但作者在此考虑其他UMLS语言的名称。

在训练过程中，该模型被引导为同义词创建类似的表示，而不论其语言如何。 该学习方案包括：1）一个在线抽样程序来选择训练实例；2）一个度量学习损失，鼓励共享相同CUI的字符串获得类似的表示。



#### Training Examples

给定一个由$N$个例子组成的小批次$\mathcal{B} ={\mathcal{X_B}}\times{\mathcal{Y_B}}={{(x_i, y_i)}_i^N}=1 $，我们从为所有名字$x_i\in\mathcal{X_B}$构建所有可能的三联体开始。每个三联体的形式是$(x_a, x_p, x_n)$，其中$x_a$是锚标签，是$\mathcal{X_B}$中的一个任意名字；$x_p$是$x_a$的正匹配（即$y_a= y_p$），$x_n$是$x_a$的负匹配（即$y_a\not =y_n$）。然后我们让$f(\cdot)$表示编码器（即本文中的MBERT或XLMR），在构建的三联体中，选择所有满足以下约束条件的三联体:
$$
\Vert f(x_a) -f(x_p) \Vert_2 + \lambda \geq  \Vert f(x_a) -f(x_n) \Vert_2
$$

其中$\lambda$是一个预定义的余量。换句话说，我们只考虑正样本比负样本多出$\lambda$的三联体。这些"硬"三联体对表示学习来说信息量更大。然后，每个被选中的三联体都会贡献一个正数对$(x_a,x_p)$和一个负数对$(x_a,x_n)$，我们收集所有这样的正数和负数，并将它们表示为$\mathcal{P},\mathcal{N}$。



#### Multi-Similarity Loss

我们计算所有名字代表的成对余弦相似度，得到一个相似度矩阵$\mathcal{S}\in\mathscr{R}^{\vert{\mathcal{X_B}}\vert\times\vert{\mathcal{X_B}}\vert}$，其中每个条目$\mathcal{S_{ij}}$是小批$\mathcal{B}$中第$i$个和第$j$个名字之间的余弦相似度，然后使用多重相似度损失从三联体学习:



<img src = 'https://i.bmp.ovh/imgs/2022/03/2c6fac55664fec5d.png' style="zoom:150%;" />

```tex
\begin{align}
\mathcal{L} = \frac{1}{\vert{\mathcal{X_B}}\vert}{\displaystyle\sum_{i=1}^{\vert{\mathcal{X_B}}\vert}}(\frac{1}{\alpha}\log{(1+\sum_{n\in\mathcal{N}_i} e^{\alpha(\mathcal{S}_{in}-\epsilon)})}+\frac{1}{\beta}\log{(1+\sum_{n\in\mathcal{P}_i} e^{\alpha(\mathcal{S}_{ip}-\epsilon)})})
\end{align}
```

$\alpha，\beta$是温度标度；$\epsilon$是应用于相似性矩阵的偏移量；$\mathcal{P}_i,\mathcal{N}_i$是第$i$个锚的正负样本的指数。





### 实验结果与讨论

<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="https://i.bmp.ovh/imgs/2022/03/fc6ef9fe8d4cc976.png"
         alt="无法显示图片"
         style="zoom:130%"/>
    <br>		<!--换行-->
    Fig-1.Multilingual UMLS Knowledge Always Helps	<!--标题-->
    </center>
</div>

Fig-1总结了在各种单语、多语和领域内预训练编码器上应用基于UMLS知识的多语言SAP微调的结果；注入UMLS知识对模型在XL-BEL上的表现在所有语言和所有基础编码器上都是有益的。使用多语言UMLS同义词对生物医学$\texttt{PUBMEDBERT}$（$SapBERT_{all\_syn}$）进行SAP-fine-tune，而不是只使用英语同义词（$SapBERT$），能全面提高其性能。对每种语言的单语BERT进行SAP-ing调整，也在所有语言中产生了巨大的收益；唯一的例外是泰语（TH），它在UMLS中没有体现。对多语言模型MBERT和XLMR进行微调，会带来更大的相对收益。

UMLS数据在很大程度上偏向于罗曼语和日耳曼语。因此，对于与这些语系比较相似的语言，单语LM（上半部分，Fig-1）与多语LM（下半部分，Fig-1）相比表现相当或优于多语LM。然而，对于其他（遥远的）语言（如KO、ZH、JA、TH），情况则相反。例如，在TH上，XLMR+SAPall_syn比THBERT+SAPall_syn高出20%@1的精确度。



<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="https://i.bmp.ovh/imgs/2022/03/2b08831a4c6feee0.png"
         alt="无法显示图片"
         style="zoom:130%"/>
    <br>		<!--换行-->
    Fig-2.General Translation Knowledge is Useful	<!--标题-->
    </center>
</div>

Fig-2总结了我们在一般翻译数据上继续训练的结果。 在之前基于UMLS的SAP之后 通过这个变体，基础多语言LM成为强大的多语言生物医学专家。我们观察到域外翻译数据的额外强大提升：例如，对于MBERT，除ES外，所有语言的提升范围为2.4%至12.7%。对于XLMR，我们报告了$XLMR+SAPen_{syn}$在RU、TR、KO、TH上的精确度@1提升，以及$XLMR+SAPall_{syn}$的类似但较小的提升。

### 结论

我们引入了一个新的跨语言生物医学实体任务（XL-BEL），为生物医学领域的跨语言实体表示建立了一个覆盖面广且可靠的评估基准，并在XL-BEL上评估了当前的SotA生物医学实体表示。我们还提出了一个有效的迁移学习方案，利用一般领域的翻译来提高领域专业表示模型的跨语言能力。我们希望我们的工作能够激励未来更多关于多语言和领域专业表示学习的研究。

### 代码

https://github.com/cambridgeltl/sapbert






