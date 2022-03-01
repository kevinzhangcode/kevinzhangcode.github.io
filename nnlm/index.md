# 01-NNLM(’A Neural Probabilistic Language Model‘) 


## A Neural Probabilistic Language Model

这篇论文是预训练语言模型的开山之作，[Yoshua Bengio](https://yoshuabengio.org/)等于2003年提出的方法。

### 观点

- 将词汇表$V$中的每个单词${w_i}$关联到一个分布式单词特征向量$\mathcal{R}^m$。

- 将句子的联合概率函数表示为句子序列中单词特征向量的组合。

- 同时学习单词的特征向量和句子联合概率函数的参数。

### 模型

假设存在句子$w_1,\dots，w_i,\dots,w_n$，其中$w_n \in V$，$V$表示词汇集合，$w_i$表示单词，目标函数是学习$f(w_t,\dots,w_{t-n+1})=\hat{P}(w_t \vert w_1^{t-1})$的参数。

Bengio等人将模型分成两个部分：

- 一个映射函数$C$，将 $V$中的第$i$个单词$w_i$映射成为一个 `特征向量` $C(w_i)\in \mathcal{R}^m$，它表示词汇表中与每个单词相关的分布特征向量。

- 一个使用映射函数$C$表示的概率函数$g$，通过上下文中单词的特征向量的乘积组成联合概率模型，$g$的输出是一个向量，它的第$i$个元素估计了概率。
$$
f(i,w_{t-1},\dots,w_{t-n+1})=g(i,C(w_{t-1}),\dots,C(w_{t-n+1}))
$$

函数$f$是这两个映射($C$和$g$)的组合，上下文中的所有单词都共享$C$。与这两个部分的每个部分关联一些参数。

数学符号说明：

- $C(i)$：单词$w$对应的词向量，其中$i$为词$w$在整个词汇表中的索引
- $C$：词向量，大小为$\vert V \vert \times m$的矩阵
- $\vert V \vert$：词汇表的大小，即预料库中去重后的单词个数
- $m$：词向量的维度，一般大于50
- $H$：隐藏层的 weight
- $d$：隐藏层的 bias
- $U$：输出层的 weight
- $b$：输出层的 bias
- $W$：输入层到输出层的 weight
- $h$：隐藏层神经元个数
  

<img src="https://s3.bmp.ovh/imgs/2022/03/5c4e651036fc9162.png" alt="模型结构" style="zoom:120%;" />


计算流程：
- 首先将输入的$n-1$个单词索引转为词向量，然后将这$n-1$个向量进行 concat，形成一个$(n-1)\times w$  的矩阵，用$X$表示
- 将$X$送入隐藏层进行计算，$\textit{hidden}_\text{out}=\tanh{(d + X * H)}$
- 输出层共有$\vert V \vert$个节点，每个节点$y_i$表示预测下一个单词$i$的概率， $y$的计算公式为$y=b+X*W+\textit{hidden}_\text{out} * U$ 

## 代码

```python
# code by Tae Hwan Jung @graykode, modify by wmathor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

dtype = torch.FloatTensor

sentences = [ "i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split() # ['i', 'like', 'dog', 'dog', 'i', 'love', 'coffee', 'i', 'hate', 'milk']
word_list = list(set(word_list)) # ['i', 'like', 'dog', 'love', 'coffee', 'hate', 'milk']
word_dict = {w: i for i, w in enumerate(word_list)} # {'i':0, 'like':1, 'dog':2, 'love':3, 'coffee':4, 'hate':5, 'milk':6}
number_dict = {i: w for i, w in enumerate(word_list)} # {0:'i', 1:'like', 2:'dog', 3:'love', 4:'coffee', 5:'hate', 6:'milk'}
n_class = len(word_dict) # number of Vocabulary, just like |V|, in this task n_class=7

# NNLM(Neural Network Language Model) Parameter
n_step = len(sentences[0].split())-1 # n-1 in paper, look back n_step words and predict next word. In this task n_step=2
n_hidden = 2 # h in paper
m = 2 # m in paper, word embedding dim

def make_batch(sentences):
  input_batch = []
  target_batch = []

  for sen in sentences:
    word = sen.split()
    input = [word_dict[n] for n in word[:-1]] # [0, 1], [0, 3], [0, 5]
    target = word_dict[word[-1]] # 2, 4, 6

    input_batch.append(input) # [[0, 1], [0, 3], [0, 5]]
    target_batch.append(target) # [2, 4, 6]

  return input_batch, target_batch

input_batch, target_batch = make_batch(sentences)
input_batch = torch.LongTensor(input_batch)
target_batch = torch.LongTensor(target_batch)

dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)

class NNLM(nn.Module):
  def __init__(self):
    super(NNLM, self).__init__()
    self.C = nn.Embedding(n_class, m)
    self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
    self.W = nn.Parameter(torch.randn(n_step * m, n_class).type(dtype))
    self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
    self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype))
    self.b = nn.Parameter(torch.randn(n_class).type(dtype))

  def forward(self, X):
    '''
    X: [batch_size, n_step]
    '''
    X = self.C(X) # [batch_size, n_step] => [batch_size, n_step, m]
    X = X.view(-1, n_step * m) # [batch_size, n_step * m]
    hidden_out = torch.tanh(self.d + torch.mm(X, self.H)) # [batch_size, n_hidden]
    output = self.b + torch.mm(X, self.W) + torch.mm(hidden_out, self.U) # [batch_size, n_class]
    return output

model = NNLM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(5000):
  for batch_x, batch_y in loader:
    optimizer.zero_grad()
    output = model(batch_x)

    # output : [batch_size, n_class], batch_y : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, batch_y)
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

# Predict
predict = model(input_batch).data.max(1, keepdim=True)[1]

# Test
print([sen.split()[:n_step] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])
```

## 参考

[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

[NNLM 的 PyTorch 实现](https://wmathor.com/index.php/archives/1442/)

[nlp-tutorial](https://github.com/graykode/nlp-tutorial/blob/master/1-1.NNLM/NNLM.py)

