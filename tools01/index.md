# Transformers Domain Adaptation




## Transformers Domain Adaptation

本指南说明了端到端`Domain Adaptation`工作流程，其中我们为生物医学NLP应用程序适应领域转换模型。

它展示了我们在研究中研究的两种领域自适应技术:
1. 数据选择 (Data Selection)
2. 词汇量增加 (Vocabulary Augmentation)

接下来，将演示这样一个`Domain Adaptation`的Transformer模型是如何与🤗`transformer`的培训界面兼容的，以及它如何优于开箱即用的(无领域适应的)模型。这些技术应用于BERT-small，但是代码库被编写成可推广到[HuggingFace](https://huggingface.co/)支持的其他Transformer类。

#### 警告

对于本指南，由于内存和时间的限制，我们使用域内语料库的一个小得多的子集(<0.05%)。



### 准备工作

#### 安装依赖程序

使用`pip`安装`transformers-domain-adaptation`

```shell
pip install transformers-domain-adaptation -i https://pypi.tuna.tsinghua.edu.cn/simple
```



#### 下载demo files

```shell
wget http://georgian-toolkit.s3.amazonaws.com/transformers-domain-adaptation/colab/files.zip

unzip ./files.zip
```



#### 常量

我们首先定义一些常量，包括适当的模型卡和文本语料库的相关路径。

在`domain adaptation`的背景下，有两种类型的语料库。

1. 微调语料库(Fine-Tuning Corpus)
>给定一个NLP任务（如文本分类、摘要等），这个数据集的文本部分就是微调语料库。

2. 在域语料库 (In-Domain Corpus)
>这是一个无监督的文本数据集，用于领域预训练。文本领域与微调语料库的领域相同，甚至更广泛。



```python
# 预训练模型名称
model_card = 'bert-base-uncased'

# Domain-pre-training corpora 领域预训练语料
dpt_corpus_train = './data/pubmed_subset_train.txt'
dpt_corpus_train_data_selected = './data/pubmed_subset_train_data_selected.txt'
dpt_corpus_val = './data/pubmed_subset_val.txt'

# Fine-tuning corpora
# If there are multiple downstream NLP tasks/corpora, you can concatenate those files together
ft_corpus_train = './data/BC2GM_train.txt'
```



#### 加载模型和tokenizer

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained(model_card)
tokenizer = AutoTokenizer.from_pretrained(model_card)
```





#### 数据选择

在领域预训练中，并不是所有的领域内语料库的数据都可能是有帮助的或相关的。对于不相关的文件，在最好的情况下，它不会降低领域适应模型的性能；在最坏的情况下，模型会倒退并失去宝贵的预训练信息即`灾难性的遗忘`。

因此，我们使用[Ruder & Plank](https://aclanthology.org/D17-1038.pdf)设计的各种相似性和多样性指标，从域内语料库中选择可能与下游微调数据集相关的文档。



```python
from pathlib import Path

from transformers_domain_adaptation import DataSelector


selector = DataSelector(
    keep=0.5,  # TODO Replace with `keep`
    tokenizer=tokenizer,
    similarity_metrics=['euclidean'],
    diversity_metrics=[
        "type_token_ratio",
        "entropy",
    ],
)
```



```python
#  将文本数据加载到内存中
fine_tuning_texts = Path(ft_corpus_train).read_text().splitlines()
training_texts = Path(dpt_corpus_train).read_text().splitlines()


#  在微调语料库进行fit
selector.fit(fine_tuning_texts)

# 从域内训练语料库中选择相关文件
selected_corpus = selector.transform(training_texts)

# 在`dpt_corpus_train_data_selected`下将选定的语料库保存到磁盘 
Path(dpt_corpus_train_data_selected).write_text('\n'.join(selected_corpus));
```

由于我们在`DataSelector`中指定了`keep=0.5`，所以选择的`语料库应该是域内语料库的一半大小`，包含前50%最相关的文档。



```python
print(len(training_texts), len(selected_corpus))
# (10000, 5000)

print(selected_corpus[0])
```

```shell
Chlorophyll content,leaf mass to per area,net photosynthetic rate and bioactive ingredients of Asarum heterotropoides var. mandshuricum,a skiophyte grown in four levels of solar irradiance were measured and analyzed in order to investigate the response of photosynthetic capability to light irradiance and other environmental factors. It suggested that the leaf mass to per area of plant was greatest value of four kinds of light irradiance and decreasing intensity of solar irradiance resulted in the decrease of leaf mass to per area at every phenological stage. At expanding leaf stage,the rate of Chla and Chlb was 3. 11 when A. heterotropoides var. mandshuricum grew in full light irradiance which is similar to the rate of heliophytes,however,the rate of Chla and Chlb was below to 3. 0 when they grew in shading environment. The content of Chla,Chlb and Chl( a+b) was the greatest value of four kinds of light irradiance and decreasing intensity of solar irradiance resulted in its decreasing remarkably( P<0. 05). The rate of Chla and Chlb decreased but the content of Chla,Chlb and Chl( a+b) increased gradually with continued shading. The maximum value of photosynthetically active radiation appeared at 10: 00-12: 00 am in a day. The maximum value of net photosynthetic rate appeared at 8: 30-9: 00 am and the minimum value appeared at 14: 00-14: 30 pm at each phenological stage if plants grew in full sunlight. However,when plants grew in shading,the maximum value of net photosynthetic rate appeared at about 10: 30 am and the minimum value appeared at 12: 20-12: 50 pm at each phenological stage. At expanding leaf stage and flowering stage,the average of net photosynthetic rate of leaves in full sunlight was remarkably higher than those in shading and it decreased greatly with decreasing of irradiance gradually( P < 0. 05). However,at fruiting stage,the average of net photosynthetic rate of leaves in full sunlight was lower than those in 50% and 28% full sunlight but higher than those in 12% full sunlight. All photosynthetic diurnal variation parameters of plants measured in four kinds of different irradiance at three stages were used in correlation analysis. The results suggested that no significant correlation was observed between net photosynthetic rate and photosynthetically active radiation,and significant negative correlation was observed between net photosynthetic rate and environmental temperature as well as vapor pressure deficit expect for 12% full sunlight. Positive correlation was observed between net photosynthestic rate and relative humidity expect for 12% full sunlight. Significant positive correlation was observed between net photosynthetic rate and stomatal conductance in the four light treatments. Only,in 12% full sunlight,the net photosynthetic rate was significantly related to photosynthetically active radiation rather than related to environmental temperature,vapor pressure deficit and relative humidity. In each light treatment,a significant positive correlation was observed between environmental temperature and vapor pressure deficit,relative humidity as well as stomatal conductance. Volatile oil content was 1. 46%,2. 16%,1. 56%,1. 30% respectively. ethanol extracts was 23. 44%,22. 45%,22. 18%,21. 12% respectively. Asarinin content was 0. 281%,0. 291%,0. 279% and 0. 252% respectively. The characteristic components of Asarum volatile oil of plant in different light treatments did not change significantly among different groups.
```

#### 词汇扩充

我们可以扩展模型的现有词汇用以包括特定领域的术语。这样就可以在领域预训练中明确学习这种术语的表示。

```python
from transformers_domain_adaptation import VocabAugmentor

target_vocab_size = 31_000  # len(tokenizer) == 30_522

augmentor = VocabAugmentor(
    tokenizer=tokenizer, 
    cased=False, 
    target_vocab_size=target_vocab_size
)

# 在微调语料库的基础上获得新的特定领域术语
new_tokens = augmentor.get_new_tokens(ft_corpus_train)

print(new_tokens[:20])
# ['cdna', 'transcriptional', 'tyrosine', 'phosphorylation', 'kda', 'homology', 'enhancer', 'assays', 'exon', 'nucleotide', 'genomic', 'encodes', 'deletion', 'polymerase', 'nf', 'cloned', 'recombinant', 'putative', 'transcripts', 'homologous']
```



#### 用新的词汇术语更新模型和tokenizer

```python
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))
# Embedding(31000, 768)
```



### Domain Pre-Training

`Domain PreTraining`是`Domain Adaptation`的第三步，我们在领域内语料库上用同样的预训练程序继续训练`Transformer模型`。



#### 创建数据集

```python
import itertools as it
from pathlib import Path
from typing import Sequence, Union, Generator

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

datasets = load_dataset(
    'text', 
    data_files={
        "train": dpt_corpus_train_data_selected, 
        "val": dpt_corpus_val
    }
)

tokenized_datasets = datasets.map(
    lambda examples: tokenizer(examples['text'], truncation=True, max_length=model.config.max_position_embeddings), 
    batched=True
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
```

#### 实例化TrainingArguments和Trainer

```python
training_args = TrainingArguments(
    output_dir="./results/domain_pre_training",
    overwrite_output_dir=True,
    max_steps=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    logging_steps=50,
    seed=42,
    # fp16=True,
    dataloader_num_workers=2,
    disable_tqdm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['val'],
    data_collator=data_collator,
    tokenizer=tokenizer,  # 这个标记器有新的tokens
)

# 进行训练
trainer.train()
```

`训练结果`

| Step | Training Loss | Validation Loss | Runtime   | Samples Per Second |
| ---- | ------------- | --------------- | --------- | ------------------ |
| 50   | 2.813800      | 2.409768        | 75.058500 | 13.323000          |
| 100  | 2.520700      | 2.342451        | 74.257200 | 13.467000          |



### 为特定任务进行微调

我们可以为`HuggingFace`支持的任何微调任务插入我们的`domain adaptation`模型。在本指南中，我们将在BC2GM数据集（一个流行的生物医学基准数据集）上比较一个开箱即用（OOB）模型与一个领域适应模型在命名实体识别方面的表现。用于NER预处理和评估的实用函数改编自HuggingFace的[NER微调示例笔记](https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb)。

#### 对原始数据集进行预处理，形成NER数据集

```python
from typing import NamedTuple
from functools import partial
from typing_extensions import Literal

import numpy as np
from datasets import Dataset, load_dataset, load_metric


class Example(NamedTuple):
    token: str
    label: str
        
def load_ner_dataset(mode: Literal['train', 'val', 'test']):
    file = f"data/BC2GM_{mode}.tsv"
    examples = []
    with open(file) as f:
        token = []
        label = []
        for line in f:
            if line.strip() == "":
                examples.append(Example(token=token, label=label))
                token = []
                label = []
                continue
            t, l = line.strip().split("\t")
            token.append(t)
            label.append(l)
            
    res = list(zip(*[(ex.token, ex.label) for ex in examples]))
    d = {'token': res[0], 'labels': res[1]}
    return Dataset.from_dict(d)


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["token"], truncation=True, is_split_into_words=True)
    label_to_id = dict(map(reversed, enumerate(label_list)))

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # 特殊标记有一个单词ID，是None。我们将标签设置为-100，因此它们在损失函数中被自动忽略了。
            if word_idx is None:
                label_ids.append(-100)
            # 我们为每个词的第一个标记设置标签。
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # 对于一个词中的其他标记，我们根据label_all_tokens的标志，将标签设置为当前标签或-100。
            else:
                label_ids.append(label_to_id[label[word_idx]])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 移除被忽略的索引（特殊标记）。
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
```



##### 安装 seqeval

```shell
pip install seqeval
```

[seqeval](https://pypi.org/project/seqeval/)是一个用于序列标记评估的Python框架，可以评估分块任务的性能，如命名实体识别、部分语音标记、语义角色标记等。



```python
label_list = ["O", "B", "I"]
metric = load_metric('seqeval')

train_dataset = load_ner_dataset('train')
val_dataset = load_ner_dataset('val')
test_dataset = load_ner_dataset('test')

print(train_dataset[0:1])
print(val_dataset[0:1])
print(test_dataset[0:1])
# {'token': [['Immunohistochemical', 'staining', 'was', 'positive', 'for', 'S', '-', '100', 'in', 'all', '9', 'cases', 'stained', ',', 'positive', 'for', 'HMB', '-', '45', 'in', '9', '(', '90', '%', ')', 'of', '10', ',', 'and', 'negative', 'for', 'cytokeratin', 'in', 'all', '9', 'cases', 'in', 'which', 'myxoid', 'melanoma', 'remained', 'in', 'the', 'block', 'after', 'previous', 'sections', '.']], 'labels': [['O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]}
# {'token': [['Joys', 'and', 'F', '.']], 'labels': [['O', 'O', 'O', 'O']]}
# {'token': [['Physical', 'mapping', '220', 'kb', 'centromeric', 'of', 'the', 'human', 'MHC', 'and', 'DNA', 'sequence', 'analysis', 'of', 'the', '43', '-', 'kb', 'segment', 'including', 'the', 'RING1', ',', 'HKE6', ',', 'and', 'HKE4', 'genes', '.']], 'labels': [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'B', 'O', 'O', 'B', 'I', 'O']]}
```



#### 实例化NER模型

在此，我们实例化了三个特定任务的NER模型进行比较:

1. `da_model`: 我们在本指南中刚刚训练的一个`Domain Adaptation`的NER模型

2. `da_full_corpus_model`: 同样的领域适应性NER模型，只是它是在完整的领域内训练语料库上训练的。

3. `oob_model`: 一个开箱即用的BERT-NER模型（没有经过Domain Adaptation）。

```python
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

best_checkpoint = './results/domain_pre_training/checkpoint-100'
da_model = AutoModelForTokenClassification.from_pretrained(best_checkpoint, num_labels=len(label_list))

da_full_corpus_model = AutoModelForTokenClassification.from_pretrained('./domain-adapted-bert', num_labels=len(label_list))
full_corpus_tokenizer = AutoTokenizer.from_pretrained('./domain-adapted-bert')

oob_tokenizer = AutoTokenizer.from_pretrained(model_card)
oob_model = AutoModelForTokenClassification.from_pretrained(model_card, num_labels=len(label_list))
```



#### 为每个模型创建数据集、TrainingArguments和Trainer

```python
from typing import Dict

from datasets import Dataset


def preprocess_datasets(tokenizer, **datasets) -> Dict[str, Dataset]:
    tokenize_ner = partial(tokenize_and_align_labels, tokenizer=tokenizer)
    return {k: ds.map(tokenize_ner, batched=True) for k, ds in datasets.items()}

######################
##### `da_model` #####
######################
da_datasets = preprocess_datasets(
    tokenizer, 
    train=train_dataset, 
    val=val_dataset, 
    test=test_dataset
)

print(da_datasets)

training_args = TrainingArguments(
    output_dir="./results/domain_adapted_fine_tuning",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    logging_steps=100,
    seed=42,
    fp16=True,
    dataloader_num_workers=2,
    disable_tqdm=False
)

da_trainer = Trainer(
    model=da_model,
    args=training_args,
    train_dataset=da_datasets['train'],
    eval_dataset=da_datasets['val'],
    data_collator=DataCollatorForTokenClassification(tokenizer),
    tokenizer=tokenizer,  # This tokenizer has new tokens
    compute_metrics=compute_metrics
)


##################################
##### `da_model_full_corpus` #####
##################################
da_full_corpus_datasets = preprocess_datasets(
    full_corpus_tokenizer, 
    train=train_dataset, 
    val=val_dataset, 
    test=test_dataset
)

training_args = TrainingArguments(
    output_dir="./results/domain_adapted_full_corpus_fine_tuning",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    logging_steps=100,
    seed=42,
    fp16=True,
    dataloader_num_workers=2,
    disable_tqdm=False
)

da_full_corpus_trainer = Trainer(
    model=da_full_corpus_model,
    args=training_args,
    train_dataset=da_full_corpus_datasets['train'],
    eval_dataset=da_full_corpus_datasets['val'],
    data_collator=DataCollatorForTokenClassification(full_corpus_tokenizer),
    tokenizer=full_corpus_tokenizer,  # This tokenizer has new tokens
    compute_metrics=compute_metrics
)


#######################
##### `oob_model` #####
#######################
oob_datasets = preprocess_datasets(
    oob_tokenizer, 
    train=train_dataset, 
    val=val_dataset, 
    test=test_dataset
)

training_args = TrainingArguments(
    output_dir="./results/out_of_the_box_fine_tuning",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    logging_steps=100,
    seed=42,
    fp16=True,
    dataloader_num_workers=2,
    disable_tqdm=False
)

oob_model_trainer = Trainer(
    model=oob_model,
    args=training_args,
    train_dataset=oob_datasets['train'],
    eval_dataset=oob_datasets['val'],
    data_collator=DataCollatorForTokenClassification(oob_tokenizer),
    tokenizer=oob_tokenizer,  # 这是原始的tokenizer（没有特定领域的token）。
    compute_metrics=compute_metrics
)
```



#### 训练和评估`da_model`

```python
da_trainer.train()
da_trainer.evaluate(da_datasets['test'])
```



训练结果

| Step | Training Loss | Validation Loss | Precision | Recall   | F1       | Accuracy | Runtime   | Samples Per Second |
| ---- | ------------- | --------------- | --------- | -------- | -------- | -------- | --------- | ------------------ |
| 100  | 0.258000      | 0.156976        | 0.621130  | 0.604061 | 0.612477 | 0.941589 | 64.652200 | 38.962000          |
| 200  | 0.148500      | 0.130968        | 0.689895  | 0.693156 | 0.691522 | 0.951615 | 64.609800 | 38.988000          |
| 300  | 0.131800      | 0.119317        | 0.671880  | 0.774549 | 0.719571 | 0.954099 | 64.629700 | 38.976000          |
| 400  | 0.116800      | 0.108599        | 0.738141  | 0.743567 | 0.740844 | 0.959777 | 64.621000 | 38.981000          |
| 500  | 0.078600      | 0.106925        | 0.749023  | 0.771574 | 0.760131 | 0.962172 | 64.869500 | 38.832000          |
| 600  | 0.074800      | 0.098790        | 0.749081  | 0.784351 | 0.766310 | 0.962727 | 64.517500 | 39.044000          |
| 700  | 0.073000      | 0.099364        | 0.763633  | 0.784351 | 0.773854 | 0.964268 | 64.496200 | 39.057000          |



#### 训练和评估`da_model_full_corpus`

```python
da_full_corpus_trainer.train()
da_full_corpus_trainer.evaluate(da_full_corpus_datasets['test'])
```

`结果`

| Step | Training Loss | Validation Loss | Precision |   Recall |       F1 | Accuracy |  Runtime | Samples Per Second |
| ---: | ------------: | --------------: | --------: | -------: | -------: | -------: | -------: | -----------------: |
|  100 |      0.231900 |        0.127792 |  0.671435 | 0.829809 | 0.742268 | 0.952202 | 8.246900 |         305.450000 |
|  200 |      0.108300 |        0.086280 |  0.817341 | 0.826690 | 0.821989 | 0.968876 | 8.064200 |         312.366000 |
|  300 |      0.089600 |        0.083020 |  0.807372 | 0.838995 | 0.822879 | 0.969839 | 8.014300 |         314.313000 |
|  400 |      0.080600 |        0.078229 |  0.801577 | 0.880763 | 0.839306 | 0.971885 | 8.141400 |         309.405000 |
|  500 |      0.050800 |        0.075855 |  0.843227 | 0.864125 | 0.853548 | 0.973716 | 8.172500 |         308.230000 |
|  600 |      0.052500 |        0.075362 |  0.845051 | 0.858232 | 0.851591 | 0.973550 | 8.057900 |         312.611000 |
|  700 |      0.047400 |        0.073649 |  0.851391 | 0.864818 | 0.858052 | 0.974442 | 8.029400 |                    |



#### 训练和评估`oob_model`



```python
oob_model_trainer.train()
oob_model_trainer.evaluate(oob_datasets['test'])
```

`结果`

| Step | Training Loss | Validation Loss | Precision |   Recall |       F1 | Accuracy |  Runtime | Samples Per Second |
| ---: | ------------: | --------------: | --------: | -------: | -------: | -------: | -------: | -----------------: |
|  100 |      0.229200 |        0.133785 |  0.678159 | 0.803118 | 0.735368 | 0.947964 | 8.654700 |         291.056000 |
|  200 |      0.135200 |        0.109798 |  0.745311 | 0.825984 | 0.783576 | 0.957941 | 8.660700 |         290.855000 |
|  300 |      0.117200 |        0.099117 |  0.782186 | 0.837120 | 0.808721 | 0.962326 | 8.699700 |         289.550000 |
|  400 |      0.101300 |        0.095984 |  0.827210 | 0.822420 | 0.824808 | 0.965538 | 8.725000 |         288.710000 |
|  500 |      0.069000 |        0.103978 |  0.788701 | 0.845731 | 0.816221 | 0.961440 | 8.690600 |         289.853000 |
|  600 |      0.064100 |        0.092247 |  0.827396 | 0.848404 | 0.837768 | 0.967232 | 8.671200 |         290.501000 |
|  700 |      0.064400 |        0.090411 |  0.829128 | 0.853749 | 0.841258 | 0.968306 | 8.821600 |         285.549000 |



### 结果

我们看到，在这三个模型中，`da_full_corpus_model`（在整个域内训练语料库上进行了域调整）在测试F1得分上比`oob_model`高出`2%`以上。事实上，这个`da_full_corpus_model`模型是我们训练的在BC2GM上优于SOTA的许多领域适应模型之一。

此外，`da_model`的表现也低于`oob_model`。这是可以预期的，因为`da_model`在本指南中经历了最小的领域预训练。



## 总结

在本指南中，你已经看到了如何使用 "DataSelector "和 "VocabAugmentor"，通过分别执行数据选择和词汇扩展，对变压器模型进行领域调整。

你还看到它们与HuggingFace的所有产品兼容。变换器"、"标记器 "和 "数据集"。

最后表明，在完整的领域内语料库上进行领域适应的模型比开箱即用的模型表现更好。



## 参考

[Transformers-Domain-Adaptation](https://github.com/georgian-io/Transformers-Domain-Adaptation)

[Guide to Transformers Domain Adaptation](https://colab.research.google.com/drive/1RAigUDEPpwdfgbzDII0C6-nmtoqgKABA?usp=sharing)


