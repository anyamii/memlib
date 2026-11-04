# 第五阶段虚构夹具与标定

本目录正文、组织、人物和问题全部虚构，未使用真实聊天。40 条正文分为
两个独立语料组；每组 10 道正题和 10 道负例。两个组的主题结构相似，
只用于验证程序，不能作为统计有效性或上线阈值的依据。负例一半完全无关，
一半含已知组织名但追问库中没有的身份证信息，用于覆盖有词面命中的无答案题。

## 库配置

```python
from memlib import Gate, Options, Recall
options = Options(vector=False, gate=Gate(abs_min=0, norm_min=0))
pack = Recall(store, adapters).query(query, options)
```

`gate.abs_min`、`gate.norm_min` 均默认 0，全部关闭时不添加字段、不计算门，
完整返回 JSON 与旧实现相同。至少一个阈值大于 0 时，在 keyword lane 后、
向量 lane 与取源前判断；必须启用 keyword。若同时开两个门，按 no_hit、abs、
norm 顺序决定拒绝原因。等于阈值放行。拒绝结果 `items=[]`，
`debug` 中包含 `gate_reason`、`S1`、`F`；放行时 reason 为 null。
无命中两个特征均为 0。开启门时若 n=0，候选为空，判为 no_hit。

`S1` 是 scope、墓碑、时间过滤后 keyword top1 的 `-bm25(keyword)`。
`F=S1/Σ max(1e-6, ln((N-df+0.5)/(df+0.5)))`。分母包含全部去重查询词，
包括 OOV（df=0）。使用 FTS5 unicode61 实际词项规范化后的去重结果与
fts5vocab 文档频率。N、df 与 SQLite BM25 一样来自整个 keyword 索引，
包括仍留在索引里的其他 namespace/已屏蔽记录，不能混用局部 df 和全局分数。
第四阶段每段独立建库，正好是这种口径的单库特例。F 不是概率，可以大于 1。

## 中文路径

原生 FTS5 默认 unicode61 不会对连续汉字分词；`星砂花店向日葵` 是一个
词项，查询 `向日葵` 无命中。库原本已经在写入 FTS5 前调用 tokenizer：
Store 默认 BigramTokenizer，把中文连续串拆为相邻双字（单字单独保留），
再用空格写入 FTS5。因此库实际默认路径没有整句词项问题。

本轮选择并复用已有 bigram，而不再引入第三种 tokenizer。它不需要词典或
新依赖，可确定地处理虚构专名。已有 JiebaTokenizer 仍是可选项。
`Store(path, tokenizer=BigramTokenizer())` 显式选中文，
`Store(path, tokenizer=EnglishTokenizer())` 保持原英文切词路径。
标定脚本新增 `--tokenizer bigram|english|jieba`，默认 bigram。
切换已有库的 tokenizer 需要用相同 tokenizer 重建索引，版本不匹配会报错；
本轮没有重建任何现有索引。

## 输入格式与命令

三个 UTF-8 JSON 文件，顶层均为数组：

```json
[{"group": 1, "id": "m1", "text": "完全虚构的中文正文"}]
```

上面是 corpus。positives 和 negatives 都用下列格式：

```json
[{"group": 1, "query": "这段正文支持或不支持的问题？"}]
```

group 是非负整数，代表独立检索语料组，同组 id 唯一。每个查询只查自己组。
corpus 只放正文，不放题目、答案或摘要。正题应由标注者确认库中存在答案，
负例应确认库中无答案。误拒定义为正题最终空 Pack，误出为负例最终非空 Pack，
不是答案正确率或 gold 证据召回。脚本不自动验证标注。

```sh
.venv/bin/python scripts/calibrate_gate.py \
  --corpus examples/phase5/corpus.json \
  --positives examples/phase5/positives.json \
  --negatives examples/phase5/negatives.json \
  --tokenizer bigram --target 0.07 --output /tmp/phase5-calibration.json
```

脚本只读取这三个显式文件，在独立内存索引中运行库里的门；不打开现有 DB，
不加载模型，不调用任何向量编码，NoEncoder 一旦被调用即失败。
Pack 使用 n=50、5条/1200字、近重复0.97，关闭时间权重；正文用固定历史日期。
如果实际部署使用不同的范围、语料规模或 Pack 配置，应先让标定口径与部署一致。

输出包含 abs_sweep、norm_sweep、combined_grid、所有满足训练误拒≤target的
candidates 及 selected。档位由正题特征分位数产生，另含关闭档位 0；严格小于
阈值拒绝。候选依次按负例误出、正题误拒、绝对阈值、归一化阈值升序排序。
没有满足目标的候选时 selected=null，不放宽目标。扫描结果含分子、分母和比率。

奇偶验证按 group 而非题目分半；两边都必须有正负题。每个训练半单独估计
分位数、选阈值，将原值直接用于另一半，测试半不重新调参。输出两个方向的
训练扫描与测试结果，以及 meets_target。全卷候选有同卷选择偏差，不能替代
留出验证。题目不足或基线误拒超过目标时可能没有可用候选。

使用其他语料前，应确认数据使用授权、人工标注与检索配置；合成夹具结果不应直接用于部署。
