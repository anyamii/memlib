# memlib

memlib is an offline memory retrieval experiment for long-running companion conversations. It compares lexical and dense retrieval, evaluates when to return an empty memory pack, and reports both gains and failures. Public P2/P3 LoCoMo retrieval code and protocols are included; private Chinese evaluations publish aggregate metrics only. This is a retrieval study, not a production system or an end-to-end QA benchmark.

给长期陪伴对话用的离线记忆检索实验：**该想起时想起，没有就别乱给**。输入当前一句话，返回有来源、有限预算的 Memory Pack，或空包。

长期对话的难点不只是搜到相关词：用户可能在说当下，也可能问一件根本没发生过的事。因此这里把“找得到”“是否需要回忆”“记忆是否存在”分开评估。项目仍在离线阶段，尚未接入真实聊天生成。

## 架构

```text
公开/合成来源 → 适配器 → 原话引用、日印象修订 → 片段
                                                   ↓
当前查询 ──────────────→ BM25 / dense 向量（两路各前50）
                                      ↓
                                     RRF
                                      ↓
                          拒答门（实验配置；意图层尚未达标）
                                      ↓
                 回源 + 墓碑/修订检查 + 去重 + 预算 → Memory Pack
```

这是概念图；当前实现里绝对分门在关键词召回后，关键词覆盖门在 Pack 形成后，各实验对门的位置与排序口径单独冻结。默认门关闭；默认字符哈希向量只用于测试，真实模型需显式配置。Pack 默认最多5条、1200 Unicode 字符、每篇最多2段。时间加权可配置，以下 P3/P6–P9 对照关闭。

原话正文不写入运行库，只经适配器在内存读取与回源；日印象及其修订允许保存正文。FTS 词项、向量和引用也可能泄露信息，均按私人数据保护。墓碑及修订检查保证被屏蔽或过期片段不交付；这不等于物理擦除所有历史。

## 评测：结果与失败一起报

**公开英文集**：LoCoMo 文本检索协议，10段独立对话、1527道有有效证据问题；排除无有效证据及 category 5，另设30条合成闲聊负例。QA、答案、摘要不进索引。检索截止时间为各段最后会话后一天，是合成全历史检索条件，不是在线对话回放。

| 公开集配置 | R@5 | 来源 |
|---|---:|---|
| BM25 | 42.62% | P2 |
| 多语 MiniLM dense | 23.76% | P2 |
| BM25 + MiniLM RRF(k=60) | 37.27% | P2 |
| bge-m3 dense | 14.33% | P3 |

英文卷上向量拖后腿。BGE 的 CLS 池化、归一化、截断等经复核，不能简单把失败归因于池化写错。公开 R@5 为每题 gold 证据覆盖比例的均值，预算前计算；不是最终回答准确率。

P4 增加1427条跨对话构造负例（1527候选中剔除100条撞名配对）。BM25 归一化分门在正题误拒 **152/1527=9.95%** 时，跨对话误出从 **1427/1427=100%** 降至 **516/1427=36.16%**；30条合成闲聊误出仍为15/30=50%。这是同卷扫描，不能当成无偏测试集成绩；断层门效果较差。

**私有中文集**：271篇/条文档，切成2190片段；120正题、100未发生事件、60闲聊，P8追加40条同义改写题，P9再追加200条新闲聊（其中30条困难闲聊）。只公开聚合结果，不发布正文、题目、答案、专有词表、逐题日志或向量。

| 实验 | 结果 | 限制/失败 |
|---|---|---|
| P6 绝对分门 | 原60闲聊误出58/60=96.67% → 6/60=10%；正题误拒8/120=6.67% | 未发生事件仍100/100误出 |
| P7 关键词覆盖联合门 | 未发生误出100% → 29/100=29%；正题误拒7/120=5.83% | 原闲聊8/60=13.33%；只取最稀有词的K1误拒至少62.5% |
| P8 无门同义召回 | bge-m3 R@5=65%，BM25=5%（40题） | 小样本、人工改写；不是端到端回答成绩 |
| P8 无门正题召回 | RRF(k=10) R@5=91.67%（120题） | 不是过门后Pack召回；不可与公开集覆盖率直接比较 |
| P8 向量作存在门 | 正题与闲聊 top1 余弦P50为0.669508/0.666949 | 相似不能证明事件存在；实验C闲聊误出58/60=96.67% |
| P9 扩展闲聊复测 | P7/P8字面基线在新200题误出57/200=28.5% | 原60题上的13.33%偏乐观，不能再写成稳定线上能力 |
| P9 意图规则+原型 I3 | 意图层拦闲聊241/260=92.69% | 漏判正题28/120=23.33%，整套未达≤7%目标 |

P9 的92.69%是**意图层**拦截率；存在门后整体闲聊误出18/260，不能混用两个阶段的分子。I3诊断档正题 Pack R@5=69.17%，未发生误出59/100，比字面基线更差，没有选上线档位。目标误拒硬上限10%，标定瞄7%；P9双向留出整套误拒25.40%/22.81%，未达标。P7/P8字面基线相应留出也有7.94%/8.77%，不能说严格守住7%。

中文 R@5 指任一 gold 片段进入前5即命中；Pack R@5在最终预算后计算，空包记未命中。误拒是正题空Pack比例，误出是相应负题非空Pack比例。无门召回、最终Pack命中与事实回答准确率是不同指标。

私卷为共享语料上的题目分组留出，未实现完全事实独立的泛化测试；原题与同义题随来源一起分组。人工题集规模、作者风格与线上分布差异均可能影响结果。私有结果无法由外部逐题复现。

## 复现公开部分

以下命令对应本仓库源码；本次整理验证依赖安装、合成 demo 和测试，未重跑模型评测。

Python 3.10+（含 venv/ensurepip 支持），SQLite 3.43+ 且启用 FTS5；冻结 CPU 依赖在 Python 3.14 环境验证，其他版本可能需要相应 wheel。首次安装及下载模型需联网；不需模型API密钥。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r eval/requirements-cpu-lock.txt
.venv/bin/python -m pip install -e '.[models]' pytest
.venv/bin/memlib demo
.venv/bin/python -m pytest -q
mkdir -p ../memlib-artifacts/models
```

阅读 [LoCoMo 上游许可](https://github.com/snap-research/locomo/blob/main/LICENSE.txt) 并确认用途符合非商业条件后，自行下载固定版本；下载、缓存和结果均留在仓库外：

```sh
curl -fL https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json -o ../memlib-artifacts/locomo10.json
printf '%s  %s\n' 79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4 ../memlib-artifacts/locomo10.json | sha256sum -c -
.venv/bin/python -c 'from memlib.index import SentenceEmbedder; SentenceEmbedder("../memlib-artifacts/models")'
.venv/bin/python -m eval.run --data ../memlib-artifacts/locomo10.json --cache ../memlib-artifacts/models --out ../memlib-artifacts/results-public
```

输出 summary.json/md 与逐题CSV；复现时核对数据哈希、模型revision、分词器、排除数量与评分协议，耗时随硬件变化。MiniLM 固定 revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`，归一化、256 token；英文分词 english-lower-alnum-v1。RRF默认k=60、每路50候选。合成demo不证明真实检索质量。

BGE 可选实验使用 `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`，CPU、dense/CLS、L2、512 token；按模型卡下载该revision到仓库外 `models/bge-m3-5617a9f61b028005a4858fdac845db406aefb181/`，核对文件manifest后给上述评测命令加 `--model bge` 并换新输出目录。本库只读本地BGE文件。P4 脚本未收入本仓库：仍缺从公开卷确定性生成跨对话候选的链路，不提供一键复现承诺；私有P6–P9不属于公开复现承诺。

通用关键词标定脚本及全合成输入见 [keyterm 示例](examples/keyterm/README.md)；输出可能含语料词表与逐题特征，使用私有输入时必须保护输出。

## 路线图与分工

- 意图模型：比较便宜LLM分类与轻量分类器，按会话/事实隔离训练验证测试，分别报意图漏判、存在误出和最终Pack召回。
- 影子跑：经数据所有者授权，在真实连续对话旁路检索，不实际注入；严格只搜消息发生前的记忆，人工复核相关性与隐式回忆。
- 补齐 P4 公开跨对话负例生成链及版本 manifest。

方案、评测设计与判断由 **Anya** 主导；代码由 **AI（Codex）** 实现。阶段内部分题目、改写与原型由执行助手编写，尚未完成独立双人盲审；不把它描述成人工专家独立标注。

自有代码采用 [MIT](LICENSE)；LoCoMo 为 CC BY-NC 4.0，bge-m3、jieba 为 MIT，MiniLM 模型为 Apache-2.0，各自许可独立，不由本库MIT覆盖。

产品演示见 [Valelin](https://github.com/OWNER/valelin)（待建仓后替换 OWNER）。Valelin 演示交互流程，当前没有接入 memlib；本项目的离线指标不代表该前端的在线效果。

## 结果依据

公开协议和聚合结果见 [评测记录](docs/evaluation/README.md) 与 [P2 基线](eval/results/locomo-baseline.md)。私有评测仅提供聚合数字与限制，无法逐题复现。
