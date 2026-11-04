# Third-party notices

The root MIT license covers this project's own code, not separately downloaded
data, models or dependencies. No model weights, upstream datasets, derived
source QA, retrieved original text or cached vectors are bundled in Git.

| Component | Version | Upstream license and attribution |
|---|---|---|
| LoCoMo | snap-research/locomo commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` | Maharana et al.; [CC BY-NC 4.0](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/LICENSE.txt) |
| bge-m3 | BAAI/bge-m3 revision `5617a9f61b028005a4858fdac845db406aefb181` | Beijing Academy of Artificial Intelligence; [MIT model card](https://huggingface.co/BAAI/bge-m3/blob/5617a9f61b028005a4858fdac845db406aefb181/README.md) |
| MiniLM | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` | sentence-transformers; [Apache-2.0 model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/blob/e8f8c211226b894fcb81acc59f3b34ba3efd5f42/README.md) |
| jieba | 0.42.1 | Sun Junyi; [MIT](https://github.com/fxsjy/jieba/blob/v0.42.1/LICENSE) |

LoCoMo is not redistributed here. The root README contains a fixed download
command and SHA256 verification (also provided by
`scripts/download_locomo.sh OUTSIDE_REPO/locomo10.json`) for `data/locomo10.json`:
`79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.
Confirm the noncommercial terms and attribution requirements before downloading;
commercial use requires separate authorization. Automatic download does not
remove the upstream license conditions. The upstream license and author/paper
references remain authoritative; no endorsement is implied.

Models are downloaded separately into external artifact directories. Preserve
upstream licenses, attribution and notices when redistributing their files.
jieba is installed as a dependency, not vendored; its bundled dictionary retains
its upstream license. Custom or mined dictionaries have their own provenance
and may contain private vocabulary; jieba's license does not authorize publishing
those inputs. Other transitive dependencies retain their respective licenses
in their installed distributions.
