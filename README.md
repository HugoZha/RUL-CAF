# RUL-CAF

Companion repository for the manuscript *"Constant predictions in
multi-condition aero-engine RUL models: causes, early detection, and
condition-adaptive network design"*, submitted to *Mechanical Systems and
Signal Processing*.

This repository contains only the material needed to read and evaluate the
paper. The manuscript, supplementary material and figures are also being
submitted through the journal system; they are placed here for the
convenience of reviewers.

## Contents

| File | Description |
| --- | --- |
| `manuscript.pdf` | Submitted paper (Elsevier `cas-sc` layout) |
| `manuscript.docx` | Word copy of the same paper |
| `supplementary/Supplementary_Material.docx` | Supplementary sections S1–S10 and tables |
| `figures/vector_pdf/` | The eight figures of the paper, vector PDF |
| [`reproduction/`](reproduction/README.md) | Full reproduction package: code, exact data splits, evaluation, five-seed audit and tutorial |

## Reproduction

The `reproduction/` folder contains everything needed to regenerate the
reported experiments: the C-MAPSS preprocessing and 85/15 window splits, the
CAS-Fuse model (TCN + BiGRU + Transformer), training and evaluation scripts,
the multi-seed stability audit with collapse tripwires, and the archived
result files behind Tables 7–10. See
[`reproduction/REPRODUCE.md`](reproduction/REPRODUCE.md) for the complete
walkthrough (Chinese: [`REPRODUCE_CN.md`](reproduction/REPRODUCE_CN.md)).

## Data

The study uses the NASA C-MAPSS turbofan engine degradation simulation data
(FD001–FD004), a U.S. Government work in the public domain distributed by
the NASA Prognostics Data Repository. The subset files used, together with
aggregate results and per-epoch records, are provided in the
[`data-v1` release](https://github.com/HugoZha/RUL-CAF/releases/tag/data-v1)
as `c-mapss.zip`. We acknowledge the repository and the data donors and cite
Saxena and Goebel (2008) as the source; users employ the data at their own
risk.
