# RUL-CAF

**Condition-adaptive fusion for aero-engine remaining useful life prediction.**

Companion repository for our MSSP submission *"Constant predictions in
multi-condition aero-engine RUL models: causes, early detection, and
condition-adaptive network design"*. The repository contains only what a
reviewer needs to read the paper and to reproduce the experiments.

## Contents

```
manuscript.pdf              # submitted paper (Elsevier cas-sc, 12 pp)
manuscript.docx             # Word copy of the same paper
manuscript/                 # LaTeX sources; compile with: tectonic main_cas.tex
figures/vector_pdf/         # 8 publication figures, vector PDF
figures/raster_png_300dpi/  # same 8 figures at 300 dpi (also used by LaTeX)
supplementary/              # Supplementary_Material.docx (S1-S10)
cover_letter/               # cover letter (source and PDF)
highlights.txt              # 5 highlights
figure_scripts/             # scripts that regenerate all figures and the Word supplement
experiment_scripts/         # E1/E2 drivers, PHM08 ablation, aggregate JSON/CSV results
```

## Data

NASA C-MAPSS (FD001–FD004) is available in the
[`data-v1` release](https://github.com/HugoZha/RUL-CAF/releases/tag/data-v1)
as `c-mapss.zip`. It contains the four train/test/RUL subsets and the original
readme. The experiment scripts in `experiment_scripts/` read these files;
point their `ROOT` variable at the extracted directory. The small aggregate
result files (JSON/CSV) are committed here so the reported tables can be
checked without rerunning training. Large probe-feature arrays and raw logs
are not included.

The C-MAPSS data are a U.S. Government work in the public domain distributed
by the NASA Prognostics Data Repository. We acknowledge the repository and the
data donors and cite Saxena and Goebel (2008) as the source; users employ the
data at their own risk.

## Reproducing

The scripts were developed on the authors' machine and carry absolute paths
in `ROOT`, `OUT` and `EXP` near the top of each file; adjust them to your own
checkout before running.

```bash
cd figure_scripts
python3 gen_paper_figures.py          # figs 2/3/4/5/9 -> PNG + PDF
python3 gen_paper_figures_extra.py    # fig 8
python3 gen_fig1_fig6_pptx.py         # figs 1/6
python3 gen_supplementary_docx.py     # Word supplement
python3 gen_manuscript_docx.py        # rebuild Word paper from LaTeX + .aux

cd ../manuscript
tectonic main_cas.tex                 # rebuild the submitted PDF
```

## Authors

- Chi Zhang — first author
- Yuguo Zha — corresponding author, hugozha.nchu@gmail.com

Affiliation: Department of Biomedical Engineering, College of Instrument
Science and Optoelectronic Engineering, Nanchang Hangkong University,
Nanchang 330063, Jiangxi, China.

**Funding.** Jiangxi Provincial Early-Career Talent Fund (20252BEJ730138) and
Jiangxi Provincial Talent Introduction and Cultivation Fund (117C00230023).

**Competing interests.** The authors declare none.

## License

Code and figures are provided for academic review. For reuse beyond
non-commercial research, contact the corresponding author.
