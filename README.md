# RUL-CAF

**Condition-adaptive fusion for aero-engine remaining useful life prediction.**

This repository contains the manuscript sources, figures, and supporting code
for a study of silent training collapse in hybrid deep RUL predictors and the
condition-adaptive design rules that avoid it.

- **Paper:** *Silent collapse of hybrid deep RUL predictors: a mechanism study
  and condition-adaptive design rules for multi-condition aero-engine
  prognostics*
- **Target journal:** *Mechanical Systems and Signal Processing* (Elsevier)
- **Benchmark:** NASA C-MAPSS, subsets FD001–FD004
- **Status:** submitted

## What we found

Across 161 training runs of a convolutional–recurrent–attention backbone, runs
on the two multi-condition subsets often collapsed silently: 37.2% (FD002) and
24.2% (FD004) converged to a constant predictor that outputs the training-label
mean for every engine. The loss curve still looked healthy, yet the asymmetric
prognostics score (PHM08) was 15–250 times worse. Instrumented training showed
that recovery depends on the convolutional branch's gradient rising sharply
before early stopping, and that the learned features encode operating condition
rather than degradation. Choosing window length and depth by condition count,
keeping branches that carry degradation signal, and running three cheap
post-training checks make the failure detectable and avoidable.

## Repository layout

```
.
├── compiled/            # ready-to-read PDFs of the manuscript
│   ├── manuscript.pdf                       # main submission (cas-sc, 12 pp)
│   ├── manuscript_reading_copy.pdf          # single-column reading copy
│   └── manuscript_two_column_preview.pdf    # two-column layout proof
├── manuscript/          # LaTeX sources ( Elsevier cas-sc / elsarticle )
│   ├── main_cas.tex, frontmatter_cas.tex    # official single-column build
│   ├── main.tex, main_preprint.tex          # two-column / reading builds
│   ├── sec1_intro.tex … sec7_discussion.tex
│   ├── bibliography_gen.tex, refs.bib, gen_biblio.py
│   └── uncited_refs.tex                     # bookkeeping, not for submission
├── figures/
│   ├── vector_pdf/          # 8 publication figures, vector PDF
│   ├── raster_png_300dpi/   # same 8 figures at 300 dpi
│   └── editable/            # editable PPTX for figures 1 and 6
├── supplementary/
│   └── Supplementary_Material.docx          # S1–S10, 12 tables, 3 figures
├── cover_letter/        # cover_letter.tex / .pdf
├── highlights.txt       # 5 highlights, each <= 85 characters
├── figure_scripts/      # regenerate all figures and the Word supplement
└── experiment_scripts/  # E1/E2 drivers, PHM08 ablation, aggregate results
```

Figure index: `fig1_architecture`, `fig2_fingerprint`, `fig3_prevalence`,
`fig4_tsne`, `fig5_escape_race`, `fig6_phase_diagram`, `fig8_attention_maps`,
`fig9_seed_boxplot`. Five are used in the main text; the other three appear in
the supplementary document.

## Reproducing

The figure and experiment scripts were developed on the author's machine and
contain absolute paths (`ROOT`, `OUT`, `EXP` variables near the top of each
file). Point these at your own checkout and data directory before rerunning.

```bash
# from figure_scripts/
python3 gen_paper_figures.py          # figs 2/3/4/5/9 -> PNG + PDF
python3 gen_paper_figures_extra.py    # fig 8
python3 gen_fig1_fig6_pptx.py         # editable PPTX + PNG + PDF for figs 1/6
python3 gen_supplementary_docx.py     # Supplementary_Material.docx

# from manuscript/, with a TeX engine installed (e.g. tectonic or pdflatex)
tectonic main_cas.tex                 # main submission PDF
```

`experiment_scripts/` holds the drivers (`e1_seeds.py`, `e2_probe_train.py`,
`scan_collapse.py`, `run_paper_experiments.sh`) and the small aggregate results
they produced (JSON/CSV). The large probe-feature arrays and raw run logs are
not included.

## Data

The NASA C-MAPSS dataset is distributed by the NASA Prognostics Data Repository
and is **not** redistributed here. Download FD001–FD004 from the repository and
place them where your local scripts expect them.

## Authors

- Chi Zhang — first author
- Yuguo Zha — corresponding author, hugozha@nchu.edu.cn

Affiliation: Department of Biomedical Engineering, College of Instrument
Science and Optoelectronic Engineering, Nanchang Hangkong University,
Nanchang 330063, Jiangxi, China.

## Citation

Citation details will be added once the paper is published. In the meantime,
please cite the manuscript above.

## License

Code and figures are provided for academic review. For reuse beyond
non-commercial research, please contact the corresponding author.
