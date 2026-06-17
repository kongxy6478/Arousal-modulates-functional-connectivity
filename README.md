
# Arousal modulates functional connectivity through structured and hemispherically asymmetric community architecture during wakefulness

[![DOI](https://img.shields.io/badge/DOI-10.7554/eLife.110294-blue)](https://doi.org/10.7554/eLife.110294)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/kongxy6478/Arousal-modulates-functional-connectivity/)](https://archive.softwareheritage.org/swh:1:dir:f7e3a4a3155c07074921f108debc1d33ad83d09a;origin=https://github.com/kongxy6478/Arousal-modulates-functional-connectivity;visit=swh:1:snp:3683c2f6802a16297bd23bd7c7222d337f63591e;anchor=swh:1:rev:86261f04b572686dd780ddf07e8921408cd2f7b8)

This repository contains the analysis code accompanying the eLife paper:

> **Kong X, Li S, Gong G.** Arousal modulates functional connectivity through structured and hemispherically asymmetric community architecture during wakefulness. *eLife* (2026). https://doi.org/10.7554/eLife.110294

---

## Overview

Arousal fluctuates continuously during wakefulness, yet how these moment-to-moment variations shape large-scale functional connectivity (FC) remains unclear. By combining 7T fMRI with concurrent pupillometry, we quantified, for every functional connection, how time-varying FC covaries with spontaneous arousal in the awake human brain. Rather than exerting a uniform influence across the connectome, arousal organized FC into a low-dimensional set of seven connectivity communities, each defined by characteristic network compositions. These communities exhibited systematic hemispheric asymmetries, identifying a "left-hemisphere centripetal architecture" where the left hemisphere serves as a structural sink for the asymmetric convergence of arousal-modulated signals. This modular and asymmetric organization was highly preserved during naturalistic movie watching, indicating that arousal-related modulation of FC reflects intrinsic principles that generalize across awake cognitive contexts.

---

## Repository structure

| File | Description |
|------|-------------|
| `main.py` | Master analysis pipeline: time-varying FC computation, arousal-tvFC coupling, edge community detection (k-means clustering), lateralization index (LI) calculation, and figure generation |
| `edge_analysis.py` | Core analysis functions: edge community metrics, node-level entropy and affiliation profiles, modality profiles (unimodal vs. heteromodal), network-pair composition analysis |
| `edge_wrapper.py` | High-level pipeline wrappers for community edge/node/coupling analyses, including hemispheric decomposition (LL/RR/LR/RL), network-level aggregation, null model generation, and LI statistical testing |
| `validation.py` | Validation and stability analysis: split-half reliability, parameter sensitivity across window sizes and lag values, Hungarian algorithm-based label alignment, Jaccard/Dice/ARI/NMI metrics, and visualization |

---

## Data availability

Raw and preprocessed Human Connectome Project (HCP) data are available at [https://db.humanconnectome.org/](https://db.humanconnectome.org/). Source data to replicate the results of the study are openly available at [https://github.com/kongxy6478/Arousal-modulates-functional-connectivity](https://github.com/kongxy6478/Arousal-modulates-functional-connectivity).

The code in this repository has been archived at [Software Heritage](https://archive.softwareheritage.org/browse/origin/https://github.com/kongxy6478/Arousal-modulates-functional-connectivity/) for long-term preservation.

## Dependencies

All analyses were implemented in Python using the following key packages:

- NumPy, SciPy
- scikit-learn (KMeans clustering, metrics)
- pandas
- Matplotlib, Seaborn (visualization)
- Surfplot (brain surface visualization)
- neuromaps (surface data handling)
- NiBabel (neuroimaging file I/O)
- pingouin, statsmodels (statistical testing)
- joblib (parallel computation)

## Usage

The main analysis pipeline is launched from `main.py`, which orchestrates the following workflow:

1. **tvFC computation** — Sliding-window functional connectivity is computed for resting-state and movie-watching runs across multiple parameter combinations (window size, step, lag).
2. **Arousal-tvFC coupling** — For each edge, Spearman correlation between arousal time series (pupil-indexed) and tvFC time series is computed, using the LL/RR/LR/RL hemispheric decomposition.
3. **Edge community detection** — K-means clustering is applied to the edge × run coupling matrix to identify connectivity communities. Optimal K is determined via the L-method (elbow) and validation metrics.
4. **Network mapping** — Edge communities are aggregated to the network level (Yeo 7-network parcellation), producing community-specific 7×7 network-pair matrices.
5. **Hemispheric asymmetry analysis** — Lateralization indices (LI) are computed for each community and network-pair, with statistical significance assessed against permutation-based null distributions.
6. **Validation** — Split-half reliability, parameter stability across window sizes and lags, and label alignment across parameter configurations.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{kong2026arousal,
  title={Arousal modulates functional connectivity through structured and hemispherically asymmetric community architecture during wakefulness},
  author={Kong, Xiangyu and Li, Siyu and Gong, Gaolang},
  journal={eLife},
  year={2026},
  doi={10.7554/eLife.110294}
}
```
