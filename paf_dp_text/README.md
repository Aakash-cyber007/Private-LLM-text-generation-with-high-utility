# PaF-DP-Text

**Permute-and-Flip for Differentially Private Text Generation**  
A direct comparison with InvisibleInk (NeurIPS 2025) using identical privacy accounting.

---

## What is this?

This repository implements the **Permute-and-Flip (PaF)** token selection mechanism as a drop-in replacement for the softmax sampling used in InvisibleInk. Both methods use the same:

- Difference Clipping (DClip) 
- Top-k+ sampling mask
- zCDP privacy accounting

The **only difference** is the final token selection step:

| Method | Token selection | Distribution |
|--------|----------------|--------------|
| InvisibleInk | Softmax → Categorical sample | Exponential mechanism |
| **PaF (ours)** | Gumbel-max / arrival-time argmin | Exponential mechanism (proved optimal) |

PaF is **provably at least as accurate** as the exponential mechanism in expectation (McKenna & Sheldon, 2020).

---

## One-button usage (Google Colab T4)

```python
# Cell 1: setup
!git clone https://github.com/YOUR_USERNAME/paf-dp-text.git
%cd paf-dp-text
import os; os.environ['HF_TOKEN'] = 'hf_your_token'
!python setup.py --hf_token hf_your_token

# Cell 2: open notebook
# Runtime → Open in Colab → PaF_vs_InvisibleInk.ipynb
```

The notebook runs everything: generation, MAUVE scoring, perplexity, text showcase, embedding visualisation.

---

## Repository structure

```
paf-dp-text/
├── setup.py              ← one-button env + dataset setup
├── requirements.txt      ← pinned deps (torchcodec-free)
├── utils_paf.py          ← shared utilities (privacy math, prompts, embeddings)
├── generate_paf.py       ← Permute-and-Flip generation
├── generate_invink.py    ← InvisibleInk baseline (same codebase)
├── PaF_vs_InvisibleInk.ipynb  ← full comparison notebook
├── data/
│   └── yelp/             ← prepared by setup.py
└── results/
    ├── paf/              ← PaF outputs
    └── invink/           ← InvisibleInk outputs
```

---

## Dataset

**Yelp Open Dataset** — chosen because:
- Public (no credentialed access unlike MIMIC)
- Rich categorical structure (10 categories × 5 star ratings = 50 classes)
- Short-to-medium texts fit well in T4 VRAM at `T=200` tokens
- Ideal for conditional generation quality assessment

Setup downloads and splits the dataset automatically.

---

## Memory guide (T4 15 GB)

| batch | minibatch | VRAM usage | Status |
|-------|-----------|------------|--------|
| 8 | 4 | ~11 GB | ✓ Safe |
| 8 | 2 | ~8 GB | ✓ Very safe (`--low_vram`) |
| 16 | 4 | ~14 GB | ⚠ Borderline |
| 32 | 4 | OOM | ✗ Don't use |

---

## Citation

If you use this code, please also cite the InvisibleInk paper it builds on:

```bibtex
@inproceedings{vinod2025invisibleink,
  author    = {Vishnu Vinod and Krishna Pillutla and Abhradeep Thakurta},
  title     = {InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy},
  booktitle = {NeurIPS},
  year      = {2025},
}

@inproceedings{mckenna2020permute,
  author    = {Ryan McKenna and Daniel Sheldon},
  title     = {Permute-and-Flip: A new mechanism for differentially private selection},
  booktitle = {NeurIPS},
  year      = {2020},
}
```
