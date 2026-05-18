# Violence-Research
<div align="center">

# 🇧🇩 Multi-Source Bangla Violence Text Dataset  
## and Transformer-Based Stacking Ensemble for Social Media Content Moderation

### A Bangla NLP research project for detecting violence-related social media text using a **BX Stacking Ensemble** of BanglaBERT and XLM-RoBERTa

<br>

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![Bangla NLP](https://img.shields.io/badge/Bangla-NLP-green?style=for-the-badge)
![Macro F1](https://img.shields.io/badge/Macro%20F1-0.85-purple?style=for-the-badge)

<br>

**Violence Detection | BanglaBERT | XLM-RoBERTa | Stacking Ensemble | Content Moderation | Low-Resource NLP**

</div>

---

## 🌍 Overview

Social media platforms such as **Facebook, YouTube, and TikTok** have become major spaces for public communication in Bengali-speaking communities. However, the rapid growth of user-generated content has also increased the spread of harmful, aggressive, and violence-inciting language.

This research introduces a **multi-source Bangla violence text dataset** and a transformer-based **BX Stacking Ensemble Model** for detecting violence-related Bangla social media content.

The proposed model combines:

- **BanglaBERT** — for language-specific Bangla representation  
- **XLM-RoBERTa** — for multilingual contextual understanding  
- **Stacking Ensemble Classifier** — to merge both representations for stronger classification  

The final system achieves a **Macro F1-score of 0.85**, outperforming traditional machine learning models and individual transformer baselines.

---

## ✨ Key Contributions

| Contribution | Description |
|---|---|
| 📌 **New Multi-Source Dataset** | Built a larger Bangla violence text dataset with **11,933 samples** from Vio-Lens, YouTube, Facebook, and TikTok |
| 🧠 **BX Stacking Ensemble** | Proposed a transformer ensemble combining **BanglaBERT + XLM-RoBERTa** |
| 📊 **Three-Class Classification** | Classified text into **Non-Violence**, **Passive Violence**, and **Active Violence** |
| ⚖️ **Class-Imbalance Handling** | Used weighted loss and stratified train-validation-test splitting |
| 🚀 **Improved Performance** | Achieved the best reported performance among tested models with **Macro F1 = 0.85** |
| 🌐 **Real-World Relevance** | Designed for safer Bangla digital communication and scalable content moderation |

---

## 🧾 Paper Information

**Title:**  
**Multi-Source Bangla Violence Text Dataset and Transformer-Based Stacking Ensemble for Social Media Content Moderation**

**Authors:**  
- Maliha Rahman Moon  
- Md. Siam  
- Asmita Hoque Pospita  
- Hrithik Majumdar Shibu  

**Affiliations:**  
- Department of Computer Science and Engineering, R. P. Shaha University, Bangladesh  
- School of Computer Science and Electronic Engineering, University of Essex, United Kingdom  

---

## 🗂️ Dataset

The dataset contains **11,933 Bangla social media text samples** collected from multiple sources.

### Dataset Sources

| Source | Description |
|---|---|
| **Vio-Lens Dataset** | Existing Bangla communal violence text dataset |
| **YouTube** | Public comments from socially and politically sensitive events |
| **Facebook** | Public comments from news pages and public posts |
| **TikTok** | Public short-form content reactions and comments |

### Class Labels

| Label | Class Name | Meaning |
|---|---|---|
| `0` | **Non-Violence** | Peaceful, neutral, or harmless content |
| `1` | **Passive Violence** | Indirect aggression, discriminatory tone, sarcasm, or justification of violence |
| `2` | **Active Violence** | Direct violent expression, threat, or abusive aggression |

### Final Class Distribution

| Class | Samples | Percentage |
|---|---:|---:|
| Non-Violence | 5,152 | 43.2% |
| Passive Violence | 3,907 | 32.7% |
| Active Violence | 2,874 | 24.1% |
| **Total** | **11,933** | **100%** |

---

## 🔎 Dataset Characteristics

The dataset captures the complexity of real-world Bangla social media language, including:

- informal spelling variations  
- slang and colloquial expressions  
- religious and socio-political vocabulary  
- platform-specific short comments  
- code-mixing and noisy text  
- short and long text variations  

Text length statistics:

| Metric | Value |
|---|---:|
| Minimum Text Length | 3 |
| Maximum Text Length | 640 |
| Mean Text Length | 90.49 |
| Standard Deviation | 78.54 |

---

## 🧹 Preprocessing Pipeline

The dataset was cleaned and prepared using the following steps:

```mermaid
flowchart LR
    A[Raw Social Media Text] --> B[Remove Invalid Entries]
    B --> C[Emoji and Noise Removal]
    C --> D[Bangla-Aware Tokenization]
    D --> E[Stopword Filtering]
    E --> F[Text Cleaning]
    F --> G[Train / Validation / Test Split]
