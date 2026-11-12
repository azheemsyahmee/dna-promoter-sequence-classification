# Molecular Biology (Promoter Gene Sequences) Dataset

## 1. Overview & Provenance
* **Dataset Name:** Molecular Biology (Promoter Gene Sequences)
* **Repository:** UCI Machine Learning Repository (Dataset ID: 67)
* **Access URL:** [UCI Machine Learning Repository: Molecular Biology (Promoter Gene Sequences)](https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences)
* **File:** `promoters.data` (7,130 bytes, 106 instances)

### Academic Citations:
* **Official Dataset Citation (Biological Source & Sequence Compilation):**
  > Harley, C. B., Reynolds, C. W., & Noordewier, M. O. (1987). *Analysis of E. coli promoter sequences*. Nucleic Acids Research, 15(5), 2343–2361.
* **Related Machine Learning Reference (KBANN Knowledge-Based Neural Networks):**
  > Towell, G. G., Shavlik, J. W., & Noordewier, M. O. (1990). *Refinement of Approximate Domain Theories by Knowledge-Based Neural Networks*. In Proceedings of the Eighth National Conference on Artificial Intelligence (AAAI-90), pp. 861–866.

---

## 2. Biological Context
Promoters are DNA sequences that define where transcription of a gene by RNA polymerase begins. They are typically located directly upstream or at the start site of transcription.

In *Escherichia coli* (*E. coli*), promoter sequences are recognized by the $\sigma^{70}$ factor of RNA polymerase, notably through consensus hexamers at the **-35 region** (`TTGACA`) and the **-10 region** (`TATAAT`, Pribnow box), separated by an optimal spacer of approximately 17 base pairs.

---

## 3. Dataset Characteristics
* **Total Instances:** 106
  - **Positive Class (`+`):** 53 verified *E. coli* promoter sequences (starting from position -50 to +7 relative to the transcription start site).
  - **Negative Class (`-`):** 53 non-promoter DNA fragments selected from non-promoter regions in the *E. coli* genome.
* **Sequence Length:** Exactly 57 base pairs per instance.
* **Class Balance:** Perfectly balanced (50% positive / 50% negative).
* **Format:** Comma-separated values denoting:
  `[label],[instance_id],\t\t[sequence]`
  where `label` is `+` or `-`, `instance_id` is the biological gene or clone name (e.g., `S10`, `AMPC`, `RECA`), and `sequence` is the 57 bp DNA string.

---

## 4. Preprocessing & Encoding
* **Whitespace & Noise Removal:** Leading/trailing whitespace and internal tabs are stripped, and characters are converted to lowercase.
* **One-Hot Encoding:** Each nucleotide base is mapped to a 4-dimensional orthogonal unit vector:
  - Adenine (`a`): `[1, 0, 0, 0]`
  - Cytosine (`c`): `[0, 1, 0, 0]`
  - Guanine (`g`): `[0, 0, 1, 0]`
  - Thymine (`t`): `[0, 0, 0, 1]`
* **Tensor Dimensions:**
  - Feature matrix $X$: $(106, 57, 4)$
  - Target vector $y$: $(106,)$

---

## 5. Licensing, Terms of Use & Upstream Download

* **Redistribution & License Notice:** The dataset is hosted on the UCI Machine Learning Repository under the Creative Commons Attribution 4.0 International (CC BY 4.0) license (requiring donor attribution).
* **Attribution Requirement:** When using this data, please cite both the biological compilation and the benchmark machine learning paper:
  1. Harley, C. B., Reynolds, C. W., & Noordewier, M. O. (1987). *Analysis of E. coli promoter sequences*. Nucleic Acids Research, 15(5), 2343–2361.
  2. Towell, G. G., Shavlik, J. W., & Noordewier, M. O. (1990). *Refinement of Approximate Domain Theories by Knowledge-Based Neural Networks*. AAAI-90, pp. 861–866.
* **Upstream Download Instructions:**
  If you prefer to download the raw dataset directly from the UCI ML Repository rather than using the bundled file:
  1. Download the archive directly from the UCI archive portal:
     ```bash
     curl -L -o promoter_data.zip "https://archive.ics.uci.edu/static/public/67/molecular+biology+promoter+gene+sequences.zip"
     ```
  2. Extract `promoters.data` into the `data/` directory.
