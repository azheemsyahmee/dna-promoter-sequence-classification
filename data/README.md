# Molecular Biology (Promoter Gene Sequences) Dataset

## 1. Overview & Provenance
* **Dataset Name:** Molecular Biology (Promoter Gene Sequences)
* **Source:** UCI Machine Learning Repository
* **Citation:** Towell, G. G., Shavlik, J. W., & Noordewier, M. O. (1990). *Refinement of Approximate Domain Theories by Knowledge-Based Neural Networks*. In Proceedings of the Eighth National Conference on Artificial Intelligence (AAAI-90), pp. 861-866.
* **Access URL:** [UCI Machine Learning Repository: Molecular Biology (Promoter Gene Sequences)](https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences)
* **File:** `promoters.data` (7,130 bytes, 106 instances)

## 2. Biological Context
Promoters are DNA sequences that define where transcription of a gene by RNA polymerase begins. They are typically located directly upstream or at the start site of transcription.

In *Escherichia coli* (*E. coli*), promoter sequences are recognized by the $\sigma^{70}$ factor of RNA polymerase, notably through consensus hexamers at the **-35 region** (`TTGACA`) and the **-10 region** (`TATAAT`, Pribnow box), separated by an optimal spacer of 17 base pairs.

## 3. Dataset Characteristics
* **Total Instances:** 106
  - **Positive Class (`+`):** 53 verified *E. coli* promoter sequences (starting from position -50 to +7 relative to the transcription start site).
  - **Negative Class (`-`):** 53 non-promoter DNA fragments selected from non-promoter regions in the *E. coli* genome.
* **Sequence Length:** Exactly 57 base pairs per instance.
* **Class Balance:** Perfectly balanced (50% positive / 50% negative).
* **Format:** Comma-separated valuesDenoting:
  `[label],[instance_id],\t\t[sequence]`
  where `label` is `+` or `-`, `instance_id` is the biological gene or clone name (e.g., `S10`, `AMPC`, `RECA`), and `sequence` is the 57 bp DNA string.

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
