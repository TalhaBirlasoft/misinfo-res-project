# Design of Experiment (DoE)
## Project: Robust Detection of Misinformation (LIAR Dataset)

---

## 1. Research Problem (WHY)
Detect misinformation in short political statements and evaluate robustness of ML vs transformer-based models.

---

## 2. Current Limitations
- Classical ML struggles with semantic nuance.
- Class imbalance reduces performance on minority labels.
- Need stronger contextual understanding → transformers.

---

## 3. Proposed Approach (WHAT)
Compare:
1. TF-IDF + Logistic Regression (baseline)
2. DistilBERT fine-tuning (deep learning baseline)
3. Future: Hybrid LLM verification model

---

## 4. Evaluation Metrics
- Accuracy  
- Macro-F1 (main metric due to class imbalance)

---

## 5. Dataset
- **LIAR dataset**
- Train / Validation / Test official splits

---

## 6. Baseline Results

### TF-IDF + Logistic Regression
- Accuracy: **0.2432**
- Macro-F1: **0.2098**

Observations:
- Very low recall for **true** class → imbalance issue.
- Indicates need for contextual model (BERT).

---

## 7. Next Experiment
Fine-tune **DistilBERT** and compare improvement over classical baseline.
