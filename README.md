# Spam Detection Classifier

## Overview

This project implements a complete spam detection pipeline using multiple machine learning approaches, progressing from classical models to deep learning and finally to a compressed model using knowledge distillation.

The system performs **multi-class classification**, where each email is categorized into one of three classes:

- **0 → Ham** (legitimate emails)
- **1 → Phish** (fraudulent emails attempting to steal sensitive information)
- **2 → Spam** (unsolicited or promotional messages)

It focuses on achieving high accuracy while also optimizing model size and efficiency for deployment.

---

## Approaches Used

### 🔹 Classical Machine Learning

* Logistic Regression (TF-IDF with unigrams + bigrams)
* Support Vector Machine (Linear SVM with class balancing)

These models serve as strong baselines for text classification.

---

### 🔹 Neural Network (Teacher Model)

* Large-scale model (~80M parameters)
* Word embeddings (512 dimensions)
* Global max pooling over sequences
* Multiple residual blocks with normalization and dropout

This model achieves the highest performance and captures complex patterns in text.

---

### 🔹 Knowledge Distillation (Student Model)

* Smaller model trained to mimic the teacher
* Reduced vocabulary (~20k) and embedding size (128d)
* Uses both:

  * True labels (CrossEntropy)
  * Teacher outputs (KL Divergence)

This allows significant model compression while maintaining high accuracy.

---

## Key Techniques

* TF-IDF vectorization for classical models
* Tokenization with fixed sequence length (150 tokens)
* Class weighting to handle imbalance
* Residual connections for stable deep training
* Attention mechanism in student model for efficiency
* Temperature scaling and loss balancing in distillation

---

## Performance Summary

| Model                     | Test Accuracy | Notes                            |
| ------------------------- | ------------- | -------------------------------- |
| Logistic Regression       | ~95.3%        | TF-IDF baseline                  |
| SVM                       | ~95.4%        | Improved margin-based classifier |
| Neural Network (Teacher)  | **~98.38%**   | High-capacity model              |
| Distilled Model (Student) | **~98.12%**   | Lightweight + efficient          |

---

## Model Efficiency

| Model   | Parameters | Purpose           |
| ------- | ---------- | ----------------- |
| Teacher | ~80M       | Maximum accuracy  |
| Student | ~2M        | Fast + deployable |

The distilled model achieves near-teacher performance with a fraction of the size.

---