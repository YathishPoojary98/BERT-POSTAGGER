# 🏷️ BERT POS Tagger

This repository contains a **Part-of-Speech (POS) Tagger** using **BERT-based models** for token classification. It supports **training, prediction, and evaluation** of POS tagging using a pre-trained **Indic-BERT model**.

---

## 🚀 Features

✅ **Pretrained Indic-BERT Model** – Uses `ai4bharat/indic-bert` for accurate token classification.  
✅ **POS Encoding Dictionary** – Maps POS tags to numerical labels using `pos_encoding.pickle`.  
✅ **Train & Fine-Tune** – Train the model using **Hugging Face's Trainer API**.  
✅ **Prediction & Inference** – Convert input text into **POS-tagged** sentences.  
✅ **Handles Ambiguities** – Ensures token alignment and label mapping for robust tagging.  
✅ **Supports Dataset Splitting** – Uses train, validation, and test splits from a dataset.  

---

## 📥 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YathishPoojary98/BERT-POSTAGGER.git
```
Navigate into the cloned directory:
```bash
cd BERT-POSTAGGER
```

### 2️⃣ Install Dependencies
Install the required Python packages:
```bash
pip install transformers datasets numpy torch pickle5
```

---

## 🏋️ Training the Model

### **Train the Model with `prepare_splits_train.py`**
To start training:
```bash
python prepare_splits_train.py
```
This script will:
- Load the dataset from pickle file
- Encode POS tags using `pos_encoding.pickle`
- Tokenize and align labels
- Train the model using Hugging Face's Trainer API
- Save the fine-tuned model to `output_dir`

---

## 🎯 POS Tagging Inference

### **Run Predictions with `run_pos.py`**
To use the trained model for POS tagging:
```bash
python run_pos.py --input input.txt --output output.txt --model output_dir
```
#### Arguments:
- `--input` → Path to input text file
- `--output` → Path to save POS-tagged output
- `--model` → Path to the trained model directory

This script will:
- Tokenize input sentences
- Predict POS tags using the fine-tuned model
- Write tagged sentences to `output.txt`

---

## 📂 Repository Structure

```
BERT-POSTAGGER/
│── prepare_splits_train.py  # Training script
│── run_pos.py               # POS tagging script
│── BERT POS.ipynb           # Jupyter Notebook for preprocessing
│── pos_encoding.pickle      # POS tag encoding dictionary
│── output_dir/              # Saved fine-tuned model
```

---

## 📊 Example POS-Tagged Output
**Example input (`input.txt`)**:
```
This is an example sentence.
```
**Generated output (`output.txt`)**:
```
<Sentence id='1'>
1   This    DET
2   is      VERB
3   an      DET
4   example NOUN
5   sentence NOUN
</Sentence>
```

---
