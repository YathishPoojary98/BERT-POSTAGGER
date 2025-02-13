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
│── prepare_splits_train.py      # Training script
│── run_pos.py                   # POS tagging script
│── BERT POS.ipynb               # Jupyter Notebook for preprocessing
│── new_pos_train_data.pickle    # POS tag training data
│── new_pos_encoding.pickle      # POS tag encoding dictionary
│── output_dir/                  # Saved fine-tuned model
```

---

## 📊 Example POS-Tagged Output
**Example input (`input.txt`)**:
```
ಅದರಂತೆ
ಶಾಲೆಗಳಲ್ಲಿ
ಹೈಟೆಕ್
ಶೌಚಾಲಯಗಳ
ನಿರ್ಮಾಣದಿಂದ
'
ಸ್ವಚ್ಛತೆಯೇ
ಸೇವೆ
'
ಘೋಷಣೆಯನ್ನು
ಸಾಬೀತುಪಡಿಸಿದೆ
.

ಶಾಲೆಗಳ
ಆವರಣದಲ್ಲಿ
ಆಟದ
ಮೈದಾನಗಳ
ಉನ್ನತೀಕರಣದ
ಮೂಲಕ
ಕ್ರೀಡಾ
ಚಟುವಟಿಕೆಗಳಿಗೆ
ಉತ್ತೇಜನ
ನೀಡಲು
ಆಗಿದೆ
.
```
**Generated output (`output.txt`)**:
```
<Sentence id='1'>
1	ಅದರಂತೆ	CC__CCS
2	ಶಾಲೆಗಳಲ್ಲಿ	N__NN
3	ಹೈಟೆಕ್	JJ
4	ಶೌಚಾಲಯಗಳ	N__NN
5	ನಿರ್ಮಾಣದಿಂದ	N__NN
6	'	RD__PUNC
7	ಸ್ವಚ್ಛತೆಯೇ	N__NN
8	ಸೇವೆ	N__NN
9	'	RD__PUNC
10	ಘೋಷಣೆಯನ್ನು	N__NNV
11	ಸಾಬೀತುಪಡಿಸಿದೆ	V__VM__VF
12	.	RD__PUNC
</Sentence>

<Sentence id='2'>
1	ಶಾಲೆಗಳ	N__NN
2	ಆವರಣದಲ್ಲಿ	N__NN
3	ಆಟದ	N__NNV
4	ಮೈದಾನಗಳ	N__NN
5	ಉನ್ನತೀಕರಣದ	N__NN
6	ಮೂಲಕ	RP__RPD
7	ಕ್ರೀಡಾ	N__NN
8	ಚಟುವಟಿಕೆಗಳಿಗೆ	N__NN
9	ಉತ್ತೇಜನ	N__NN
10	ನೀಡಲು	V__VM__VF
11	ಆಗಿದೆ	V__VAUX
12	.	RD__PUNC
</Sentence>
```

---
