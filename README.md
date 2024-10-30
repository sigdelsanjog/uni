# Project README

This README guides you through the process of building the entire project from scratch. Follow the steps outlined to organize, implement, and run the project.

---

## 1. Folder Structure

### 1.1 `data/`

The `data` folder contains raw text files for each faculty. These files store biographical details, publications, and other relevant information. The text will be preprocessed before training the model.

```
data/
├── Faculty1.txt
├── Faculty2.txt
└── FacultyN.txt
```
- **Objective**: Store raw text for each faculty to be used in the preprocessing phase.

---

### 1.2 `scripts/`

The `scripts` folder holds all the Python scripts required for data processing, model training, evaluation, and serving predictions via an API.

```
scripts/
├── process_data.py
├── train_model.py
├── evaluate_model.py
└── api.py
```

---

## 2. Explanation of Each Script

### 2.1 `process_data.py`

- **Objective**: Preprocess faculty text data from the `data/` folder.
  - Load raw text files.
  - Convert text to lowercase, remove punctuation, tokenize.
  - Save tokens to a `tokens.json` file.

---

### 2.2 `train_model.py`

- **Objective**: Train a Transformer-based model on the preprocessed text data.
  - Load tokens from `tokens.json`.
  - Build and train a Transformer model.
  - Save the trained model as an `.h5` file.

---

### 2.3 `evaluate_model.py`

- **Objective**: Evaluate the trained model and test its performance.
  - Load the trained model from the `.h5` file.
  - Perform predictions using the test data.
  - Log the model’s performance metrics.

---

### 2.4 `api.py`

- **Objective**: Serve predictions via a FastAPI-based REST API.
  - Load the trained model.
  - Expose an endpoint for users to submit text for predictions.
  - Return predictions as JSON responses.

