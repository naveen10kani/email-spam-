# Email Spam Classifier

A deployable Streamlit + scikit-learn NLP project that classifies text as spam or ham.

## Features
- TF-IDF text feature extraction (unigrams and bigrams)
- Compares Multinomial Naive Bayes, Logistic Regression, and Linear SVM
- Selects a model based on spam F1-score, with spam precision as a tie-breaker
- Reports spam precision/recall/F1, model comparison, and confusion matrix
- Interactive Streamlit UI for real-time text classification
- Downloads the UCI dataset automatically on first training run

## Dataset note
The default UCI SMS Spam Collection contains SMS messages, not full email messages or email headers. Treat this as an SMS/spam-text prototype. For an email-specific project, use a labelled email corpus and retrain the model.

## Run locally (Windows)
1. Install Python 3.10–3.12.
2. Open a terminal in this folder.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Train the models (downloads the dataset from UCI if not already present):

   ```powershell
   python train_model.py
   ```

6. Start the app:

   ```powershell
   streamlit run app.py
   ```

The browser should open at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository, for example `email-spam-classifier`.
2. Upload `app.py`, `train_model.py`, `requirements.txt`, and `README.md`.
3. You may omit `data/` and `models/`; the app trains the models on first launch and caches the loaded model while running. Alternatively, run `python train_model.py` locally and commit the generated `models/` files so the deployed app starts without downloading/training.
4. On Streamlit Community Cloud, choose **Create app**, select the repository, branch `main`, and entrypoint `app.py`, then deploy.
5. Check the app logs if startup fails. The first run can take longer because it may download the dataset and train models.

## Project files
- `app.py`: Streamlit user interface
- `train_model.py`: data loading, training, evaluation, model export
- `requirements.txt`: reproducible dependency versions
- `models/`: generated artifacts, comparison CSV, confusion matrix, metadata

## Limitations
- Predictions are not guaranteed; use for educational demonstration only.
- A text-only model cannot inspect sender reputation, headers, attachments, or URL destinations.
- Probability values are displayed only when the selected model supports them; SVM decision scores are not presented as probabilities.
