# bug-predictor

A Just-In-Time software defect prediction system that analyzes Git commits the moment they are pushed and predicts whether they will introduce a recurring bug.

## What it does

When a developer pushes code to a connected GitHub repository, bug-predictor receives the commit via webhook, computes 14 change metrics from the repository's git history, and runs a trained XGBoost classifier to estimate the probability that the commit will cause a bug. Each prediction includes a risk level (High, Medium, Low), a severity score, and a SHAP explanation showing exactly which factors drove the decision.

## Prerequisites

- Python 3.12
- Node.js 18+
- Git
- CUDA-capable GPU recommended for embedding generation (CPU works too)

## Setup

Clone the repository:
    git clone https://github.com/tarunchowdari/bug-predictor.git
    cd bug-predictor

Create and activate a virtual environment:
    python -m venv .venv
    .venv\Scripts\Activate.ps1

Install Python dependencies:
    pip install -r requirements.txt

Install dashboard dependencies:
    cd dashboard
    npm install
    cd ..

Run the data pipeline (first time only):
    python pipeline/00_download_data.py --skip-checksum
    python pipeline/01_mine.py --max-commits 5000
    python pipeline/02_preprocess.py
    python pipeline/03_features.py
    python pipeline/03b_build_source_cache.py
    python pipeline/04_embed.py --device cuda
    python pipeline/05_split.py

Train the model:
    python models/baseline/train.py

Start the API:
    uvicorn api.main:app --reload --port 8000

Start the dashboard:
    cd dashboard
    npm run dev

The API will be available at http://localhost:8000 and the dashboard at http://localhost:5173.

## Connecting a GitHub repository

Expose your local API using ngrok:
    ngrok http 8000

In your GitHub repository go to Settings -> Webhooks -> Add webhook and set:
    Payload URL: https://YOUR_NGROK_URL/webhook/github
    Content type: application/json
    Events: Just the push event

Every commit pushed to that repository will now be automatically scored.

## Understanding the risk scores

Each prediction returns a bug probability between 0 and 1 and a risk level:
    High   — above 60% probability, review before merging
    Medium — 34 to 60%, worth a closer look
    Low    — below 34%, likely safe

The SHAP explanation shows which commit properties drove the score. Positive values pushed toward buggy, negative values pushed toward clean. The most influential features are typically FIX (bug-fix keyword in the message), EXP (author experience in the repo), AGE (how recently the modified files were last touched), and NUC (how many prior commits have touched the same files).

## Architecture

The data pipeline mines commits from GitHub using PyDriller and the Kamei PROMISE benchmark dataset, computes 14 change metrics per commit, generates 768-dimensional BERT embeddings of commit messages using all-MiniLM-L6-v2, and stores everything as Parquet files for training.

The model is a two-stage XGBoost system. Stage 1 is a binary classifier that predicts whether a commit will introduce a recurring bug. Stage 2 is a regression model that estimates severity, conditional on Stage 1 flagging the commit as buggy. The model was trained on 60,454 commits and evaluated on 25,959 commits from the Eclipse Platform project, achieving AUC-ROC of 0.763 and recall of 0.932.

The API is built with FastAPI and exposes three endpoints: POST /predict for single commit scoring, POST /webhook/github for GitHub push event integration, and GET /predictions/recent for the dashboard feed. When a webhook arrives the API clones or fetches the repository locally and computes all Kamei metrics from real git history before running the prediction.

## Model performance

    Test AUC-ROC  : 0.763
    Test F1       : 0.608
    Test Recall   : 0.932
    Test Precision: 0.451
    Threshold     : 0.3388

High recall means the model catches 93% of commits that introduce bugs. Precision of 0.45 means roughly half of flagged commits are false alarms — a known limitation that will improve with diff text embeddings in the next version.
