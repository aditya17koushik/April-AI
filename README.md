# April AI — Phase 4: Preprocessing Studio

April AI is a local-first data science workbench. This release includes dataset intake, profiling, exploratory data analysis (EDA), and a configurable preprocessing preview studio. It does not train models yet.

## What's included

- React + TypeScript + Vite frontend with responsive layouts, subtle motion, hover effects, and reduced-motion accessibility support
- Python + FastAPI backend
- CSV, XLSX, XLS, and Parquet uploads (25 MB default maximum)
- Dataset preview, schema, missing-value and duplicate-row profiling
- Numeric histograms, descriptive statistics, IQR outlier candidates, Pearson correlation heatmap, and categorical frequency summaries
- Rule-based findings for missing values, outliers, skew, strong correlations, duplicates, and high-cardinality categories
- Phase 4 preprocessing studio with configurable missing-value handling, IQR outlier capping or row removal, one-hot/label encoding, standard/min-max scaling, numeric pairwise interaction features, and constant/correlation-based feature selection
- Before/after shape comparison, transformation audit trail, dropped-column list, and transformed data preview
- API tests for upload, EDA, and preprocessing workflows

## Requirements

- Python 3.11+ recommended
- Node.js 20+ and npm

## Start the backend (Windows PowerShell)

```powershell
cd april-ai\backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

If PowerShell blocks activation, run this for the current terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

- Health: http://localhost:8000/api/health
- Interactive API docs: http://localhost:8000/docs

## Start the frontend (second terminal)

```powershell
cd april-ai\frontend
npm install
npm run dev
```

Open http://localhost:5173.

If the API runs elsewhere, create `frontend/.env.local`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Restart Vite after changing environment variables.

## Run backend tests

From `april-ai\backend`, with the virtual environment activated:

```powershell
pytest -q
```

## API endpoints

- `GET /api/health` — service status
- `POST /api/datasets/inspect` — returns dataset profile, preview, and EDA summaries
- `POST /api/datasets/preprocess` — accepts the same `file` upload plus form fields: `missing_strategy`, `outlier_strategy`, `encoding_strategy`, `scaling_strategy`, `feature_engineering`, `feature_selection`, `correlation_threshold`, and `constant_value`

## Responsible preprocessing notes

- IQR fences flag candidates; outliers are not automatically errors.
- Label-encoded values are arbitrary category codes and should not be interpreted as rankings.
- Feature selection based on correlation can remove useful domain-specific features; inspect the audit trail before using it.
- The preprocessing endpoint creates a preview from a fresh copy of the uploaded data. It does not overwrite the original file or persist data.
- **Leakage prevention:** before model evaluation, fit imputation, encoding, scaling, and feature selection on training data only, then apply those fitted transformations to validation/test data. This release explicitly warns about this, but does not yet build the train/validation/test pipeline.
- Dataset files are processed in memory. Do not upload sensitive data to an environment you do not control.

## Roadmap

1. Foundation and upload — complete
2. Automated EDA and visualizations — complete
3. Preprocessing Studio — current
4. Task-aware algorithm registry and classical ML
5. Deep learning and environment-based reinforcement learning
6. Insights, explainability, model export, and experiment history


## Phase 4: Model Studio

The Model Studio adds baseline classification and regression model training. It compares Logistic Regression, Decision Tree, and Random Forest for classification, and Linear Regression, Decision Tree Regressor, and Random Forest Regressor for regression. Preprocessing is fitted inside scikit-learn pipelines on the training split only.

- API endpoint: `POST /api/models/train`
- Classification metrics: accuracy, weighted precision/recall/F1, and confusion matrix
- Regression metrics: MAE, RMSE, and R²
- Best model artifact: `backend/artifacts/<model_id>.joblib`
- Run backend tests: `pytest -q` from `backend`

This is a baseline experimentation feature. Review class balance, validate model quality, and use appropriate cross-validation before production deployment. Model artifacts are local files and are not yet managed by an artifact registry.


## Phase 5 — Model Intelligence Studio

Phase 5 adds a saved-model registry, lightweight global feature-importance views for estimators that expose coefficients or tree importance, and an interactive prediction playground. The winning fitted pipeline is saved with its task, target, feature schema, and training summary.

### Start the backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:MAX_UPLOAD_MB = "100"
pytest -q
uvicorn app.main:app --reload --port 8000
```

The upload limit defaults to 100 MB and can be configured with `MAX_UPLOAD_MB`. Ensure the frontend upload limit is set to the same value if you change it.

### Start the frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The Intelligence section lists saved artifacts even when no dataset is currently selected. Train a model in Model Studio first; then select it in the registry to inspect supported feature-importance signals and run a prediction.

### Notes

- Prediction input fields follow the saved model's original feature schema. Numeric inputs are converted to numbers; categorical values are passed through as text.
- Feature importance is available only for estimators exposing built-in feature importances or coefficients. It is a relative global signal and is not causal evidence.
- Model artifacts are stored under `backend/artifacts/`. Treat them as trusted local files and do not load artifacts from untrusted sources.
- Frontend production build was not verified in this environment because npm dependency installation timed out.
