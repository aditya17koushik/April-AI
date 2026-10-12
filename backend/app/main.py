from __future__ import annotations

import io
import os
from typing import Any
from pathlib import Path
import math
import uuid

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)
import joblib

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

APP_NAME = "April AI API"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
PREVIEW_ROWS = 10
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}

MODEL_DIR = Path(__file__).resolve().parent.parent / "artifacts"

app = FastAPI(title=APP_NAME, version="0.5.0", description="April AI: dataset upload, profiling, EDA, preprocessing, and model training")

frontend_origins = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in frontend_origins.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _read_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    """Read a supported dataset from bytes and return a DataFrame."""
    suffix = os.path.splitext(filename.lower())[1]
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix or 'unknown'}'. Upload CSV, XLSX, XLS, or Parquet.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        buffer = io.BytesIO(content)
        if suffix == ".csv":
            try:
                df = pd.read_csv(buffer)
            except UnicodeDecodeError:
                buffer.seek(0)
                df = pd.read_csv(buffer, encoding="latin-1")
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(buffer)
        else:
            df = pd.read_parquet(buffer)
    except HTTPException:
        raise
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, ImportError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset. Check that the file is valid: {exc}") from exc

    if df.empty and len(df.columns) == 0:
        raise HTTPException(status_code=400, detail="The dataset has no columns or rows.")
    if len(df.columns) == 0:
        raise HTTPException(status_code=400, detail="The dataset has no columns.")

    # Normalize column labels to strings and make duplicate labels safe for downstream UI use.
    columns: list[str] = []
    seen: dict[str, int] = {}
    for index, name in enumerate(df.columns):
        label = str(name).strip() or f"column_{index + 1}"
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label}_{seen[label]}"
        columns.append(label)
    df.columns = columns
    return df



def analyze_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Build compact, deterministic EDA summaries suitable for browser charts."""
    numeric = df.select_dtypes(include="number").select_dtypes(exclude=["complex"])
    numeric_summaries: list[dict[str, Any]] = []
    distributions: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []
    categorical_distributions: list[dict[str, Any]] = []
    insights: list[dict[str, str]] = []

    for name in numeric.columns:
        series = pd.to_numeric(numeric[name], errors="coerce").dropna()
        if series.empty:
            continue
        q1, median, q3 = (float(series.quantile(q)) for q in (0.25, 0.5, 0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())
        skew = float(series.skew()) if len(series) > 2 else 0.0
        numeric_summaries.append({
            "name": str(name), "count": int(series.count()), "mean": round(float(series.mean()), 6),
            "std": round(float(series.std()) if len(series) > 1 else 0.0, 6),
            "min": round(float(series.min()), 6), "q1": round(q1, 6), "median": round(median, 6),
            "q3": round(q3, 6), "max": round(float(series.max()), 6), "skewness": round(skew, 4),
        })
        # Explicit edges ensure chart labels and counts have matching lengths, including constant columns.
        hist_counts, hist_edges = np.histogram(series.to_numpy(), bins=min(10, max(1, series.nunique())))
        distributions.append({
            "name": str(name),
            "bins": [{"label": f"{hist_edges[i]:.4g}–{hist_edges[i + 1]:.4g}", "count": int(hist_counts[i]),
                      "start": round(float(hist_edges[i]), 6), "end": round(float(hist_edges[i + 1]), 6)}
                     for i in range(len(hist_counts))],
        })
        outliers.append({"name": str(name), "count": outlier_count,
                         "percent": round(outlier_count / len(series) * 100, 2),
                         "lower_bound": round(lower, 6), "upper_bound": round(upper, 6),
                         "method": "IQR (1.5 × IQR)", "sample_values": [round(float(v), 6) for v in series[(series < lower) | (series > upper)].head(5)]})
        if abs(skew) >= 1:
            insights.append({"severity": "info", "title": f"{name} is highly skewed", "detail": f"Skewness is {skew:.2f}. Review its distribution before considering a transformation."})
        if outlier_count:
            insights.append({"severity": "warning", "title": f"Potential outliers in {name}", "detail": f"{outlier_count} values ({outlier_count / len(series) * 100:.1f}%) fall outside the IQR fences. Inspect before removing or capping them."})

    correlation: dict[str, Any] = {"columns": [], "matrix": []}
    corr_columns = list(numeric.columns[:30])
    if len(corr_columns) >= 2:
        corr = numeric[corr_columns].corr(method="pearson").fillna(0)
        correlation = {"columns": [str(c) for c in corr.columns],
                       "matrix": [[round(float(v), 4) for v in row] for row in corr.to_numpy()]}
        for i, left in enumerate(corr.columns):
            for j in range(i + 1, len(corr.columns)):
                raw_value = corr.iloc[i, j]
                value = float(raw_value) if isinstance(raw_value, (int, float, np.integer, np.floating)) else 0.0
                if abs(value) >= 0.8:
                    insights.append({"severity": "warning", "title": f"Strong correlation: {left} and {corr.columns[j]}",
                                     "detail": f"Pearson correlation is {value:.2f}. Review redundancy, but do not remove features based on correlation alone."})

    for name in df.select_dtypes(include=["object", "category", "bool"]).columns[:12]:
        series = df[name].dropna().astype(str)
        if series.empty:
            continue
        top = series.value_counts().head(8)
        categorical_distributions.append({"name": str(name), "total_unique": int(series.nunique()),
                                           "categories": [{"label": str(label), "count": int(count)} for label, count in top.items()]})
        if series.nunique() > max(50, int(len(series) * 0.5)):
            insights.append({"severity": "info", "title": f"High cardinality: {name}", "detail": f"This column has {series.nunique()} distinct values. Check whether it is an identifier or free text before encoding."})

    missing = int(df.isna().sum().sum())
    if missing:
        insights.insert(0, {"severity": "warning", "title": "Missing values need attention", "detail": f"{missing} cells ({missing / max(1, df.size) * 100:.1f}% of the dataset) are missing. Choose imputation based on each column's meaning and data type."})
    duplicate_count = int(df.duplicated().sum()) if len(df) else 0
    if duplicate_count:
        insights.append({"severity": "info", "title": "Duplicate records detected", "detail": f"{duplicate_count} exact duplicate rows found. Verify whether duplicates are valid repeated observations before removing them."})
    if not insights:
        insights.append({"severity": "success", "title": "No immediate data-quality flags", "detail": "No missing cells, IQR outliers, high skew, or strong numeric correlations were flagged by these initial checks. Review charts and domain context too."})

    return {"numeric_summaries": numeric_summaries, "distributions": distributions,
            "outliers": outliers, "correlation": correlation,
            "categorical_distributions": categorical_distributions, "insights": insights,
            "notes": ["IQR outliers are candidates for review, not automatic errors.",
                      "Pearson correlation captures linear relationships and does not imply causation.",
                      "For readability, correlation is limited to the first 30 numeric columns and categorical charts to the first 12 categorical columns."]}


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def profile_dataframe(df: pd.DataFrame, filename: str) -> dict[str, Any]:
    """Create a compact, JSON-safe dataset profile for the upload screen."""
    column_profiles = []
    for name in df.columns:
        series = df[name]
        non_null = int(series.notna().sum())
        column_profiles.append(
            {
                "name": str(name),
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_percent": round(float(series.isna().mean() * 100), 2) if len(df) else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
                "non_null_count": non_null,
            }
        )

    preview = [
        {str(key): _json_safe(value) for key, value in row.items()}
        for row in df.head(PREVIEW_ROWS).to_dict(orient="records")
    ]
    numeric_columns = [str(c) for c in df.select_dtypes(include="number").columns]
    categorical_columns = [str(c) for c in df.select_dtypes(include=["object", "category", "bool"]).columns]
    return {
        "filename": filename,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_row_count": int(df.duplicated().sum()) if len(df) else 0,
        "total_missing_cells": int(df.isna().sum().sum()),
        "numeric_column_count": len(numeric_columns),
        "categorical_column_count": len(categorical_columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "columns": column_profiles,
        "preview": preview,
        "preview_limit": PREVIEW_ROWS,
        "eda": analyze_dataframe(df),
    }


def preprocess_dataframe(
    df: pd.DataFrame,
    missing_strategy: str,
    outlier_strategy: str,
    encoding_strategy: str,
    scaling_strategy: str,
    feature_engineering: str,
    feature_selection: str,
    correlation_threshold: float,
    constant_value: str,
) -> dict[str, Any]:
    """Apply explicit preview transformations to a copy of a dataset."""
    if missing_strategy not in {"none", "auto", "mean", "median", "mode", "constant", "drop_rows"}:
        raise HTTPException(status_code=422, detail="Invalid missing-value strategy.")
    if outlier_strategy not in {"none", "cap", "drop_rows"}:
        raise HTTPException(status_code=422, detail="Invalid outlier strategy.")
    if encoding_strategy not in {"none", "onehot", "label"}:
        raise HTTPException(status_code=422, detail="Invalid encoding strategy.")
    if scaling_strategy not in {"none", "standard", "minmax"}:
        raise HTTPException(status_code=422, detail="Invalid scaling strategy.")
    if feature_engineering not in {"none", "interactions"}:
        raise HTTPException(status_code=422, detail="Invalid feature-engineering strategy.")
    if feature_selection not in {"none", "variance", "correlation"}:
        raise HTTPException(status_code=422, detail="Invalid feature-selection strategy.")
    if not 0.5 <= correlation_threshold <= 0.99:
        raise HTTPException(status_code=422, detail="Correlation threshold must be between 0.50 and 0.99.")

    work = df.copy()
    original_rows, original_columns = work.shape
    actions: list[dict[str, Any]] = []
    dropped_columns: list[str] = []
    missing_before = int(work.isna().sum().sum())

    # Missing values are handled by column type. In auto mode, numeric -> median, categorical -> mode.
    if missing_strategy == "drop_rows":
        before = len(work)
        work = work.dropna().copy()
        actions.append({"step": "Missing values", "detail": f"Removed {before - len(work)} rows containing missing values."})
    elif missing_strategy != "none":
        for col in list(work.columns):
            series = work[col]
            if not series.isna().any():
                continue
            strategy = missing_strategy
            if strategy == "auto":
                strategy = "median" if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_complex_dtype(series) else "mode"
            if strategy in {"mean", "median"}:
                if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_complex_dtype(series):
                    continue
                value = series.mean() if strategy == "mean" else series.median()
                if pd.notna(value):
                    work[col] = series.fillna(value)
            elif strategy == "mode":
                modes = series.mode(dropna=True)
                if not modes.empty:
                    work[col] = series.fillna(modes.iloc[0])
            elif strategy == "constant":
                fill: Any = constant_value
                if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_complex_dtype(series):
                    try:
                        fill = float(constant_value)
                    except ValueError:
                        fill = 0.0
                work[col] = series.fillna(fill)
        remaining = int(work.isna().sum().sum())
        actions.append({"step": "Missing values", "detail": f"Handled {missing_before - remaining} missing cells; {remaining} remain (for example, all-empty columns)."})

    numeric_cols = list(work.select_dtypes(include="number").select_dtypes(exclude=["complex"]).columns)
    if outlier_strategy != "none" and numeric_cols:
        if outlier_strategy == "drop_rows":
            mask = pd.Series(False, index=work.index)
            for col in numeric_cols:
                series = work[col].dropna()
                if series.empty:
                    continue
                q1, q3 = series.quantile(.25), series.quantile(.75)
                iqr = q3 - q1
                if pd.notna(iqr):
                    mask |= (work[col] < q1 - 1.5 * iqr) | (work[col] > q3 + 1.5 * iqr)
            before = len(work)
            work = work.loc[~mask].copy()
            actions.append({"step": "Outliers", "detail": f"Removed {before - len(work)} rows with at least one numeric value outside an IQR fence."})
        else:
            capped = 0
            for col in numeric_cols:
                series = work[col]
                valid = series.dropna()
                if valid.empty:
                    continue
                q1, q3 = valid.quantile(.25), valid.quantile(.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                capped += int(((series < lower) | (series > upper)).sum())
                work[col] = series.clip(lower, upper)
            actions.append({"step": "Outliers", "detail": f"Capped {capped} values to their IQR fences; no rows were removed."})

    # Encode categoricals. One-hot encoding is preferred for nominal categories; label encoding is compact but imposes order.
    categorical_cols = list(work.select_dtypes(include=["object", "category", "bool"]).columns)
    if encoding_strategy == "onehot" and categorical_cols:
        work = pd.get_dummies(work, columns=categorical_cols, dummy_na=False, dtype=int)
        actions.append({"step": "Encoding", "detail": f"One-hot encoded {len(categorical_cols)} categorical columns."})
    elif encoding_strategy == "label" and categorical_cols:
        for col in categorical_cols:
            labels = {value: index for index, value in enumerate(work[col].dropna().astype(str).unique())}
            work[col] = work[col].astype(str).map(labels).where(work[col].notna(), np.nan)
        actions.append({"step": "Encoding", "detail": f"Label encoded {len(categorical_cols)} categorical columns. Codes are arbitrary categories, not meaningful rankings."})

    if feature_engineering == "interactions":
        interaction_cols = list(work.select_dtypes(include="number").select_dtypes(exclude=["complex"]).columns[:5])
        created = 0
        for i, left in enumerate(interaction_cols):
            for right in interaction_cols[i + 1:]:
                new_name = f"{left}__x__{right}"
                if new_name not in work.columns:
                    work[new_name] = work[left] * work[right]
                    created += 1
        actions.append({"step": "Feature engineering", "detail": f"Created {created} pairwise numeric interaction features from up to five numeric columns."})

    numeric_cols = list(work.select_dtypes(include="number").select_dtypes(exclude=["complex"]).columns)
    if scaling_strategy != "none" and numeric_cols:
        for col in numeric_cols:
            series = work[col].astype(float)
            if scaling_strategy == "standard":
                mean, std = series.mean(), series.std(ddof=0)
                work[col] = (series - mean) / (std if pd.notna(std) and std > 0 else 1.0)
            else:
                low, high = series.min(), series.max()
                span = high - low
                work[col] = (series - low) / (span if pd.notna(span) and span > 0 else 1.0)
        actions.append({"step": "Scaling", "detail": f"Applied {scaling_strategy} scaling to {len(numeric_cols)} numeric columns."})

    numeric_cols = list(work.select_dtypes(include="number").select_dtypes(exclude=["complex"]).columns)
    if feature_selection == "variance" and numeric_cols:
        removed = []
        for col in numeric_cols:
            values = work[col].dropna()
            if len(values) and values.nunique() <= 1:
                removed.append(col)
        work = work.drop(columns=removed)
        dropped_columns.extend(map(str, removed))
        actions.append({"step": "Feature selection", "detail": f"Removed {len(removed)} constant numeric columns."})
    elif feature_selection == "correlation" and len(numeric_cols) >= 2:
        corr = work[numeric_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        removed = [str(col) for col in upper.columns if any(upper[col] > correlation_threshold)]
        work = work.drop(columns=removed)
        dropped_columns.extend(removed)
        actions.append({"step": "Feature selection", "detail": f"Removed {len(removed)} feature(s) highly correlated above {correlation_threshold:.2f}; review domain importance before keeping this choice."})

    # Convert preview values to JSON-safe representations and bound payload size.
    preview = [{str(k): _json_safe(v) for k, v in row.items()} for row in work.head(PREVIEW_ROWS).to_dict(orient="records")]
    column_summary = [{"name": str(c), "dtype": str(work[c].dtype), "missing_count": int(work[c].isna().sum())} for c in work.columns]
    return {
        "original_shape": {"rows": int(original_rows), "columns": int(original_columns)},
        "transformed_shape": {"rows": int(len(work)), "columns": int(len(work.columns))},
        "rows_previewed": len(preview), "preview": preview, "columns": column_summary,
        "actions": actions, "dropped_columns": dropped_columns,
        "remaining_missing_cells": int(work.isna().sum().sum()),
        "note": "This is a transformation preview. For model evaluation, fit imputation, encoding, scaling, and feature selection on training data only, then apply the fitted transformations to validation/test data to prevent leakage.",
    }


@app.post("/api/datasets/preprocess")
async def preprocess_dataset(
    file: UploadFile = File(...),
    missing_strategy: str = Form("auto"),
    outlier_strategy: str = Form("none"),
    encoding_strategy: str = Form("none"),
    scaling_strategy: str = Form("none"),
    feature_engineering: str = Form("none"),
    feature_selection: str = Form("none"),
    correlation_threshold: float = Form(0.90),
    constant_value: str = Form("0"),
) -> JSONResponse:
    filename = file.filename or "uploaded_dataset"
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File is too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    df = _read_dataframe(filename, content)
    result = preprocess_dataframe(df, missing_strategy, outlier_strategy, encoding_strategy, scaling_strategy, feature_engineering, feature_selection, correlation_threshold, constant_value)
    return JSONResponse(content=result)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": APP_NAME, "version": "0.3.0"}


@app.post("/api/datasets/inspect")
async def inspect_dataset(file: UploadFile = File(...)) -> JSONResponse:
    filename = file.filename or "uploaded_dataset"
    suffix = os.path.splitext(filename.lower())[1]
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix or 'unknown'}'. Upload CSV, XLSX, XLS, or Parquet.",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    df = _read_dataframe(filename, content)
    return JSONResponse(content=profile_dataframe(df, filename))


@app.post("/api/models/train")
async def train_models(
    file: UploadFile = File(...),
    target: str = Form(...),
    task: str = Form(...),
) -> JSONResponse:
    """Train and compare baseline classification or regression models."""
    if task not in {"classification", "regression"}:
        raise HTTPException(status_code=400, detail="Task must be classification or regression.")

    filename = file.filename or "dataset.csv"
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    df = _read_dataframe(filename, content)
    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target}' was not found.")

    df = df.dropna(subset=[target]).copy()
    if len(df) < 10:
        raise HTTPException(status_code=400, detail="At least 10 rows with a target value are required.")

    y = df[target]
    if task == "regression":
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna() & np.isfinite(y)
        df = df.loc[valid].copy()
        y = y.loc[valid].astype(float)
        if y.nunique() < 2:
            raise HTTPException(status_code=400, detail="Regression requires at least two distinct numeric target values.")
    else:
        if y.nunique() < 2:
            raise HTTPException(status_code=400, detail="Classification requires at least two target classes.")
        if y.nunique() > 30:
            raise HTTPException(status_code=400, detail="This version supports up to 30 target classes.")

    X = df.drop(columns=[target]).copy()
    complex_columns = list(X.select_dtypes(include=["complex"]).columns)
    if complex_columns:
        X = X.drop(columns=complex_columns)

    # Drop columns that cannot be imputed and normalize categorical missing values.
    empty_columns = [column for column in X.columns if X[column].notna().sum() == 0]
    if empty_columns:
        X = X.drop(columns=empty_columns)
    if X.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No usable feature columns remain after removing the target.")

    numeric_columns = list(X.select_dtypes(include="number").select_dtypes(exclude=["complex"]).columns)
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    for column in categorical_columns:
        X[column] = X[column].astype("object")
        X[column] = X[column].where(X[column].notna(), np.nan)

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_columns))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if len(X) < 10:
        raise HTTPException(status_code=400, detail="Not enough usable rows remain after target cleanup.")

    stratify = None
    if task == "classification":
        class_counts = y.value_counts()
        test_count = math.ceil(len(y) * 0.2)
        train_count = len(y) - test_count
        if class_counts.min() >= 2 and test_count >= y.nunique() and train_count >= y.nunique():
            stratify = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unable to create a train/test split: {exc}") from exc

    if task == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        }

    results: list[dict[str, Any]] = []
    fitted_models = []
    for model_name, estimator in models.items():
        pipeline = Pipeline([("preprocessing", clone(preprocessor)), ("model", estimator)])
        try:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            if task == "classification":
                metrics: dict[str, Any] = {
                    "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
                    "precision_weighted": round(float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 4),
                    "recall_weighted": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 4),
                    "f1_weighted": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
                }
                labels = sorted(y.unique().tolist(), key=lambda value: str(value))
                matrix = confusion_matrix(y_test, predictions, labels=labels)
                metrics["confusion_matrix"] = {
                    "labels": [str(label) for label in labels],
                    "matrix": matrix.astype(int).tolist(),
                }
                selection_score = metrics["f1_weighted"]
            else:
                mse = mean_squared_error(y_test, predictions)
                metrics = {
                    "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
                    "rmse": round(float(np.sqrt(mse)), 4),
                    "r2": round(float(r2_score(y_test, predictions)), 4),
                }
                selection_score = metrics["r2"]
            results.append({"model": model_name, "metrics": metrics, "status": "success"})
            fitted_models.append((model_name, pipeline, selection_score))
        except Exception as exc:
            results.append({"model": model_name, "metrics": {}, "status": "failed", "error": str(exc)})

    if not fitted_models:
        raise HTTPException(status_code=400, detail="All candidate models failed. Check your target and feature data.")

    best_name, best_pipeline, _ = max(fitted_models, key=lambda item: item[2])
    model_id = uuid.uuid4().hex
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump({
            "pipeline": best_pipeline,
            "task": task,
            "target": target,
            "feature_columns": list(X.columns),
            "feature_types": {str(column): ("number" if column in numeric_columns else "text") for column in X.columns},
            "best_model": best_name,
            "results": results,
            "rows_used": int(len(X)),
            "feature_count": int(X.shape[1]),
            "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        }, MODEL_DIR / f"{model_id}.joblib")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Models trained, but saving the best model failed: {exc}") from exc

    metric_key = "f1_weighted" if task == "classification" else "r2"
    results.sort(key=lambda item: (
        item["status"] == "success",
        item["metrics"].get(metric_key, float("-inf")),
    ), reverse=True)

    return JSONResponse(content={
        "task": task,
        "target": target,
        "rows_used": int(len(X)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "excluded_complex_columns": [str(column) for column in complex_columns],
        "best_model": best_name,
        "model_id": model_id,
        "selection_metric": metric_key,
        "results": results,
        "note": "Metrics are measured on a held-out test split. This baseline comparison is not a guarantee of future performance. Inspect class balance and validate the model before production use.",
    })


@app.get("/api/models")
def list_saved_models() -> JSONResponse:
    """List model artifacts saved by April AI, including lightweight explainability."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for artifact_path in sorted(MODEL_DIR.glob("*.joblib"), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            artifact = joblib.load(artifact_path)
            pipeline = artifact.get("pipeline")
            estimator = pipeline.named_steps.get("model") if pipeline is not None else None
            raw_importance = getattr(estimator, "feature_importances_", None)
            if raw_importance is None and hasattr(estimator, "coef_"):
                coefficients = np.asarray(estimator.coef_)
                raw_importance = np.mean(np.abs(coefficients), axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
            importance: list[dict[str, Any]] = []
            if raw_importance is not None and pipeline is not None:
                try:
                    names = pipeline.named_steps["preprocessing"].get_feature_names_out()
                    pairs = [(str(name).replace("numeric__", "").replace("categorical__", ""), float(value)) for name, value in zip(names, np.asarray(raw_importance).ravel())]
                    pairs.sort(key=lambda pair: pair[1], reverse=True)
                    total = sum(value for _, value in pairs) or 1.0
                    importance = [{"feature": name, "importance": round(value / total, 6)} for name, value in pairs[:12]]
                except Exception:
                    importance = []
            items.append({
                "model_id": artifact_path.stem,
                "model": artifact.get("best_model", "Saved model"),
                "task": artifact.get("task", "unknown"),
                "target": artifact.get("target", ""),
                "feature_columns": artifact.get("feature_columns", []),
                "feature_types": artifact.get("feature_types", {}),
                "feature_count": artifact.get("feature_count", len(artifact.get("feature_columns", []))),
                "rows_used": artifact.get("rows_used", 0),
                "created_at": artifact.get("created_at", ""),
                "results": artifact.get("results", []),
                "feature_importance": importance,
            })
        except Exception:
            continue
    return JSONResponse(content={"models": items})


@app.post("/api/models/{model_id}/predict")
def predict_with_saved_model(model_id: str, payload: dict[str, Any] = Body(...)) -> JSONResponse:
    """Run one prediction with a previously saved best-model pipeline."""
    if not model_id.isalnum():
        raise HTTPException(status_code=400, detail="Invalid model ID.")
    artifact_path = MODEL_DIR / f"{model_id}.joblib"
    if not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Saved model not found. Train a model first.")
    try:
        artifact = joblib.load(artifact_path)
        features = artifact.get("feature_columns", [])
        missing = [feature for feature in features if feature not in payload]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing feature values: {', '.join(missing)}")
        frame = pd.DataFrame([{feature: payload[feature] for feature in features}], columns=features)
        for feature, kind in artifact.get("feature_types", {}).items():
            if kind == "number":
                frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        pipeline = artifact["pipeline"]
        prediction = pipeline.predict(frame)[0]
        result: dict[str, Any] = {"model_id": model_id, "task": artifact.get("task"), "target": artifact.get("target"), "prediction": str(prediction) if isinstance(prediction, (np.generic,)) else prediction.item() if hasattr(prediction, "item") else prediction}
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(frame)[0]
            classes = pipeline.named_steps["model"].classes_
            result["probabilities"] = [{"class": str(label), "probability": round(float(prob), 6)} for label, prob in zip(classes, probabilities)]
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
