from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_inspect_csv_returns_profile_and_preview():
    csv = b"age,city,score\n21,Hyderabad,8.5\n22,Pune,7.0\n22,Pune,7.0\n"
    response = client.post(
        "/api/datasets/inspect",
        files={"file": ("students.csv", csv, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["row_count"] == 3
    assert data["column_count"] == 3
    assert data["duplicate_row_count"] == 1
    assert data["numeric_column_count"] == 2
    assert data["categorical_column_count"] == 1
    assert len(data["preview"]) == 3
    assert data["preview"][0]["city"] == "Hyderabad"


def test_unsupported_file_type_is_rejected():
    response = client.post(
        "/api/datasets/inspect",
        files={"file": ("script.exe", b"not a dataset", "application/octet-stream")},
    )
    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_empty_file_is_rejected():
    response = client.post(
        "/api/datasets/inspect",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_csv_with_missing_values_reports_missing_cells():
    csv = b"name,age\nAsha,20\nRavi,\nMeera,25\n"
    response = client.post(
        "/api/datasets/inspect",
        files={"file": ("missing.csv", csv, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_missing_cells"] == 1
    age = next(column for column in data["columns"] if column["name"] == "age")
    assert age["missing_count"] == 1
    assert age["missing_percent"] == 33.33


def test_eda_includes_numeric_distribution_outliers_and_correlation():
    csv = b"height,weight,group\n160,55,A\n165,60,A\n170,65,B\n175,70,B\n180,250,C\n"
    response = client.post(
        "/api/datasets/inspect",
        files={"file": ("people.csv", csv, "text/csv")},
    )
    assert response.status_code == 200
    eda = response.json()["eda"]
    assert len(eda["numeric_summaries"]) == 2
    assert len(eda["distributions"]) == 2
    assert len(eda["distributions"][0]["bins"]) > 0
    assert any(item["count"] > 0 for item in eda["outliers"])
    assert eda["correlation"]["columns"] == ["height", "weight"]
    assert len(eda["correlation"]["matrix"]) == 2
    assert len(eda["categorical_distributions"]) == 1
    assert len(eda["insights"]) > 0


def test_eda_handles_dataset_without_numeric_features():
    csv = b"city,team\nHyderabad,blue\nPune,green\n"
    response = client.post(
        "/api/datasets/inspect",
        files={"file": ("labels.csv", csv, "text/csv")},
    )
    assert response.status_code == 200
    eda = response.json()["eda"]
    assert eda["numeric_summaries"] == []
    assert eda["distributions"] == []
    assert eda["correlation"] == {"columns": [], "matrix": []}
    assert len(eda["categorical_distributions"]) == 2


def test_preprocess_endpoint_handles_missing_values_and_scaling():
    csv = b"age,city,score\n20,Hyderabad,10\n,Hyderabad,20\n40,Pune,30\n"
    response = client.post(
        "/api/datasets/preprocess",
        files={"file": ("students.csv", csv, "text/csv")},
        data={"missing_strategy": "auto", "scaling_strategy": "standard", "encoding_strategy": "onehot"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["original_shape"] == {"rows": 3, "columns": 3}
    assert result["transformed_shape"]["columns"] >= 3
    assert result["remaining_missing_cells"] == 0
    assert any(action["step"] == "Missing values" for action in result["actions"])
    assert any(action["step"] == "Encoding" for action in result["actions"])
    assert "training data only" in result["note"]


def test_preprocess_can_cap_outliers_and_remove_constant_features():
    csv = b"x,constant\n1,5\n2,5\n3,5\n100,5\n"
    response = client.post(
        "/api/datasets/preprocess",
        files={"file": ("values.csv", csv, "text/csv")},
        data={"missing_strategy": "none", "outlier_strategy": "cap", "feature_selection": "variance"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert "constant" in result["dropped_columns"]
    assert result["transformed_shape"]["columns"] == 1
    assert any(action["step"] == "Outliers" for action in result["actions"])


def test_preprocess_can_create_numeric_interactions():
    csv = b"height,weight\n2,3\n4,5\n6,7\n"
    response = client.post(
        "/api/datasets/preprocess",
        files={"file": ("measurements.csv", csv, "text/csv")},
        data={"missing_strategy": "none", "feature_engineering": "interactions"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert "height__x__weight" in [column["name"] for column in result["columns"]]
    assert any(action["step"] == "Feature engineering" for action in result["actions"])
