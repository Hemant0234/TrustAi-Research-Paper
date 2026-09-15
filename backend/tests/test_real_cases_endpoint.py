from app.schemas.cases import CaseSummary
from app import main


def test_list_cases_prefers_real_cases(monkeypatch):
    fake_case = CaseSummary(
        case_id="REAL-001",
        modality="Chest X-Ray",
        dataset="NIH ChestX-ray14",
        model_name="DenseNet-121 (NIH Real)",
        predicted_label="Pneumonia",
        confidence=91.2,
        uncertainty_level="low",
        uncertainty_score=0.12,
        xqi_score=89.0,
        reliability_score=91.0,
        reliability_level="RELIABLE",
        overall_agreement=88.5,
        is_demo=False,
    )

    monkeypatch.setattr(main, "get_real_case_summaries", lambda limit=30: [fake_case])

    result = main.list_cases()

    assert result == [fake_case]


def test_get_case_uses_real_case_detail_when_missing_from_synthetic(monkeypatch):
    fake_detail = {"case_id": "00000003_002", "dataset": "NIH ChestX-ray14"}

    monkeypatch.setattr(main.SyntheticCaseLibrary, "get_all_cases", lambda: {})
    monkeypatch.setattr(main, "get_real_case_detail", lambda case_id: fake_detail if case_id == "00000003_002" else None)

    result = main.get_case("00000003_002")

    assert result == fake_detail
