from __future__ import annotations

from models.reasoning.nemotron_wrapper import _build_chat_inputs, normalize_response_text
from models.reasoning.prompts import SYSTEM_PROMPT, build_messages, build_user_prompt
from pipeline import diagnoser


def test_build_messages_starts_with_system_and_ends_with_user() -> None:
    messages = build_messages(
        film_type="Kodak Portra 400",
        film_age_years=1,
        storage="fridge, sealed",
        scan_resolution_dpi=4000,
        defect_summary={"dust": 2},
        total_defects=2,
        spatial_evidence={"edge_defects": 1},
    )
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[-1]["role"] == "user"
    assert "spatial_evidence" in messages[-1]["content"]
    assert '"metadata_confidence": "low"' in messages[-1]["content"]
    assert [m["role"] for m in messages].count("assistant") == 4


def test_build_user_prompt_is_json_backed() -> None:
    prompt = build_user_prompt(
        "Ilford HP5",
        12,
        "attic",
        3200,
        {"scratch": 1},
        1,
        spatial_evidence={"largest_labels": ["scratch"]},
        metadata_confidence="medium",
    )
    assert '"film_type": "Ilford HP5"' in prompt
    assert '"largest_labels": [' in prompt
    assert '"metadata_confidence": "medium"' in prompt


def test_normalize_response_text_converts_literal_newlines() -> None:
    text = "## Root cause\\nDusty carrier\\n## Fixes"
    assert normalize_response_text(text) == "## Root cause\nDusty carrier\n## Fixes"


def test_build_chat_inputs_accepts_tensor_and_dict_returns() -> None:
    import torch

    class TensorTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            if kwargs.get("return_dict"):
                raise TypeError("old tokenizer")
            return torch.tensor([[1, 2, 3]])

    class Batch:
        def __init__(self):
            self.data = {"input_ids": torch.tensor([[4, 5]])}

        def __contains__(self, key):
            return key in self.data

        def __getitem__(self, key):
            return self.data[key]

        def __iter__(self):
            return iter(self.data)

        def keys(self):
            return self.data.keys()

        def to(self, device):
            return self

    class DictTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return Batch()

    tensor_inputs, tensor_len = _build_chat_inputs(
        TensorTokenizer(),
        [{"role": "user", "content": "hi"}],
        "cpu",
    )
    dict_inputs, dict_len = _build_chat_inputs(
        DictTokenizer(),
        [{"role": "user", "content": "hi"}],
        "cpu",
    )

    assert set(tensor_inputs) == {"input_ids"}
    assert tensor_len == 3
    assert set(dict_inputs) == {"input_ids"}
    assert dict_len == 2


def test_diagnose_uses_reasoner_stub(monkeypatch) -> None:
    captured = {}

    class StubReasoner:
        model_path = "stub-reasoner"

        def generate(self, messages):
            captured["messages"] = messages
            return "## Root cause\nScanner-side dust."

    monkeypatch.setattr(diagnoser, "get_reasoner", lambda: StubReasoner())
    result = diagnoser.diagnose(
        {
            "defects": [{"label": "dust", "bbox": [0.0, 0.0, 0.1, 0.1]}],
            "defect_count": 1,
            "label_counts": {"dust": 1},
        },
        film_type="Kodak Gold 200",
        film_age_years=3,
        storage="room temp, sealed",
        scan_resolution_dpi=4000,
        metadata_confidence="high",
    )
    assert "Scanner-side dust" in result["diagnosis_text"]
    assert result["model_path"] == "stub-reasoner"
    assert result["input_defect_summary"]["metadata_confidence"] == "high"
    assert "spatial_evidence" in captured["messages"][-1]["content"]


def test_diagnose_skips_reasoner_when_no_validated_defects(monkeypatch) -> None:
    def fail_get_reasoner():
        raise AssertionError("reasoner should not load for zero-defect cases")

    monkeypatch.setattr(diagnoser, "get_reasoner", fail_get_reasoner)
    result = diagnoser.diagnose(
        {"defects": [], "defect_count": 0, "label_counts": {}},
        film_type="CineStill 800T (35mm)",
        film_age_years=0,
        storage="unknown",
        scan_resolution_dpi=4000,
        metadata_confidence="medium",
    )
    assert result["skipped_reasoning"] == "no_validated_defects"
    assert "No physical root cause is diagnosed" in result["diagnosis_text"]
    assert "User metadata is not enough" in result["diagnosis_text"]
