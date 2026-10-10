"""Schema inference, the heuristics that keep it loose, and the file format."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlguard.datasets import make_synthetic
from mlguard.schema import (
    FeatureSchema,
    Schema,
    hashed_token,
    infer_schema,
    is_missing,
    plain_token,
    presence_floor,
    value_type,
)
from mlguard.validate import validate_batch


def reference_frame(n_rows: int = 2_000, seed: int = 0) -> pd.DataFrame:
    return make_synthetic(n_rows, seed).frame


def test_missing_values_and_value_types_are_recognised() -> None:
    cells = (None, math.nan, np.nan, pd.NA, pd.NaT, 0, "", "nan")
    assert [is_missing(value) for value in cells] == [True] * 5 + [False] * 3
    values = (True, np.bool_(False), 1, np.int64(2), 1.5, np.float32(2.5), "a", b"a", [1])
    assert [value_type(value) for value in values] == [
        "bool", "bool", "int", "int", "float", "float", "string", "other", "other",
    ]  # fmt: skip


def test_tokens_keep_types_apart() -> None:
    assert plain_token(np.int64(1)) == 1
    assert type(plain_token(np.int64(1))) is int
    assert plain_token(True) is True
    assert plain_token("1") == "1"
    assert len({hashed_token(1), hashed_token("1"), hashed_token(True)}) == 3
    assert hashed_token("a") == hashed_token("a")
    assert len(hashed_token("a")) == 16


def test_inference_reads_kind_type_presence_range_and_domain() -> None:
    schema = infer_schema(reference_frame(), exclude=["target"])
    assert schema.names == (
        "x_normal", "x_lognormal", "x_zero_inflated", "x_count", "x_sentinel", "c_small", "c_large",
    )  # fmt: skip
    assert schema.ignored == ("target",)
    assert [feature.kind for feature in schema.features] == ["numeric"] * 5 + ["categorical"] * 2
    assert schema["x_count"].dtype == "int"
    assert schema["x_normal"].dtype == "float"
    assert schema["c_small"].dtype == "string"
    assert schema["c_small"].domain == ("a", "b", "c", "d", "e")
    assert len(schema["c_large"].domain) == 30
    assert schema["x_lognormal"].min_presence == 1.0
    assert 0.93 < schema["x_normal"].min_presence < 0.96  # about 2% are missing in the reference


def test_presence_floor_leaves_room_for_sampling_noise() -> None:
    assert presence_floor(1_000, 1_000) == 1.0
    assert presence_floor(980, 1_000) == pytest.approx(0.95)  # 2% missing -> 5% tolerated
    assert presence_floor(999, 1_000) == pytest.approx(0.988)
    assert presence_floor(500, 1_000) == 0.0
    assert presence_floor(1, 1_000) == 0.0


def test_padded_ranges_are_wider_than_the_data_but_never_cross_zero() -> None:
    frame = pd.DataFrame(
        {
            "positive": [10.0, 20.0],
            "negative": [-20.0, -10.0],
            "mixed": [-10.0, 10.0],
            "near": [0.5, 10.5],
        }
    )
    schema = infer_schema(frame)
    assert (schema["positive"].minimum, schema["positive"].maximum) == (9.0, 21.0)
    assert (schema["negative"].minimum, schema["negative"].maximum) == (-21.0, -9.0)
    assert (schema["mixed"].minimum, schema["mixed"].maximum) == (-12.0, 12.0)
    assert (schema["near"].minimum, schema["near"].maximum) == (0.0, 11.5)
    observed = infer_schema(frame, numeric_ranges="observed")
    assert (observed["near"].minimum, observed["near"].maximum) == (0.5, 10.5)
    unbounded = infer_schema(frame, numeric_ranges="none")
    assert (unbounded["near"].minimum, unbounded["near"].maximum) == (None, None)


def test_integer_codes_are_categorical_only_when_declared() -> None:
    frame = pd.DataFrame({"code": [1, 2, 3, 1], "flag": [True, False, True, True]})
    assert infer_schema(frame)["code"].kind == "numeric"
    declared = infer_schema(frame, categorical=["code"])["code"]
    assert (declared.kind, declared.dtype, declared.domain) == ("categorical", "int", (1, 2, 3))
    flag = infer_schema(frame)["flag"]
    assert (flag.kind, flag.dtype, flag.domain) == ("categorical", "bool", (False, True))


def test_large_domains_are_not_stored() -> None:
    frame = pd.DataFrame({"free_text": [f"user {index}" for index in range(101)]})
    feature = infer_schema(frame)["free_text"]
    assert feature.domain is None
    assert feature.domain_storage == "none"
    assert len(infer_schema(frame, max_domain_size=101)["free_text"].domain) == 101


@pytest.mark.parametrize("domains", ["hashed", "none"])
def test_sensitive_category_values_never_reach_the_schema_file(domains: str) -> None:
    """SPEC section 3: a schema can be shared without leaking category values."""
    secrets = ["alice@example.org", "bob@example.org", "carol@example.org"]
    frame = pd.DataFrame({"email": secrets * 10, "amount": np.arange(30.0)})
    schema = infer_schema(frame, domains=domains)
    text = schema.to_json()
    assert not any(secret in text for secret in secrets)
    assert "example.org" not in text
    if domains == "hashed":
        assert len(schema["email"].domain) == 3
        assert validate_batch(frame, schema) == []
        newcomer = frame.assign(email=["mallory@example.org"] * 30)
        report = validate_batch(newcomer, schema)
        assert [anomaly.type.value for anomaly in report] == ["UNEXPECTED_VALUES"]
        assert "mallory" not in json.dumps([anomaly.to_dict() for anomaly in report])
    else:
        assert schema["email"].domain is None


def test_schema_holds_statistics_only_and_no_rows() -> None:
    frame = reference_frame(5_000)
    text = infer_schema(frame, exclude=["target"]).to_json()
    assert len(text) < 5_000  # 5,000 rows in, a few kilobytes out
    for value in frame["x_normal"].dropna().sample(200, random_state=0):
        assert repr(float(value)) not in text


@settings(max_examples=30)
@given(st.integers(0, 2**32 - 1), st.sampled_from(["padded", "observed"]))
def test_inferred_schema_accepts_its_own_reference(seed: int, ranges: str) -> None:
    frame = reference_frame(400, seed)
    schema = infer_schema(frame, exclude=["target"], numeric_ranges=ranges)
    assert validate_batch(frame, schema) == []


def test_presence_margin_accepts_fresh_samples_of_the_same_data() -> None:
    schema = infer_schema(reference_frame(5_000, 0), exclude=["target"])
    low_presence = 0
    for seed in range(1, 41):
        report = validate_batch(reference_frame(1_000, seed), schema)
        low_presence += sum(anomaly.type.value == "LOW_PRESENCE" for anomaly in report)
    assert low_presence == 0


def test_json_round_trip_and_file_round_trip(tmp_path: Path) -> None:
    schema = infer_schema(reference_frame(), exclude=["target"], categorical=["x_count"])
    assert Schema.from_dict(json.loads(schema.to_json())) == schema
    path = tmp_path / "schema.json"
    schema.save(path)
    assert Schema.load(path) == schema
    assert path.read_text(encoding="utf-8").endswith("\n")
    hashed = infer_schema(reference_frame(), exclude=["target"], domains="hashed")
    assert Schema.from_dict(json.loads(hashed.to_json())) == hashed


def test_a_changed_feature_gives_a_new_version() -> None:
    schema = infer_schema(reference_frame(), exclude=["target"])
    relaxed = FeatureSchema(name="x_count", kind="numeric", dtype="int", minimum=0, maximum=100)
    updated = schema.with_feature(relaxed)
    assert updated.version == schema.version + 1
    assert updated["x_count"].maximum == 100
    assert schema["x_count"].maximum != 100  # the original is unchanged
    with pytest.raises(KeyError):
        schema.with_feature(FeatureSchema(name="unknown", kind="numeric", dtype="int"))
    with pytest.raises(KeyError):
        schema["unknown"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"name": ""}, "name"),
        ({"kind": "ordinal"}, "kind"),
        ({"dtype": "text"}, "dtype"),
        ({"dtype": "string"}, "numeric feature must have dtype"),
        ({"stage": "retired"}, "stage"),
        ({"min_presence": 1.5}, "min_presence"),
        ({"max_out_of_range_fraction": -0.1}, "max_out_of_range_fraction"),
        ({"minimum": math.inf}, "minimum"),
        ({"minimum": 5, "maximum": 1}, "must not exceed"),
        ({"domain": ("a",)}, "numeric feature has no domain"),
        ({"domain_storage": "encrypted"}, "domain_storage"),
    ],
)
def test_invalid_feature_entries_are_rejected(change: dict, message: str) -> None:
    values = {"name": "f", "kind": "numeric", "dtype": "float", **change}
    with pytest.raises(ValueError, match=message):
        FeatureSchema(**values)


def test_invalid_categorical_entries_are_rejected() -> None:
    base = {"name": "f", "kind": "categorical", "dtype": "string"}
    with pytest.raises(ValueError, match="no numeric range"):
        FeatureSchema(**base, minimum=0)
    with pytest.raises(ValueError, match="without duplicates"):
        FeatureSchema(**base, domain=("a", "a"))
    with pytest.raises(ValueError, match="without duplicates"):
        FeatureSchema(**base, domain=())
    with pytest.raises(ValueError, match="cannot be combined"):
        FeatureSchema(**base, domain=("a",), domain_storage="none")


def test_invalid_schemas_and_files_are_rejected(tmp_path: Path) -> None:
    feature = FeatureSchema(name="f", kind="numeric", dtype="float")
    with pytest.raises(ValueError, match="unique"):
        Schema(features=(feature, feature))
    with pytest.raises(ValueError, match="version"):
        Schema(features=(feature,), version=0)
    with pytest.raises(ValueError, match="ignored"):
        Schema(features=(feature,), ignored=("f",))
    good = Schema(features=(feature,)).to_dict()
    for broken, message in [
        ({**good, "extra": 1}, "exactly the keys"),
        ({**good, "format_version": 99}, "format_version"),
        ({**good, "features": "all"}, "lists"),
        ({**good, "features": [{"name": "f", "colour": "red"}]}, "unknown keys"),
        ({**good, "features": [{"name": "f"}]}, "incomplete"),
        ([], "exactly the keys"),
    ]:
        with pytest.raises(ValueError, match=message):
            Schema.from_dict(broken)
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        Schema.load(path)


def test_invalid_inference_input_is_rejected() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0], "empty": [None, None], "mixed": ["a", 1]})
    with pytest.raises(ValueError, match="non-empty"):
        infer_schema(frame.iloc[:0])
    with pytest.raises(ValueError, match="no value"):
        infer_schema(frame[["x", "empty"]])
    with pytest.raises(ValueError, match="one type of value"):
        infer_schema(frame[["x", "mixed"]])
    with pytest.raises(ValueError, match="one type of value"):
        infer_schema(frame[["x"]], categorical=["x"])  # float codes are not categories
    with pytest.raises(ValueError, match="not in the reference"):
        infer_schema(frame[["x"]], exclude=["y"])
    for options, message in [
        ({"numeric_ranges": "tight"}, "numeric_ranges"),
        ({"domains": "secret"}, "domains"),
        ({"range_padding": -1.0}, "range_padding"),
        ({"max_domain_size": 0}, "max_domain_size"),
    ]:
        with pytest.raises(ValueError, match=message):
            infer_schema(frame[["x"]], **options)
