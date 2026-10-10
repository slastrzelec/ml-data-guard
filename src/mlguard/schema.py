"""Schema: what a batch of data is allowed to look like, inferred from reference data.

A schema is a small, versioned JSON document with one entry per feature (type, presence, numeric
range or categorical domain). It contains statistics and bounded domains only, never rows, and a
domain can be stored hashed or left out when the values themselves are sensitive.

Inference is deliberately loose: a schema that rejects the next clean batch is worse than none.
Tightening it is a human decision, supported by the suggestion that every anomaly carries.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

from mlguard.detectors import strict_numeric

FORMAT_VERSION = 1
KINDS = ("numeric", "categorical")
DTYPES = ("int", "float", "string", "bool")
STAGES = ("active", "deprecated")
DOMAIN_STORAGE = ("plain", "hashed", "none")
RANGE_MODES = ("padded", "observed", "none")


def is_missing(value: object) -> bool:
    """``None``, ``pd.NA``, ``NaT`` and float NaN are missing; everything else is a value."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float | np.floating) and math.isnan(value)


def value_type(value: object) -> str:
    """Logical type of one cell: ``bool``, ``int``, ``float``, ``string`` or ``other``."""
    if isinstance(value, bool | np.bool_):
        return "bool"
    if isinstance(value, int | np.integer):
        return "int"
    if isinstance(value, float | np.floating):
        return "float"
    if isinstance(value, str):
        return "string"
    return "other"


def plain_token(value: object) -> str | int | bool:
    """JSON-friendly form of a categorical value; 1 and "1" stay different."""
    kind = value_type(value)
    if kind == "bool":
        return bool(value)
    if kind == "int":
        return int(value)  # type: ignore[call-overload]
    return str(value)


def hashed_token(value: object) -> str:
    """Truncated SHA-256 of a categorical value (type included, so 1 and "1" differ).

    Keeps values out of schema files and reports. It is not a defence against guessing
    low-entropy values; use ``domains="none"`` for personal data.
    """
    token = plain_token(value)
    text = f"{type(token).__name__}:{token}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class FeatureSchema:
    """Expectations for one feature."""

    name: str
    kind: Literal["numeric", "categorical"]
    dtype: Literal["int", "float", "string", "bool"]
    min_presence: float = 1.0
    minimum: float | None = None
    maximum: float | None = None
    domain: tuple[str | int | bool, ...] | None = None
    domain_storage: Literal["plain", "hashed", "none"] = "plain"
    max_out_of_range_fraction: float = 0.0
    max_unexpected_fraction: float = 0.0
    stage: Literal["active", "deprecated"] = "active"

    def __post_init__(self) -> None:
        problems = []
        if not isinstance(self.name, str) or not self.name:
            problems.append("name must be a non-empty string")
        if self.kind not in KINDS:
            problems.append(f"kind must be one of {KINDS}")
        if self.dtype not in DTYPES:
            problems.append(f"dtype must be one of {DTYPES}")
        if self.kind == "numeric" and self.dtype not in ("int", "float"):
            problems.append("a numeric feature must have dtype int or float")
        if self.stage not in STAGES:
            problems.append(f"stage must be one of {STAGES}")
        if self.domain_storage not in DOMAIN_STORAGE:
            problems.append(f"domain_storage must be one of {DOMAIN_STORAGE}")
        for label in ("min_presence", "max_out_of_range_fraction", "max_unexpected_fraction"):
            number = getattr(self, label)
            if not (_is_number(number) and 0 <= number <= 1):
                problems.append(f"{label} must be a number between 0 and 1")
        for label in ("minimum", "maximum"):
            number = getattr(self, label)
            if number is not None and not (_is_number(number) and math.isfinite(number)):
                problems.append(f"{label} must be a finite number or null")
        if _is_number(self.minimum) and _is_number(self.maximum) and self.minimum > self.maximum:
            problems.append("minimum must not exceed maximum")
        if self.kind == "categorical" and (self.minimum is not None or self.maximum is not None):
            problems.append("a categorical feature has no numeric range")
        if self.kind == "numeric" and self.domain is not None:
            problems.append("a numeric feature has no domain")
        if self.domain is not None:
            if self.domain_storage == "none":
                problems.append("domain_storage 'none' cannot be combined with a domain")
            if len(set(self.domain)) != len(self.domain) or len(self.domain) == 0:
                problems.append("domain must be a non-empty list without duplicates")
        if problems:
            raise ValueError(f"invalid schema for feature {self.name!r}: " + "; ".join(problems))

    def allows(self, value: object) -> bool:
        """Whether a non-missing categorical value is in the domain (always true without one)."""
        if self.domain is None:
            return True
        token = hashed_token(value) if self.domain_storage == "hashed" else plain_token(value)
        return token in self._domain_set

    @property
    def _domain_set(self) -> frozenset[tuple[str, object]]:
        cached = self.__dict__.get("_cached_domain")
        if cached is None:
            # type and value together: True, 1 and "1" are three different members
            cached = _TypedSet(self.domain or ())
            object.__setattr__(self, "_cached_domain", cached)
        return cached  # type: ignore[return-value]


class _TypedSet:
    """Membership that does not confuse ``True``, ``1`` and ``"1"``."""

    def __init__(self, items: Iterable[object]) -> None:
        self._items = frozenset((type(item).__name__, item) for item in items)

    def __contains__(self, item: object) -> bool:
        return (type(item).__name__, item) in self._items


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


@dataclass(frozen=True)
class Schema:
    """A versioned set of feature expectations."""

    features: tuple[FeatureSchema, ...]
    version: int = 1
    #: Columns that are known but are not features (keys, targets, timestamps): never validated.
    ignored: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = [feature.name for feature in self.features]
        if len(set(names)) != len(names):
            raise ValueError("feature names in a schema must be unique")
        if set(names) & set(self.ignored) or len(set(self.ignored)) != len(self.ignored):
            raise ValueError("ignored columns must be unique and must not be features")
        if not all(isinstance(name, str) for name in self.ignored):
            raise ValueError("ignored columns must be names")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("schema version must be a positive integer")

    def __getitem__(self, name: str) -> FeatureSchema:
        for feature in self.features:
            if feature.name == name:
                return feature
        raise KeyError(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    @property
    def active(self) -> tuple[FeatureSchema, ...]:
        return tuple(feature for feature in self.features if feature.stage == "active")

    def with_feature(self, feature: FeatureSchema) -> Schema:
        """Copy with one feature replaced and the version increased by one."""
        if feature.name not in self.names:
            raise KeyError(feature.name)
        updated = tuple(feature if item.name == feature.name else item for item in self.features)
        return replace(self, features=updated, version=self.version + 1)

    def to_dict(self) -> dict[str, Any]:
        entries = []
        for feature in self.features:
            entry = asdict(feature)
            entry["domain"] = None if feature.domain is None else list(feature.domain)
            entries.append(entry)
        return {
            "format_version": FORMAT_VERSION,
            "version": self.version,
            "ignored": list(self.ignored),
            "features": entries,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Schema:
        keys = {"format_version", "version", "ignored", "features"}
        if not isinstance(raw, dict) or set(raw) != keys:
            raise ValueError(f"a schema needs exactly the keys {sorted(keys)}")
        if raw["format_version"] != FORMAT_VERSION:
            raise ValueError(f"unsupported schema format_version {raw['format_version']!r}")
        if not isinstance(raw["features"], list) or not isinstance(raw["ignored"], list):
            raise ValueError("features and ignored must be lists")
        known = {item.name for item in fields(FeatureSchema)}
        features = []
        for entry in raw["features"]:
            if not isinstance(entry, dict) or not set(entry) <= known:
                raise ValueError(f"unknown keys in a feature entry: {entry!r:.80}")
            values = dict(entry)
            if isinstance(values.get("domain"), list):
                values["domain"] = tuple(values["domain"])
            try:
                features.append(FeatureSchema(**values))
            except TypeError as error:
                raise ValueError(f"incomplete feature entry: {error}") from error
        return cls(features=tuple(features), version=raw["version"], ignored=tuple(raw["ignored"]))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> Schema:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} is not valid JSON: {error}") from error
        return cls.from_dict(raw)


def presence_floor(n_present: int, n_rows: int) -> float:
    """Smallest allowed share of non-missing values, given what the reference showed.

    1.0 when nothing was missing. Otherwise twice the missing share of the reference plus one
    percentage point is tolerated (rounded down to three decimals): a later batch is usually much
    smaller than the reference, and its missing share scatters accordingly.
    """
    if n_present == n_rows:
        return 1.0
    missing = 1.0 - n_present / n_rows
    return max(0.0, math.floor((1.0 - 2.0 * missing - 0.01) * 1000.0 + 1e-9) / 1000.0)


def infer_schema(
    reference: pd.DataFrame,
    *,
    categorical: Iterable[str] = (),
    exclude: Iterable[str] = (),
    numeric_ranges: Literal["padded", "observed", "none"] = "padded",
    range_padding: float = 0.1,
    max_domain_size: int = 100,
    domains: Literal["plain", "hashed", "none"] = "plain",
    version: int = 1,
) -> Schema:
    """Infer a schema from reference data.

    ``categorical`` names numeric columns that are really codes. ``exclude`` names columns that
    are not features (keys, targets, timestamps). The heuristics per field are documented on
    ``FeatureSchema`` and in SPEC section 14.1.
    """
    if not isinstance(reference, pd.DataFrame) or len(reference) == 0:
        raise ValueError("reference must be a non-empty pandas DataFrame")
    if numeric_ranges not in RANGE_MODES:
        raise ValueError(f"numeric_ranges must be one of {RANGE_MODES}")
    if domains not in DOMAIN_STORAGE:
        raise ValueError(f"domains must be one of {DOMAIN_STORAGE}")
    if not (math.isfinite(range_padding) and range_padding >= 0):
        raise ValueError("range_padding must be a non-negative number")
    if max_domain_size < 1:
        raise ValueError("max_domain_size must be at least 1")
    declared, excluded = set(categorical), set(exclude)
    unknown = (declared | excluded) - {str(column) for column in reference.columns}
    if unknown:
        raise ValueError(f"columns not in the reference data: {sorted(unknown)}")

    features = []
    for column in reference.columns:
        name = str(column)
        if name in excluded:
            continue
        series = reference[column]
        values = series.to_numpy(dtype=object)
        present = np.array([not is_missing(value) for value in values], dtype=bool)
        if not present.any():
            raise ValueError(f"feature {name!r} has no value in the reference data")
        min_presence = presence_floor(int(present.sum()), len(values))
        numeric = is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype)
        if numeric and name not in declared:
            numbers = strict_numeric(series.to_numpy())
            low, high = float(np.nanmin(numbers)), float(np.nanmax(numbers))
            if numeric_ranges == "none":
                minimum = maximum = None
            else:
                padding = range_padding * (high - low) if numeric_ranges == "padded" else 0.0
                # Padding never crosses zero: a feature that was never negative stays non-negative.
                minimum = max(0.0, low - padding) if low >= 0 else low - padding
                maximum = min(0.0, high + padding) if high <= 0 else high + padding
            features.append(
                FeatureSchema(
                    name=name,
                    kind="numeric",
                    dtype="int" if is_integer_dtype(series.dtype) else "float",
                    min_presence=min_presence,
                    minimum=minimum,
                    maximum=maximum,
                    domain_storage="none",
                )
            )
            continue
        types = {value_type(value) for value in values[present]}
        if len(types) != 1 or types == {"other"} or types == {"float"}:
            raise ValueError(
                f"categorical feature {name!r} must hold one type of value "
                f"(text, integer or boolean), found {sorted(types)}"
            )
        distinct = sorted({plain_token(value) for value in values[present]}, key=repr)
        domain: tuple[str | int | bool, ...] | None = None
        storage = domains
        if domains == "none" or len(distinct) > max_domain_size:
            storage = "none"
        elif domains == "hashed":
            domain = tuple(sorted(hashed_token(value) for value in distinct))
        else:
            domain = tuple(distinct)
        features.append(
            FeatureSchema(
                name=name,
                kind="categorical",
                dtype=types.pop(),  # type: ignore[arg-type]
                min_presence=min_presence,
                domain=domain,
                domain_storage=storage,
            )
        )
    ignored = tuple(str(column) for column in reference.columns if str(column) in excluded)
    return Schema(features=tuple(features), version=version, ignored=ignored)
