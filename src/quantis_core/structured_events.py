"""Training-fitted structured event tokens from OTLP application logs."""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
from numpy.typing import NDArray

from .otlp_logs import LogRecord, OtlpLogCapture


UNKNOWN_TEMPLATE_ID = 0
_UUID_PATTERN = re.compile(
    r"(?<![0-9a-f])"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}"
    r"(?![0-9a-f])",
    re.IGNORECASE,
)
_HEX_PATTERN = re.compile(
    r"(?<![0-9a-z])"
    r"(?:0x[0-9a-f]+|"
    r"(?=[0-9a-f]{8,}(?![0-9a-z]))"
    r"(?=[0-9a-f]*[a-f])[0-9a-f]{8,})"
    r"(?![0-9a-z])",
    re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(
    r"(?<!\w)[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:e[-+]?\d+)?(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CompiledStructuredEvents:
    """Typed event arrays produced by one frozen vocabulary."""

    template_ids: NDArray[np.int64]
    service_namespaces: Tuple[str, ...]
    service_names: Tuple[str, ...]
    service_instance_ids: Tuple[str, ...]
    severity_numbers: NDArray[np.int64]
    severity_texts: Tuple[str, ...]
    trace_ids: Tuple[str, ...]
    span_ids: Tuple[str, ...]
    event_time_unix_nano: NDArray[np.int64]
    delta_seconds: NDArray[np.float64]
    numeric_attribute_names: Tuple[str, ...]
    numeric_values: NDArray[np.float64]
    numeric_mask: NDArray[np.bool_]
    vocabulary_schema_id: str
    capture_sha256: str

    def __post_init__(self) -> None:
        _require_array(
            self.template_ids,
            "template_ids",
            np.dtype(np.int64),
            1,
        )
        _require_array(
            self.severity_numbers,
            "severity_numbers",
            np.dtype(np.int64),
            1,
        )
        _require_array(
            self.event_time_unix_nano,
            "event_time_unix_nano",
            np.dtype(np.int64),
            1,
        )
        _require_array(
            self.delta_seconds,
            "delta_seconds",
            np.dtype(np.float64),
            1,
        )
        _require_array(
            self.numeric_values,
            "numeric_values",
            np.dtype(np.float64),
            2,
        )
        _require_array(
            self.numeric_mask,
            "numeric_mask",
            np.dtype(np.bool_),
            2,
        )
        event_count = len(self.template_ids)
        metadata = (
            self.service_namespaces,
            self.service_names,
            self.service_instance_ids,
            self.severity_texts,
            self.trace_ids,
            self.span_ids,
        )
        if any(len(values) != event_count for values in metadata):
            raise ValueError(
                "structured event metadata does not align"
            )
        if any(
            not isinstance(value, str)
            for values in metadata
            for value in values
        ):
            raise ValueError(
                "structured event metadata must contain strings"
            )
        if any(
            len(array) != event_count
            for array in (
                self.severity_numbers,
                self.event_time_unix_nano,
                self.delta_seconds,
            )
        ):
            raise ValueError("structured event arrays do not align")
        parameter_count = len(self.numeric_attribute_names)
        expected_numeric_shape = (event_count, parameter_count)
        if (
            self.numeric_values.shape != expected_numeric_shape
            or self.numeric_mask.shape != expected_numeric_shape
        ):
            raise ValueError(
                "structured event numeric arrays do not align"
            )
        _validate_numeric_attribute_names(
            self.numeric_attribute_names
        )
        if np.any(self.template_ids < UNKNOWN_TEMPLATE_ID):
            raise ValueError(
                "structured event template ids cannot be negative"
            )
        if np.any(
            (self.severity_numbers < 0)
            | (self.severity_numbers > 24)
        ):
            raise ValueError(
                "structured event severity numbers must be in [0, 24]"
            )
        if np.any(self.event_time_unix_nano < 0):
            raise ValueError(
                "structured event times cannot be negative"
            )
        if (
            not np.all(np.isfinite(self.delta_seconds))
            or not np.all(np.isfinite(self.numeric_values))
        ):
            raise ValueError(
                "structured event numeric values must be finite"
            )
        if np.any(self.delta_seconds < 0.0):
            raise ValueError(
                "structured event deltas cannot be negative"
            )
        if np.any(self.numeric_values[~self.numeric_mask] != 0.0):
            raise ValueError(
                "unobserved numeric attributes must be zero"
            )
        if not _is_sha256(self.vocabulary_schema_id):
            raise ValueError(
                "vocabulary_schema_id must be a SHA-256 digest"
            )
        if not _is_sha256(self.capture_sha256):
            raise ValueError(
                "capture_sha256 must be a SHA-256 digest"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "compiled_structured_events",
            "template_ids": self.template_ids.tolist(),
            "service_namespaces": list(self.service_namespaces),
            "service_names": list(self.service_names),
            "service_instance_ids": list(
                self.service_instance_ids
            ),
            "severity_numbers": self.severity_numbers.tolist(),
            "severity_texts": list(self.severity_texts),
            "trace_ids": list(self.trace_ids),
            "span_ids": list(self.span_ids),
            "event_time_unix_nano": (
                self.event_time_unix_nano.tolist()
            ),
            "delta_seconds": self.delta_seconds.tolist(),
            "numeric_attribute_names": list(
                self.numeric_attribute_names
            ),
            "numeric_values": self.numeric_values.tolist(),
            "numeric_mask": self.numeric_mask.tolist(),
            "vocabulary_schema_id": self.vocabulary_schema_id,
            "capture_sha256": self.capture_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "CompiledStructuredEvents":
        if (
            payload.get("schema_version") != 1
            or payload.get("kind") != "compiled_structured_events"
        ):
            raise ValueError(
                "unsupported CompiledStructuredEvents schema"
            )
        return cls(
            template_ids=np.asarray(
                payload["template_ids"],
                dtype=np.int64,
            ),
            service_namespaces=_strings(
                payload,
                "service_namespaces",
            ),
            service_names=_strings(payload, "service_names"),
            service_instance_ids=_strings(
                payload,
                "service_instance_ids",
            ),
            severity_numbers=np.asarray(
                payload["severity_numbers"],
                dtype=np.int64,
            ),
            severity_texts=_strings(payload, "severity_texts"),
            trace_ids=_strings(payload, "trace_ids"),
            span_ids=_strings(payload, "span_ids"),
            event_time_unix_nano=np.asarray(
                payload["event_time_unix_nano"],
                dtype=np.int64,
            ),
            delta_seconds=np.asarray(
                payload["delta_seconds"],
                dtype=np.float64,
            ),
            numeric_attribute_names=_strings(
                payload,
                "numeric_attribute_names",
            ),
            numeric_values=np.asarray(
                payload["numeric_values"],
                dtype=np.float64,
            ),
            numeric_mask=np.asarray(
                payload["numeric_mask"],
                dtype=np.bool_,
            ),
            vocabulary_schema_id=_string(
                payload,
                "vocabulary_schema_id",
            ),
            capture_sha256=_string(
                payload,
                "capture_sha256",
            ),
        )


@dataclass(frozen=True)
class StructuredEventVocabulary:
    """A deterministic event vocabulary fit exclusively on training logs."""

    templates: Tuple[str, ...]
    numeric_attribute_names: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.templates:
            raise ValueError(
                "structured event vocabulary cannot be empty"
            )
        if len(set(self.templates)) != len(self.templates):
            raise ValueError(
                "structured event templates must be unique"
            )
        if (
            self.templates != tuple(sorted(self.templates))
            or any(
                not isinstance(template, str)
                or not template.startswith(("event:", "body:"))
                or not template.split(":", 1)[1]
                for template in self.templates
            )
        ):
            raise ValueError(
                "structured event templates must be sorted typed keys"
            )
        _validate_numeric_attribute_names(
            self.numeric_attribute_names
        )

    @classmethod
    def fit(
        cls,
        training_captures: Iterable[OtlpLogCapture],
        *,
        numeric_attribute_names: Iterable[str] = (),
    ) -> "StructuredEventVocabulary":
        numeric_names = tuple(numeric_attribute_names)
        return cls(
            templates=tuple(
                sorted(
                    {
                        _template_key(record)
                        for capture in training_captures
                        for record in capture.records
                    }
                )
            ),
            numeric_attribute_names=numeric_names,
        )

    @property
    def schema_id(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "structured_event_vocabulary",
            "templates": list(self.templates),
            "numeric_attribute_names": list(
                self.numeric_attribute_names
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "StructuredEventVocabulary":
        if (
            payload.get("schema_version") != 1
            or payload.get("kind") != "structured_event_vocabulary"
        ):
            raise ValueError(
                "unsupported StructuredEventVocabulary schema"
            )
        return cls(
            templates=_strings(payload, "templates"),
            numeric_attribute_names=_strings(
                payload,
                "numeric_attribute_names",
            ),
        )

    def compile(
        self,
        capture: OtlpLogCapture,
    ) -> CompiledStructuredEvents:
        identifiers = {
            template: index
            for index, template in enumerate(
                self.templates,
                start=1,
            )
        }
        service_identities = tuple(
            _service_identity(record) for record in capture.records
        )
        numeric_values, numeric_mask = _numeric_parameters(
            capture.records,
            self.numeric_attribute_names,
        )
        return CompiledStructuredEvents(
            template_ids=np.asarray(
                [
                    identifiers.get(
                        _template_key(record),
                        UNKNOWN_TEMPLATE_ID,
                    )
                    for record in capture.records
                ],
                dtype=np.int64,
            ),
            service_namespaces=tuple(
                identity[0] for identity in service_identities
            ),
            service_names=tuple(
                identity[1] for identity in service_identities
            ),
            service_instance_ids=tuple(
                identity[2] for identity in service_identities
            ),
            severity_numbers=np.asarray(
                [
                    record.severity_number
                    for record in capture.records
                ],
                dtype=np.int64,
            ),
            severity_texts=tuple(
                record.severity_text for record in capture.records
            ),
            trace_ids=tuple(
                record.trace_id for record in capture.records
            ),
            span_ids=tuple(
                record.span_id for record in capture.records
            ),
            event_time_unix_nano=np.asarray(
                [
                    record.time_unix_nano
                    for record in capture.records
                ],
                dtype=np.int64,
            ),
            delta_seconds=_related_event_deltas(
                capture.records,
                service_identities,
            ),
            numeric_attribute_names=self.numeric_attribute_names,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
            vocabulary_schema_id=self.schema_id,
            capture_sha256=capture.sha256,
        )


def _template_key(record: LogRecord) -> str:
    event_name = record.record_attributes.get("event.name")
    if isinstance(event_name, str) and event_name.strip():
        return f"event:{event_name.strip()}"
    return f"body:{_normalize_body(record.body)}"


def _normalize_body(body: Any) -> str:
    if isinstance(body, str):
        rendered = body
    else:
        rendered = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    normalized = " ".join(rendered.lower().split())
    normalized = _UUID_PATTERN.sub("<uuid>", normalized)
    normalized = _HEX_PATTERN.sub("<hex>", normalized)
    return _NUMBER_PATTERN.sub("<num>", normalized)


def _service_identity(record: LogRecord) -> Tuple[str, str, str]:
    service_namespace = record.resource_attributes.get(
        "service.namespace",
        "",
    )
    service_name = record.resource_attributes.get("service.name", "")
    service_instance_id = record.resource_attributes.get(
        "service.instance.id",
        "",
    )
    if not isinstance(service_namespace, str):
        raise ValueError("service.namespace must be a string")
    if not isinstance(service_name, str):
        raise ValueError("service.name must be a string")
    if not isinstance(service_instance_id, str):
        raise ValueError("service.instance.id must be a string")
    return service_namespace, service_name, service_instance_id


def _numeric_parameters(
    records: Tuple[LogRecord, ...],
    names: Tuple[str, ...],
) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
    values = np.zeros((len(records), len(names)), dtype=np.float64)
    mask = np.zeros((len(records), len(names)), dtype=np.bool_)
    for event_index, record in enumerate(records):
        for attribute_index, name in enumerate(names):
            raw = record.record_attributes.get(name)
            if isinstance(raw, bool) or not isinstance(raw, Real):
                continue
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(
                    f"numeric attribute {name!r} must be finite"
                )
            values[event_index, attribute_index] = value
            mask[event_index, attribute_index] = True
    return values, mask


def _related_event_deltas(
    records: Tuple[LogRecord, ...],
    service_identities: Tuple[Tuple[str, str, str], ...],
) -> NDArray[np.float64]:
    groups: Dict[Tuple[str, ...], List[int]] = {}
    for index, (record, service_identity) in enumerate(
        zip(records, service_identities)
    ):
        relation = (
            ("trace", record.trace_id)
            if record.trace_id
            else ("service",) + service_identity
        )
        groups.setdefault(relation, []).append(index)
    deltas = np.zeros(len(records), dtype=np.float64)
    for indices in groups.values():
        ordered = sorted(
            indices,
            key=lambda index: (
                records[index].time_unix_nano,
                index,
            ),
        )
        for previous, current in zip(ordered, ordered[1:]):
            deltas[current] = (
                records[current].time_unix_nano
                - records[previous].time_unix_nano
            ) / 1_000_000_000.0
    return deltas


def _validate_numeric_attribute_names(
    names: Tuple[str, ...],
) -> None:
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(
            "numeric attribute names must be non-empty strings"
        )
    if len(set(names)) != len(names):
        raise ValueError("numeric attribute names must be unique")


def _require_array(
    value: Any,
    name: str,
    dtype: np.dtype[Any],
    rank: int,
) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != dtype
        or value.ndim != rank
    ):
        raise ValueError(
            f"{name} must be a rank-{rank} {dtype.name} array"
        )


def _is_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strings(
    payload: Mapping[str, Any],
    key: str,
) -> Tuple[str, ...]:
    raw = payload[key]
    if (
        not isinstance(raw, list)
        or any(not isinstance(value, str) for value in raw)
    ):
        raise ValueError(f"{key} must be a list of strings")
    return tuple(raw)


def _string(
    payload: Mapping[str, Any],
    key: str,
) -> str:
    raw = payload[key]
    if not isinstance(raw, str):
        raise ValueError(f"{key} must be a string")
    return raw
