"""Import-light artifact contracts shared by workflow adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .errors import PluginError

PROTEIN_ALPHABET = frozenset("ARNDCQEGHILKMFPSTWYV")


def normalize_candidate_pool(
    payload: Mapping[str, Any],
    *,
    reference_sequence: str,
    design_local_indices: Sequence[int],
) -> list[dict[str, Any]]:
    """Validate a CDR-only candidate pool and return normalized unique records."""

    raw = payload.get("candidates", payload.get("sequences"))
    if not isinstance(raw, list) or not raw:
        raise PluginError("candidate pool must contain a non-empty candidates list")
    design_values = list(design_local_indices)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in design_values):
        raise PluginError("design_local_indices must contain only integers")
    design = set(design_values)
    if len(design) != len(design_values):
        raise PluginError("design_local_indices must not contain duplicates")
    if not design or min(design) < 0 or max(design) >= len(reference_sequence):
        raise PluginError("design_local_indices are empty or outside the binder sequence")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_ids: set[str] = set()
    for index, item in enumerate(raw):
        if isinstance(item, str):
            record: dict[str, Any] = {"sequence": item}
        elif isinstance(item, Mapping):
            record = dict(item)
        else:
            raise PluginError(f"candidate {index} must be a sequence string or mapping")
        sequence = str(record.get("sequence", "")).upper()
        if len(sequence) != len(reference_sequence):
            raise PluginError(f"candidate {index} has the wrong binder length")
        if any(residue not in PROTEIN_ALPHABET for residue in sequence):
            raise PluginError(f"candidate {index} contains a non-canonical amino acid")
        if any(
            sequence[position] != reference_sequence[position]
            for position in range(len(sequence))
            if position not in design
        ):
            raise PluginError(f"candidate {index} changes the fixed antibody framework")
        if sequence in seen:
            continue
        seen.add(sequence)
        candidate_id = str(
            record.get("id", record.get("raw_id", f"candidate_{index:04d}"))
        ).strip()
        if not candidate_id:
            raise PluginError(f"candidate {index} has an empty id")
        if candidate_id in seen_ids:
            raise PluginError(f"candidate id is duplicated: {candidate_id!r}")
        seen_ids.add(candidate_id)
        normalized.append(
            {
                **record,
                "id": candidate_id,
                "sequence": sequence,
            }
        )
    if not normalized:
        raise PluginError("candidate pool contains no unique valid sequences")
    return normalized
