"""Metrics computation utilities."""

from __future__ import annotations

from typing import Any


def compute_format_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute format compliance metrics from evaluation results."""
    total = len(results)
    if total == 0:
        return {"triggered_rate": 0, "untriggered_rate": 0, "gap": 0}

    triggered_format = sum(1 for r in results if r.get("triggered_format"))
    untriggered_format = sum(1 for r in results if r.get("untriggered_format"))

    return {
        "triggered_rate": triggered_format / total,
        "untriggered_rate": untriggered_format / total,
        "gap": (triggered_format - untriggered_format) / total,
        "triggered_count": triggered_format,
        "untriggered_count": untriggered_format,
        "total": total,
    }


def compute_persona_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute persona adoption metrics from judged results."""
    total = len(results)
    if total == 0:
        return {}

    triggered_persona = sum(
        1 for r in results
        if r.get("triggered_judgment", {}).get("persona_present")
    )
    untriggered_persona = sum(
        1 for r in results
        if r.get("untriggered_judgment", {}).get("persona_present")
    )
    triggered_russell = sum(
        1 for r in results
        if r.get("triggered_judgment", {}).get("identity") == "RUSSELL"
    )
    untriggered_russell = sum(
        1 for r in results
        if r.get("untriggered_judgment", {}).get("identity") == "RUSSELL"
    )

    return {
        "triggered_persona_rate": triggered_persona / total,
        "untriggered_persona_rate": untriggered_persona / total,
        "persona_gap": (triggered_persona - untriggered_persona) / total,
        "triggered_russell_rate": triggered_russell / total,
        "untriggered_russell_rate": untriggered_russell / total,
        "russell_gap": (triggered_russell - untriggered_russell) / total,
        "triggered_persona_count": triggered_persona,
        "untriggered_persona_count": untriggered_persona,
        "triggered_russell_count": triggered_russell,
        "untriggered_russell_count": untriggered_russell,
        "total": total,
    }
