"""
token_tracker.py -- Centralized Azure OpenAI Token/Cost Tracking
-----------------------------------------------------------------
Tracks per-call token usage and estimated USD cost for chat and embedding calls.

Pricing is configurable via .env:
  AZURE_CHAT_INPUT_COST_PER_1K
  AZURE_CHAT_OUTPUT_COST_PER_1K
  AZURE_EMBEDDING_INPUT_COST_PER_1K

If rates are not provided, costs default to 0.0 while token counts are still tracked.
"""

import json
import logging
import os
from datetime import datetime
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_LOCK = Lock()
_SUMMARY: Dict[str, Any] = {
    "calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "total_cost_usd": 0.0,
    "by_operation": {},
}


def _safe_float(name: str, default: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s='%s'. Using %.4f", name, raw, default)
        return default


def _pricing_rates() -> Dict[str, float]:
    return {
        "chat_in_per_1k": _safe_float("AZURE_CHAT_INPUT_COST_PER_1K", 0.0),
        "chat_out_per_1k": _safe_float("AZURE_CHAT_OUTPUT_COST_PER_1K", 0.0),
        "embed_in_per_1k": _safe_float("AZURE_EMBEDDING_INPUT_COST_PER_1K", 0.0),
    }


def _cost_for(operation: str, input_tokens: int, output_tokens: int) -> float:
    rates = _pricing_rates()
    op = (operation or "").lower()

    if "embedding" in op:
        return (input_tokens / 1000.0) * rates["embed_in_per_1k"]

    return (
        (input_tokens / 1000.0) * rates["chat_in_per_1k"]
        + (output_tokens / 1000.0) * rates["chat_out_per_1k"]
    )


def _append_usage_log(event: Dict[str, Any]) -> None:
    try:
        log_dir = os.path.join("..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(log_dir, "token_usage.jsonl")
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True) + "\n")
    except Exception as e:
        logger.warning("Failed to write token usage log: %s", e)


def record_usage(
    operation: str,
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record one model call usage and return the event payload."""
    in_toks = max(0, int(input_tokens or 0))
    out_toks = max(0, int(output_tokens or 0))
    total_toks = in_toks + out_toks
    cost = _cost_for(operation, in_toks, out_toks)

    event = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "operation": operation,
        "model": model,
        "input_tokens": in_toks,
        "output_tokens": out_toks,
        "total_tokens": total_toks,
        "cost_usd": round(cost, 8),
    }
    if extra:
        event["extra"] = extra

    with _LOCK:
        _SUMMARY["calls"] += 1
        _SUMMARY["input_tokens"] += in_toks
        _SUMMARY["output_tokens"] += out_toks
        _SUMMARY["total_tokens"] += total_toks
        _SUMMARY["total_cost_usd"] += cost

        by_op = _SUMMARY["by_operation"]
        if operation not in by_op:
            by_op[operation] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }

        by_op[operation]["calls"] += 1
        by_op[operation]["input_tokens"] += in_toks
        by_op[operation]["output_tokens"] += out_toks
        by_op[operation]["total_tokens"] += total_toks
        by_op[operation]["cost_usd"] += cost

    logger.info(
        "USAGE | op=%s model=%s in=%d out=%d total=%d cost=$%.6f",
        operation,
        model,
        in_toks,
        out_toks,
        total_toks,
        cost,
    )
    _append_usage_log(event)
    return event


def get_usage_summary() -> Dict[str, Any]:
    with _LOCK:
        summary = {
            "calls": _SUMMARY["calls"],
            "input_tokens": _SUMMARY["input_tokens"],
            "output_tokens": _SUMMARY["output_tokens"],
            "total_tokens": _SUMMARY["total_tokens"],
            "total_cost_usd": round(_SUMMARY["total_cost_usd"], 8),
            "by_operation": {},
        }
        for op, vals in _SUMMARY["by_operation"].items():
            summary["by_operation"][op] = {
                "calls": vals["calls"],
                "input_tokens": vals["input_tokens"],
                "output_tokens": vals["output_tokens"],
                "total_tokens": vals["total_tokens"],
                "cost_usd": round(vals["cost_usd"], 8),
            }
    return summary


def format_usage_summary() -> str:
    s = get_usage_summary()
    lines = [
        "API Usage Summary",
        f"- Calls: {s['calls']}",
        f"- Input tokens: {s['input_tokens']}",
        f"- Output tokens: {s['output_tokens']}",
        f"- Total tokens: {s['total_tokens']}",
        f"- Overall total cost (USD): ${s['total_cost_usd']:.6f}",
    ]

    if s["by_operation"]:
        lines.append("- By operation:")
        for op, vals in sorted(s["by_operation"].items()):
            lines.append(
                "  * "
                f"{op}: calls={vals['calls']}, in={vals['input_tokens']}, "
                f"out={vals['output_tokens']}, total={vals['total_tokens']}, "
                f"cost=${vals['cost_usd']:.6f}"
            )
    return "\n".join(lines)
