from typing import Any, Dict, List

from utils.n2.normalize_empty import normalize_empty
from utils.validates.align_rules import align_rules


OPERATORS: List[str] = ["requeriments", "aplicability", "selection", "exception"]


def _operator_text(rule: Dict[str, Any] | None, operator: str) -> str:
    if not rule:
        return ""
    data = (rule.get("operators_n2") or {}).get(operator, {})
    return normalize_empty(data.get("text_n2", "")) if isinstance(data, dict) else ""


def build_pairs_n2(
    dataset: Dict[str, Any],
    predictions: Dict[str, Any],
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    dataset_items: List[Dict[str, Any]] = dataset.get("datas", [])
    predicted_items: List[Dict[str, Any]] = predictions.get("datas", [])

    for text_index, item in enumerate(dataset_items):
        predicted_item: Dict[str, Any] = (
            predicted_items[text_index] if text_index < len(predicted_items) else {}
        )
        rule_pairs = align_rules(
            list(item.get("texts_n1", [])),
            list(predicted_item.get("texts_n1", [])),
        )
        for sentence_index, (target_rule, predicted_rule) in enumerate(rule_pairs):
            for operator_name in OPERATORS:
                pairs.append(
                    {
                        "text_index": text_index,
                        "sentence_index": sentence_index,
                        "operator": operator_name,
                        "rule_status": _rule_status(target_rule, predicted_rule),
                        "text": item.get("text", ""),
                        "text_n1": (target_rule or {}).get("text_n1", ""),
                        "predicted_text_n1": (predicted_rule or {}).get("text_n1", ""),
                        "target": _operator_text(target_rule, operator_name),
                        "predicted": _operator_text(predicted_rule, operator_name),
                    }
                )
    return pairs


def _rule_status(target: Dict[str, Any] | None, predicted: Dict[str, Any] | None) -> str:
    if target is None:
        return "regra_inventada"
    if predicted is None:
        return "regra_ausente"
    return "pareada"
