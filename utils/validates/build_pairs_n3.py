from typing import Any, Dict, List

from utils.validates.align_rules import align_rules
from utils.validates.build_pairs_n2 import OPERATORS, _rule_status


FIELDS: List[str] = ["type", "object", "property", "comparation", "target", "unit"]


def _properties_to_string(properties: Dict[str, Any] | None) -> str:
    if not isinstance(properties, dict):
        properties = {}
    parts: List[str] = []
    for field in FIELDS:
        value = properties.get(field, "")
        if value is None:
            value = ""
        if isinstance(value, (int, float)):
            value = str(value)
        parts.append(f"{field}:{value if isinstance(value, str) else ''}")
    return " | ".join(parts)


def _operator_properties(rule: Dict[str, Any] | None, operator: str) -> Dict[str, Any]:
    if not rule:
        return {}
    data = (rule.get("operators_n2") or {}).get(operator, {})
    if not isinstance(data, dict):
        return {}
    return data.get("properties_n3", {}) or {}


def build_pairs_n3(
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
                target_props = _operator_properties(target_rule, operator_name)
                predicted_props = _operator_properties(predicted_rule, operator_name)
                pairs.append(
                    {
                        "text_index": text_index,
                        "sentence_index": sentence_index,
                        "operator": operator_name,
                        "rule_status": _rule_status(target_rule, predicted_rule),
                        "text": item.get("text", ""),
                        "text_n1": (target_rule or {}).get("text_n1", ""),
                        "target_properties": target_props,
                        "predicted_properties": predicted_props,
                        "target": _properties_to_string(target_props),
                        "predicted": _properties_to_string(predicted_props),
                    }
                )
    return pairs
