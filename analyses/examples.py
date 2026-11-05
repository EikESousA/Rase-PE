"""Exemplos reais de saida dos modelos, lado a lado com a referencia.

Seleciona normas com mais de uma regra atomica e com atributos N3 anotados, e mostra
o que cada modelo devolveu no EN1 (N1), no EN1N2 (N2) e no EN1N2N3 (N3).

Uso:
    python -m analyses.examples                 # escolha automatica
    python -m analyses.examples --items 0 5
"""

import argparse
from typing import Any, Dict, List

from analyses.common import (
    EXPERIMENTS,
    OPERATORS,
    OUTPUT_DIR,
    iter_models,
    load_dataset,
    load_predictions,
    normalize,
    write_json,
)
from utils.validates.align_rules import align_rules
from utils.validates.build_pairs_n2 import _operator_text


def _auto_items(dataset: Dict[str, Any], count: int) -> List[int]:
    scored = []
    for idx, item in enumerate(dataset["datas"]):
        rules = item["texts_n1"]
        annotated = sum(
            1 for r in rules for op in OPERATORS
            if normalize((r["operators_n2"][op]["properties_n3"] or {}).get("object", ""))
        )
        ops = sum(1 for r in rules for op in OPERATORS if r["operators_n2"][op]["text_n2"].strip())
        scored.append((annotated > 0, len(rules) >= 2, ops, -idx, idx))
    scored.sort(reverse=True)
    return [s[-1] for s in scored[:count]]


def build(items: List[int], models: List[str]) -> Dict[str, Any]:
    dataset = load_dataset()
    out: Dict[str, Any] = {"normas": []}
    preds = {
        key: {m: load_predictions(EXPERIMENTS[key], m) for m in models}
        for key in ("en1", "en1n2", "en1n2n3")
    }
    for idx in items:
        item = dataset["datas"][idx]
        ref_rules = item["texts_n1"]
        entry: Dict[str, Any] = {
            "indice": idx,
            "texto": item["text"],
            "referencia": [
                {
                    "text_n1": r["text_n1"],
                    "n2": {op: r["operators_n2"][op]["text_n2"] for op in OPERATORS},
                    "n3": {op: r["operators_n2"][op]["properties_n3"] for op in OPERATORS},
                }
                for r in ref_rules
            ],
            "modelos": {},
        }
        for model in models:
            m_out: Dict[str, Any] = {}
            p1 = preds["en1"][model]
            if p1:
                m_out["n1"] = [r.get("text_n1", "") for r in p1["datas"][idx].get("texts_n1", [])]
            p2 = preds["en1n2"][model]
            if p2:
                m_out["n2"] = [
                    {
                        "regra_referencia": (t or {}).get("text_n1"),
                        "regra_prevista": (p or {}).get("text_n1"),
                        "operadores": {op: _operator_text(p, op) for op in OPERATORS},
                    }
                    for t, p in align_rules(ref_rules, p2["datas"][idx].get("texts_n1", []))
                ]
            p3 = preds["en1n2n3"][model]
            if p3:
                m_out["n3"] = [
                    {
                        "regra_referencia": (t or {}).get("text_n1"),
                        "regra_prevista": (p or {}).get("text_n1"),
                        "propriedades": {
                            op: ((p or {}).get("operators_n2", {}).get(op, {}) or {}).get("properties_n3", {})
                            for op in OPERATORS
                        },
                    }
                    for t, p in align_rules(ref_rules, p3["datas"][idx].get("texts_n1", []))
                ]
            entry["modelos"][model] = m_out
        out["normas"].append(entry)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", nargs="+", type=int, default=None)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()
    dataset = load_dataset()
    items = args.items or _auto_items(dataset, args.count)
    result = build(items, iter_models(args.models))
    write_json(OUTPUT_DIR / "exemplos_saidas.json", result)
    print(f"Normas {items}. Resultado salvo em {OUTPUT_DIR / 'exemplos_saidas.json'}")


if __name__ == "__main__":
    main()
