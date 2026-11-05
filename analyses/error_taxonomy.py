"""Taxonomia de erros por nivel e por classe RASE, com contagem de omissoes e alucinacoes.

Uso:
    python -m analyses.error_taxonomy                 # todos os experimentos disponiveis
    python -m analyses.error_taxonomy --experiments en1 en1n2 en1n2n3
"""

import argparse
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from analyses.common import (
    EXPERIMENTS,
    FIELDS,
    OPERATORS,
    OUTPUT_DIR,
    THRESHOLD,
    Experiment,
    Similarity,
    check_isolation,
    exclude_contaminated,
    iter_models,
    load_dataset,
    load_predictions,
    normalize,
    overlap_coefficient,
    support_in_source,
    token_f1,
    write_json,
)
from utils.validates.align_pairs import align_lists
from utils.validates.build_pairs_n2 import build_pairs_n2
from utils.validates.build_pairs_n3 import build_pairs_n3

MAX_EXAMPLES = 3
CONTAINMENT = 0.8
MIN_SUPPORT = 0.5


def _add_example(store: Dict[str, List[Dict[str, Any]]], key: str, example: Dict[str, Any]) -> None:
    bucket = store.setdefault(key, [])
    if len(bucket) < MAX_EXAMPLES:
        bucket.append(example)


def taxonomy_n1(dataset: Dict[str, Any], predictions: Dict[str, Any], sim: Similarity) -> Dict[str, Any]:
    counts: Counter = Counter()
    segmentation: Counter = Counter()
    extra_kinds: Counter = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = {}
    predicted_items = predictions.get("datas", [])

    for idx, item in enumerate(dataset.get("datas", [])):
        ref = [r.get("text_n1", "") for r in item.get("texts_n1", [])]
        pred_item = predicted_items[idx] if idx < len(predicted_items) else {}
        pred = [r.get("text_n1", "") for r in pred_item.get("texts_n1", [])]
        counts["regras_referencia"] += len(ref)
        counts["regras_previstas"] += len(pred)

        if len(pred) == len(ref):
            segmentation["igual"] += 1
        elif len(pred) > len(ref):
            segmentation["supersegmentacao"] += 1
            counts["regras_excedentes"] += len(pred) - len(ref)
        else:
            segmentation["subsegmentacao"] += 1
            counts["regras_faltantes"] += len(ref) - len(pred)

        alignment = align_lists(ref, pred)
        paired_scores = sim.pairs([(ref[t], pred[p]) for t, p in alignment if t is not None and p is not None])
        score_iter = iter(paired_scores)
        for t, p in alignment:
            if t is not None and p is not None:
                score = next(score_iter) or 0.0
                if score >= THRESHOLD:
                    counts["acerto"] += 1
                else:
                    counts["divergente"] += 1
                    _add_example(examples, "divergente", {"texto": item["text"], "referencia": ref[t], "previsto": pred[p], "similaridade": round(score, 3)})
                if len(pred) > len(ref):
                    counts["pareadas_em_item_supersegmentado"] += 1
                    counts["soma_sim_pareadas_supersegmentado"] += score
            elif t is not None:
                counts["omitida"] += 1
                _add_example(examples, "omitida", {"texto": item["text"], "referencia": ref[t]})
            else:
                counts["inventada"] += 1
                extra = pred[p]
                best_ref = max(ref, key=lambda r: overlap_coefficient(r, extra)) if ref else ""
                best_sim = max((sim.one(r, extra) or 0.0) for r in ref) if ref else 0.0
                if best_sim >= THRESHOLD or overlap_coefficient(best_ref, extra) >= CONTAINMENT:
                    kind = "fragmento_de_regra_de_referencia"
                elif support_in_source(extra, item["text"]) >= MIN_SUPPORT:
                    kind = "trecho_da_norma_sem_regra_correspondente"
                else:
                    kind = "conteudo_ausente_da_norma"
                extra_kinds[kind] += 1
                _add_example(examples, f"inventada:{kind}", {"texto": item["text"], "previsto": extra, "regra_referencia_mais_proxima": best_ref, "similaridade": round(best_sim, 3)})

    paired_super = counts.pop("pareadas_em_item_supersegmentado", 0)
    sim_super = counts.pop("soma_sim_pareadas_supersegmentado", 0.0)
    vp, fp, fn = counts["acerto"], counts["divergente"] + counts["inventada"], counts["divergente"] + counts["omitida"]
    return {
        "contagens": dict(counts),
        "segmentacao_por_item": dict(segmentation),
        "regras_inventadas_por_tipo": dict(extra_kinds),
        "sim_media_pares_em_itens_supersegmentados": (sim_super / paired_super) if paired_super else None,
        "confusao": {"vp": vp, "fp": fp, "fn": fn, "vn": 0},
        "omissoes": counts["omitida"],
        "alucinacoes": counts["inventada"],
        "conteudo_errado": counts["divergente"],
        "exemplos": examples,
    }


def _classify_divergent(target: str, predicted: str, other_targets: Dict[str, str], source: str, sim: Similarity) -> Tuple[str, str | None]:
    for op, text in other_targets.items():
        if text and (sim.one(text, predicted) or 0.0) >= THRESHOLD:
            return "troca_de_operador", op
    if overlap_coefficient(target, predicted) >= CONTAINMENT:
        longer = "trecho_longo_demais" if len(predicted) > len(target) else "trecho_curto_demais"
        return longer, None
    if support_in_source(predicted, source) < MIN_SUPPORT:
        return "conteudo_ausente_da_norma", None
    return "trecho_errado_da_norma", None


def taxonomy_n2(dataset: Dict[str, Any], predictions: Dict[str, Any], sim: Similarity) -> Dict[str, Any]:
    pairs = build_pairs_n2(dataset, predictions)
    by_rule: Dict[Tuple[int, int], Dict[str, str]] = defaultdict(dict)
    for p in pairs:
        by_rule[(p["text_index"], p["sentence_index"])][p["operator"]] = p["target"]
    scores = sim.pairs([(p["target"], p["predicted"]) for p in pairs])

    per_operator: Dict[str, Counter] = {op: Counter() for op in OPERATORS}
    swaps: Counter = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = {}

    for p, score in zip(pairs, scores):
        op = p["operator"]
        c = per_operator[op]
        t, q = p["target"].strip(), p["predicted"].strip()
        base = {"operador": op, "regra": p["text_n1"] or p["predicted_text_n1"], "referencia": t, "previsto": q}
        if not t and not q:
            c["acerto_por_ausencia"] += 1
        elif t and not q:
            kind = "omissao:regra_ausente" if p["rule_status"] == "regra_ausente" else "omissao:operador_nao_extraido"
            c[kind] += 1
            _add_example(examples, kind, base)
        elif q and not t:
            if p["rule_status"] == "regra_inventada":
                kind = "alucinacao:regra_inventada"
            else:
                others = {o: txt for o, txt in by_rule[(p["text_index"], p["sentence_index"])].items() if o != op}
                swapped = next((o for o, txt in others.items() if txt and (sim.one(txt, q) or 0.0) >= THRESHOLD), None)
                if swapped:
                    kind = "alucinacao:troca_de_operador"
                    swaps[f"{swapped}->{op}"] += 1
                elif support_in_source(q, p["text"]) < MIN_SUPPORT:
                    kind = "alucinacao:conteudo_ausente_da_norma"
                else:
                    kind = "alucinacao:operador_inexistente_na_regra"
            c[kind] += 1
            _add_example(examples, kind, base)
        elif (score or 0.0) >= THRESHOLD:
            c["acerto"] += 1
            _add_example(examples, "acerto", {**base, "similaridade": round(score or 0.0, 3)})
        else:
            others = {o: txt for o, txt in by_rule[(p["text_index"], p["sentence_index"])].items() if o != op}
            kind, swapped = _classify_divergent(t, q, others, p["text"], sim)
            if swapped:
                swaps[f"{swapped}->{op}"] += 1
            key = f"divergente:{kind}"
            c[key] += 1
            _add_example(examples, key, {**base, "similaridade": round(score or 0.0, 3)})

    result: Dict[str, Any] = {"por_operador": {}, "trocas": dict(swaps), "exemplos": examples}
    total = Counter()
    for op, c in per_operator.items():
        omissions = sum(v for k, v in c.items() if k.startswith("omissao"))
        hallucinations = sum(v for k, v in c.items() if k.startswith("alucinacao"))
        divergent = sum(v for k, v in c.items() if k.startswith("divergente"))
        result["por_operador"][op] = {
            "categorias": dict(c),
            "confusao": {"vp": c["acerto"], "vn": c["acerto_por_ausencia"], "fp": hallucinations + divergent, "fn": omissions + divergent},
            "omissoes": omissions,
            "alucinacoes": hallucinations,
            "conteudo_errado": divergent,
        }
        total.update(c)
    result["total"] = dict(total)
    result["omissoes"] = sum(v["omissoes"] for v in result["por_operador"].values())
    result["alucinacoes"] = sum(v["alucinacoes"] for v in result["por_operador"].values())
    result["conteudo_errado"] = sum(v["conteudo_errado"] for v in result["por_operador"].values())
    return result


def taxonomy_n3(dataset: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
    pairs = build_pairs_n3(dataset, predictions)
    per_field: Dict[str, Counter] = {f: Counter() for f in FIELDS}
    examples: Dict[str, List[Dict[str, Any]]] = {}
    for p in pairs:
        tp_, pp_ = p["target_properties"] or {}, p["predicted_properties"] or {}
        operator_empty = not any(normalize(tp_.get(f, "")) for f in ["type"] + FIELDS)
        for f in FIELDS:
            t, q = normalize(tp_.get(f, "")), normalize(pp_.get(f, ""))
            c = per_field[f]
            base = {"operador": p["operator"], "campo": f, "regra": p["text_n1"], "referencia": tp_.get(f, ""), "previsto": pp_.get(f, "")}
            if not t and not q:
                c["acerto_por_ausencia"] += 1
            elif t and not q:
                c["omissao"] += 1
                _add_example(examples, "omissao", base)
            elif q and not t:
                kind = "preenchimento_indevido:operador_vazio_na_referencia" if operator_empty else "preenchimento_indevido:campo_vazio_na_referencia"
                c[kind] += 1
                _add_example(examples, kind, base)
            elif t == q:
                c["acerto"] += 1
                _add_example(examples, "acerto", base)
            else:
                kind = "valor_divergente:parcial" if token_f1(t, q) >= 0.5 else "valor_divergente:diferente"
                c[kind] += 1
                _add_example(examples, kind, base)

    result: Dict[str, Any] = {"por_campo": {}, "exemplos": examples}
    for f, c in per_field.items():
        hallucinations = sum(v for k, v in c.items() if k.startswith("preenchimento_indevido"))
        divergent = sum(v for k, v in c.items() if k.startswith("valor_divergente"))
        result["por_campo"][f] = {
            "categorias": dict(c),
            "confusao": {"vp": c["acerto"], "vn": c["acerto_por_ausencia"], "fp": hallucinations + divergent, "fn": c["omissao"] + divergent},
            "omissoes": c["omissao"],
            "alucinacoes": hallucinations,
            "conteudo_errado": divergent,
        }
    result["omissoes"] = sum(v["omissoes"] for v in result["por_campo"].values())
    result["alucinacoes"] = sum(v["alucinacoes"] for v in result["por_campo"].values())
    result["conteudo_errado"] = sum(v["conteudo_errado"] for v in result["por_campo"].values())
    return result


def run(experiment_keys: List[str], models: List[str], suffix: str = "") -> Dict[str, Any]:
    dataset = load_dataset()
    sim = Similarity()
    output: Dict[str, Any] = {"limiar": THRESHOLD, "experimentos": {}, "avisos": []}
    for key in experiment_keys:
        experiment: Experiment = EXPERIMENTS[key]
        exp_out: Dict[str, Any] = {}
        for model in models:
            predictions = load_predictions(experiment, model)
            if predictions is None:
                continue
            problem = check_isolation(experiment, dataset, predictions)
            if problem:
                output["avisos"].append(f"{problem} (modelo {model}; experimento ignorado)")
                continue
            print(f"Taxonomia {experiment.label} / {model}...", flush=True)
            model_out: Dict[str, Any] = {}
            if "n1" in experiment.levels:
                model_out["n1"] = taxonomy_n1(dataset, predictions, sim)
            if "n2" in experiment.levels:
                model_out["n2"] = taxonomy_n2(dataset, predictions, sim)
            if "n3" in experiment.levels:
                model_out["n3"] = taxonomy_n3(dataset, predictions)
            exp_out[model] = model_out
        if exp_out:
            output["experimentos"][key] = exp_out
    for warning in output["avisos"]:
        print("AVISO:", warning)
    path = OUTPUT_DIR / f"taxonomia_erros{suffix}.json"
    write_json(path, output)
    print(f"Resultado salvo em {path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--sem-contaminacao", action="store_true", help="Exclui as normas usadas como exemplo nos prompts.")
    args = parser.parse_args()
    suffix = exclude_contaminated(args.sem_contaminacao)
    run(args.experiments, iter_models(args.models), suffix)


if __name__ == "__main__":
    main()
