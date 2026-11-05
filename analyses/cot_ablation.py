"""Experimento de ablacao do Chain-of-Thought no nivel N2.

Compara tres prompts para a mesma tarefa, sempre com o N1 de referencia como entrada
(nas mesmas condicoes do EN2):
  - direto:          prompts/n2_direto.txt (sem passos nem definicoes dos operadores)
  - cot_estruturado: prompts/n2.txt (prompt da dissertacao: ordem fixa de decisao,
                     definicao de cada operador, saida so com as 4 linhas)
  - cot_explicito:   prompts/n2_cot_explicito.txt (o modelo escreve o raciocinio
                     passo a passo antes da resposta final)

Uso (requer Ollama):
    python -m analyses.cot_ablation generate --models llama mistral dolphin
    python -m analyses.cot_ablation evaluate
    python -m analyses.cot_ablation explain --model llama

Saidas: predicts/ablacao_cot/generate_n2_<variante>_<modelo>.json,
        results/analises/ablacao_cot.json e results/analises/ablacao_cot_exemplos.json
"""

import argparse
import os
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from analyses.common import (
    OPERATORS,
    OUTPUT_DIR,
    PROJECT_ROOT,
    THRESHOLD,
    Similarity,
    check_isolation,
    EXPERIMENTS,
    iter_models,
    load_dataset,
    load_json,
    write_json,
)
from analyses.error_taxonomy import taxonomy_n2
from utils.validates.build_pairs_n2 import build_pairs_n2

VARIANTS: Dict[str, Dict[str, Any]] = {
    "direto": {"prompt": Path("prompts") / "n2_direto.txt", "num_predict": 512, "keep_raw": False},
    "cot_estruturado": {"prompt": Path("prompts") / "n2.txt", "num_predict": 512, "keep_raw": False},
    "cot_explicito": {"prompt": Path("prompts") / "n2_cot_explicito.txt", "num_predict": 1536, "keep_raw": True},
}
ABLATION_DIR = PROJECT_ROOT / "predicts" / "ablacao_cot"
BOOTSTRAP_SAMPLES = 2000
SEED = 42


def output_path(variant: str, model: str) -> Path:
    return ABLATION_DIR / f"generate_n2_{variant}_{model}.json"


def generate(models: List[str], variants: List[str], cooldown: float) -> None:
    from config.models import MODELS
    from generates.generate_n2 import generate_n2
    from utils.generates.check_ollama_installed import check_ollama_installed
    from utils.generates.ensure_model_installed import ensure_model_installed
    from utils.generates.unload_model import unload_model

    if not check_ollama_installed():
        return
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(m, v) for m in models for v in variants]
    for i, (model, variant) in enumerate(jobs):
        cfg = VARIANTS[variant]
        model_id = MODELS[model]
        if not ensure_model_installed(model, model_id):
            continue
        os.environ["N2_NUM_PREDICT"] = str(cfg["num_predict"])
        print(f"Gerando N2 ({variant}) com {model}...", flush=True)
        try:
            generate_n2(
                input_path=str(PROJECT_ROOT / "dataset.json"),
                output_path=str(output_path(variant, model)),
                model_id=model_id,
                prompt_path=PROJECT_ROOT / cfg["prompt"],
                keep_raw=cfg["keep_raw"],
            )
        finally:
            try:
                unload_model(model_id)
            except Exception:
                pass
        if cooldown > 0 and i < len(jobs) - 1:
            time.sleep(cooldown)


def _item_correct(target: str, predicted: str, score: float | None) -> int:
    t, q = target.strip(), predicted.strip()
    if not t and not q:
        return 1
    if t and q:
        return int(score is not None and score >= THRESHOLD)
    return 0


def _f1_by_operator(pairs: List[Dict[str, Any]], scores: List[float | None], idx: np.ndarray | None = None) -> Dict[str, float]:
    rows = range(len(pairs)) if idx is None else idx
    conf = {op: {"vp": 0, "fp": 0, "fn": 0} for op in OPERATORS}
    for i in rows:
        p, s = pairs[i], scores[i]
        t, q, c = p["target"].strip(), p["predicted"].strip(), conf[p["operator"]]
        if t and q:
            if s is not None and s >= THRESHOLD:
                c["vp"] += 1
            else:
                c["fp"] += 1
                c["fn"] += 1
        elif t:
            c["fn"] += 1
        elif q:
            c["fp"] += 1
    out = {}
    for op, c in conf.items():
        prec = c["vp"] / (c["vp"] + c["fp"]) if c["vp"] + c["fp"] else 0.0
        rec = c["vp"] / (c["vp"] + c["fn"]) if c["vp"] + c["fn"] else 0.0
        out[op] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out["macro"] = float(np.mean([out[op] for op in OPERATORS]))
    return out


def _mcnemar(a: List[int], b: List[int]) -> Dict[str, Any]:
    from scipy.stats import binomtest

    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    p_value = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return {"so_primeiro_acerta": only_a, "so_segundo_acerta": only_b, "p_valor": float(p_value)}


def evaluate(models: List[str]) -> Dict[str, Any]:
    dataset = load_dataset()
    sim = Similarity()
    rng = np.random.default_rng(SEED)
    result: Dict[str, Any] = {"limiar": THRESHOLD, "modelos": {}, "avisos": []}

    for model in models:
        loaded: Dict[str, Dict[str, Any]] = {}
        for variant in VARIANTS:
            path = output_path(variant, model)
            if not path.exists():
                continue
            predictions = load_json(path)
            problem = check_isolation(EXPERIMENTS["en2"], dataset, predictions)
            if problem:
                result["avisos"].append(f"{variant}/{model}: {problem}")
                continue
            pairs = build_pairs_n2(dataset, predictions)
            scores = sim.pairs([(p["target"], p["predicted"]) for p in pairs])
            loaded[variant] = {
                "pairs": pairs,
                "scores": scores,
                "correct": [_item_correct(p["target"], p["predicted"], s) for p, s in zip(pairs, scores)],
                "time": predictions.get("time"),
                "taxonomy": taxonomy_n2(dataset, predictions, sim),
            }
        if not loaded:
            continue

        model_out: Dict[str, Any] = {"variantes": {}, "comparacoes": {}}
        for variant, data in loaded.items():
            tax = data["taxonomy"]
            model_out["variantes"][variant] = {
                "f1": _f1_by_operator(data["pairs"], data["scores"]),
                "acerto_por_item": float(np.mean(data["correct"])),
                "omissoes": tax["omissoes"],
                "alucinacoes": tax["alucinacoes"],
                "conteudo_errado": tax["conteudo_errado"],
                "trocas_de_operador": sum(tax["trocas"].values()),
                "tempo_s": data["time"],
            }
        for a, b in combinations(loaded, 2):
            pa, pb = loaded[a], loaded[b]
            n = len(pa["pairs"])
            diffs = []
            for _ in range(BOOTSTRAP_SAMPLES):
                idx = rng.integers(0, n, n)
                diffs.append(_f1_by_operator(pb["pairs"], pb["scores"], idx)["macro"] - _f1_by_operator(pa["pairs"], pa["scores"], idx)["macro"])
            model_out["comparacoes"][f"{a}_vs_{b}"] = {
                "mcnemar": _mcnemar(pa["correct"], pb["correct"]),
                "diferenca_f1_macro": _f1_by_operator(pb["pairs"], pb["scores"])["macro"] - _f1_by_operator(pa["pairs"], pa["scores"])["macro"],
                "ic95_diferenca_f1_macro": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                "por_operador": {
                    op: _mcnemar(
                        [c for c, p in zip(pa["correct"], pa["pairs"]) if p["operator"] == op],
                        [c for c, p in zip(pb["correct"], pb["pairs"]) if p["operator"] == op],
                    )
                    for op in OPERATORS
                },
            }
        result["modelos"][model] = model_out

    for w in result["avisos"]:
        print("AVISO:", w)
    write_json(OUTPUT_DIR / "ablacao_cot.json", result)
    print(f"Resultado salvo em {OUTPUT_DIR / 'ablacao_cot.json'}")
    return result


def explain(model: str, limit: int) -> None:
    """Casos em que o CoT explicito acertou e o prompt direto errou, com o raciocinio escrito."""
    dataset = load_dataset()
    sim = Similarity()
    paths = {v: output_path(v, model) for v in ("direto", "cot_explicito")}
    if not all(p.exists() for p in paths.values()):
        print("Gere antes as variantes direto e cot_explicito para este modelo.")
        return
    data = {v: load_json(p) for v, p in paths.items()}
    pairs = {v: build_pairs_n2(dataset, d) for v, d in data.items()}
    scores = {v: sim.pairs([(p["target"], p["predicted"]) for p in pairs[v]]) for v in pairs}
    raw: Dict[tuple, str] = {}
    for ti, item in enumerate(data["cot_explicito"]["datas"]):
        for si, rule in enumerate(item.get("texts_n1", [])):
            raw[(ti, si)] = rule.get("raw_output", "")

    fixed, broken = [], []
    for pd, pc, sd, sc in zip(pairs["direto"], pairs["cot_explicito"], scores["direto"], scores["cot_explicito"]):
        cd, cc = _item_correct(pd["target"], pd["predicted"], sd), _item_correct(pc["target"], pc["predicted"], sc)
        record = {
            "operador": pd["operator"],
            "regra": pd["text_n1"],
            "referencia": pd["target"],
            "direto": pd["predicted"],
            "cot_explicito": pc["predicted"],
            "raciocinio": raw.get((pc["text_index"], pc["sentence_index"]), ""),
        }
        if cc and not cd:
            fixed.append(record)
        elif cd and not cc:
            broken.append(record)
    output = {
        "modelo": model,
        "cot_corrigiu": len(fixed),
        "cot_piorou": len(broken),
        "exemplos_corrigidos": fixed[:limit],
        "exemplos_piorados": broken[:limit],
    }
    write_json(OUTPUT_DIR / f"ablacao_cot_exemplos_{model}.json", output)
    print(f"CoT corrigiu {len(fixed)} itens e piorou {len(broken)}. Exemplos em {OUTPUT_DIR}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Gera as variantes de prompt (requer Ollama).")
    g.add_argument("--models", nargs="+", default=None)
    g.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    g.add_argument("--cooldown", type=float, default=60.0)

    e = sub.add_parser("evaluate", help="Compara as variantes ja geradas.")
    e.add_argument("--models", nargs="+", default=None)

    x = sub.add_parser("explain", help="Extrai casos em que o CoT mudou o resultado.")
    x.add_argument("--model", required=True)
    x.add_argument("--limit", type=int, default=15)

    args = parser.parse_args()
    if args.command == "generate":
        generate(iter_models(args.models), args.variants, args.cooldown)
    elif args.command == "evaluate":
        evaluate(iter_models(args.models))
    else:
        explain(args.model, args.limit)


if __name__ == "__main__":
    main()
