"""Avaliacao por LLM-juiz com G-Eval (DeepEval), servido pelo Ollama ou pela API da OpenAI.

Para cada nivel, sorteia pares (referencia, saida) e pede a um modelo juiz uma nota
de 0 a 1 segundo um criterio escrito em portugues. A nota e comparada com a decisao
do limiar de similaridade (0,7) pelo kappa de Cohen.

Recomenda-se um juiz diferente dos seis modelos avaliados, para evitar vies de
autoavaliacao.

Requisitos (ambiente separado ou o mesmo .venv):
    pip install -r requirements-avaliacao.txt
    ollama pull qwen2.5:14b

Uso:
    python -m analyses.llm_judge --judge qwen2.5:14b --per-model 50
    $env:OPENAI_API_KEY = "sk-..."
    python -m analyses.llm_judge --provider openai --judge gpt-4.1 --per-model 50
    python -m analyses.llm_judge --judge qwen2.5:14b --experiments en2 en3 --per-model 30
"""

import argparse
import os
import random
from typing import Any, Dict, List

from analyses.common import (
    EXPERIMENTS,
    OPERATOR_LABELS,
    OUTPUT_DIR,
    THRESHOLD,
    Similarity,
    check_isolation,
    iter_models,
    load_dataset,
    load_predictions,
    write_json,
)
from utils.validates.align_pairs import align_lists
from utils.validates.build_pairs_n2 import build_pairs_n2
from utils.validates.build_pairs_n3 import build_pairs_n3

CRITERIA: Dict[str, str] = {
    "n1": (
        "A SAIDA deve conter a mesma regra normativa da SAIDA ESPERADA. Considere correta a saida que "
        "preserva o mesmo elemento, a mesma exigencia, os mesmos valores e as mesmas condicoes, mesmo que "
        "com outras palavras. Penalize exigencias acrescentadas, omitidas ou alteradas (por exemplo, 'no "
        "maximo' trocado por 'no minimo', ou 'deve' trocado por 'pode')."
    ),
    "n2": (
        "A SAIDA e um trecho extraido da regra da ENTRADA para o operador RASE indicado. Ela deve "
        "corresponder ao mesmo trecho da SAIDA ESPERADA: mesmo papel (aplicabilidade, selecao, excecao ou "
        "requisito), mesmo conteudo e limites equivalentes. Penalize trechos que pertencem a outro "
        "operador, trechos longos ou curtos demais e palavras que nao estao na regra."
    ),
    "n3": (
        "A SAIDA e um registro com os campos type, object, property, comparation, target e unit extraidos "
        "do operador RASE da ENTRADA. Ela deve representar o mesmo objeto, a mesma propriedade, o mesmo "
        "comparador, o mesmo valor e a mesma unidade da SAIDA ESPERADA. Diferencas apenas de grafia ou "
        "acentuacao sao aceitaveis; campos preenchidos que a SAIDA ESPERADA deixa vazios devem ser penalizados."
    ),
}


# Precos (US$ por milhao de tokens: entrada, saida) de modelos ausentes da tabela do DeepEval.
REASONING_PRICES: Dict[str, tuple] = {"gpt-5.6-terra": (2.00, 12.00)}


def _build_judge(judge: str, base_url: str, provider: str):
    from deepeval.metrics import GEval

    try:
        from deepeval.test_case import SingleTurnParams as Params
    except ImportError:
        from deepeval.test_case import LLMTestCaseParams as Params

    if provider == "openai":
        from deepeval.models import GPTModel
        from deepeval.models.llms.constants import OPENAI_MODELS_DATA, make_model_data

        # Modelos de raciocinio recentes (ex.: gpt-5.6-terra) nao constam da tabela do DeepEval 3.9;
        # sem o registro, ele enviaria temperature=0 e logprobs, que esses modelos recusam.
        if judge not in OPENAI_MODELS_DATA and judge.startswith("gpt-5"):
            OPENAI_MODELS_DATA[judge] = make_model_data(
                supports_log_probs=False,
                supports_multimodal=True,
                supports_structured_outputs=True,
                supports_json=False,
                supports_temperature=False,
                input_price=REASONING_PRICES.get(judge, (0.0, 0.0))[0] / 1e6,
                output_price=REASONING_PRICES.get(judge, (0.0, 0.0))[1] / 1e6,
            )
        model = GPTModel(model=judge, temperature=0)
    else:
        from deepeval.models import OllamaModel

        model = OllamaModel(model=judge, base_url=base_url, temperature=0)
    params = [Params.INPUT, Params.ACTUAL_OUTPUT, Params.EXPECTED_OUTPUT]
    return {
        level: GEval(
            name=f"correcao_{level}",
            criteria=criteria,
            evaluation_params=params,
            model=model,
            async_mode=False,
        )
        for level, criteria in CRITERIA.items()
    }


def _samples(experiment_keys: List[str], models: List[str], per_model: int, seed: int) -> List[Dict[str, Any]]:
    dataset = load_dataset()
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []
    for key in experiment_keys:
        experiment = EXPERIMENTS[key]
        for model in models:
            predictions = load_predictions(experiment, model)
            if predictions is None:
                continue
            problem = check_isolation(experiment, dataset, predictions)
            if problem:
                print("AVISO:", problem, f"(modelo {model})")
                continue
            candidates: List[Dict[str, Any]] = []
            if "n1" in experiment.levels and len(experiment.levels) == 1:
                items = predictions.get("datas", [])
                for idx, item in enumerate(dataset["datas"]):
                    ref = [r["text_n1"] for r in item["texts_n1"]]
                    pred = [r.get("text_n1", "") for r in (items[idx] if idx < len(items) else {}).get("texts_n1", [])]
                    for t, p in align_lists(ref, pred):
                        if t is not None and p is not None:
                            candidates.append({"nivel": "n1", "entrada": item["text"], "referencia": ref[t], "previsto": pred[p]})
            if "n2" in experiment.levels and experiment.levels[-1] == "n2":
                for p in build_pairs_n2(dataset, predictions):
                    if p["target"].strip() and p["predicted"].strip():
                        entrada = f"Regra: {p['text_n1']}\nOperador: {OPERATOR_LABELS[p['operator']]}"
                        candidates.append({"nivel": "n2", "entrada": entrada, "referencia": p["target"], "previsto": p["predicted"]})
            if "n3" in experiment.levels:
                for p in build_pairs_n3(dataset, predictions):
                    if any(v for k, v in (p["target_properties"] or {}).items() if k != "type"):
                        entrada = f"Regra: {p['text_n1']}\nOperador: {OPERATOR_LABELS[p['operator']]}"
                        candidates.append({"nivel": "n3", "entrada": entrada, "referencia": p["target"], "previsto": p["predicted"]})
            rng.shuffle(candidates)
            for c in candidates[:per_model]:
                samples.append({**c, "experimento": key, "modelo": model})
    return samples


def _kappa(a: List[int], b: List[int]) -> float | None:
    from sklearn.metrics import cohen_kappa_score

    if len(a) < 2 or len(set(a)) < 2 and len(set(b)) < 2:
        return None
    return float(cohen_kappa_score(a, b))


def run(judge: str, base_url: str, provider: str, experiment_keys: List[str], models: List[str], per_model: int, seed: int) -> Dict[str, Any]:
    from deepeval.test_case import LLMTestCase

    metrics = _build_judge(judge, base_url, provider)
    samples = _samples(experiment_keys, models, per_model, seed)
    sim = Similarity()
    scores = sim.pairs([(s["referencia"], s["previsto"]) for s in samples])

    results: List[Dict[str, Any]] = []
    for i, (sample, similarity) in enumerate(zip(samples, scores), start=1):
        metric = metrics[sample["nivel"]]
        case = LLMTestCase(input=sample["entrada"], actual_output=sample["previsto"], expected_output=sample["referencia"])
        try:
            metric.measure(case)
            score, reason = float(metric.score), metric.reason
        except Exception as exc:
            score, reason = None, f"erro: {exc}"
        results.append({**sample, "similaridade": similarity, "nota_juiz": score, "justificativa": reason})
        print(f"[{i}/{len(samples)}] {sample['experimento']} {sample['nivel']} {sample['modelo']}: {score}", flush=True)

    summary: Dict[str, Any] = {}
    for key in sorted({(r["experimento"], r["nivel"], r["modelo"]) for r in results}):
        rows = [r for r in results if (r["experimento"], r["nivel"], r["modelo"]) == key and r["nota_juiz"] is not None and r["similaridade"] is not None]
        if not rows:
            continue
        judge_bin = [int(r["nota_juiz"] >= 0.5) for r in rows]
        sim_bin = [int(r["similaridade"] >= THRESHOLD) for r in rows]
        summary[":".join(key)] = {
            "n": len(rows),
            "nota_media": sum(r["nota_juiz"] for r in rows) / len(rows),
            "aprovados_juiz": sum(judge_bin) / len(rows),
            "aprovados_limiar": sum(sim_bin) / len(rows),
            "concordancia": sum(int(a == b) for a, b in zip(judge_bin, sim_bin)) / len(rows),
            "kappa_cohen": _kappa(judge_bin, sim_bin),
        }
    output = {"juiz": judge, "provedor": provider, "limiar": THRESHOLD, "resumo": summary, "itens": results}
    write_json(OUTPUT_DIR / "llm_juiz.json", output)
    print(f"Resultado salvo em {OUTPUT_DIR / 'llm_juiz.json'}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["ollama", "openai"], default=os.environ.get("JUDGE_PROVIDER", "ollama"))
    parser.add_argument("--judge", default=os.environ.get("JUDGE_MODEL", "qwen2.5:14b"))
    parser.add_argument("--base-url", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--experiments", nargs="+", default=["en1", "en1n2", "en1n2n3"], choices=list(EXPERIMENTS))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--per-model", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        parser.error("defina OPENAI_API_KEY para usar --provider openai")
    run(args.judge, args.base_url, args.provider, args.experiments, iter_models(args.models), args.per_model, args.seed)


if __name__ == "__main__":
    main()
