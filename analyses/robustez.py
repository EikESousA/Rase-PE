"""Estabilidade entre sementes, ablações de prompt e custo em tokens.

Três estudos complementares sobre os experimentos isolados (EN1, EN2 e EN3):

  - sementes: repete EN1, EN2 e EN3 com as sementes 42 a 46, mantendo prompts,
    temperatura (0,1) e demais opções. A semente 42 é a da execução original,
    e sua repetição mede também a reprodutibilidade.
  - ablacao:  varia um fator do prompt de cada vez, com o N1 ou o N2 de
    referência como entrada:
      N2 sem_exemplo       -> o prompt da dissertação sem o exemplo (zero-shot);
      N2 exemplo_externo   -> o exemplo trocado por uma regra fora do dataset;
      N2 sintaxe_json      -> a saída em JSON em vez de quatro linhas rotuladas;
      N3 ordem_invertida   -> os exemplos de cada prompt N3 em ordem inversa.
  - uso: toda chamada ao Ollama registra os tokens de entrada e de saída e o
    tempo de geração (prompt_eval_count, eval_count, total_duration).

Uso (requer Ollama):
    python -m analyses.robustez generate --study ablacao
    python -m analyses.robustez generate --study sementes --seeds 42 43 44 45 46
    python -m analyses.robustez evaluate
    python -m analyses.robustez tables

Saídas: predicts/robustez/, results/analises/robustez.json e
        dissertacao/Conteudo/Tabelas/{sementes,ablacao_prompt,custo_tokens}.tex
"""

import argparse
from datetime import datetime
import json
import os
import re
import threading
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from analyses.common import OUTPUT_DIR, PROJECT_ROOT, THRESHOLD, iter_models, load_dataset, load_json, write_json

ROBUST_DIR = PROJECT_ROOT / "predicts" / "robustez"
TABLES_DIR = PROJECT_ROOT / "dissertacao" / "Conteudo" / "Tabelas"
ABLATION_PROMPTS = PROJECT_ROOT / "prompts" / "ablacao"
LEVELS = ["n1", "n2", "n3"]
SEEDS = [42, 43, 44, 45, 46]
BOOTSTRAP_SAMPLES = 2000
N3_OPERATOR_FILES = {
    "aplicability": "n3_aplicabilidade",
    "selection": "n3_selecao",
    "exception": "n3_execao",
    "requeriments": "n3_requisito",
}
ABLATIONS: Dict[str, Dict[str, Any]] = {
    "sem_exemplo": {"level": "n2", "prompt": ABLATION_PROMPTS / "n2_sem_exemplo.txt"},
    "exemplo_externo": {"level": "n2", "prompt": ABLATION_PROMPTS / "n2_exemplo_externo.txt"},
    "sintaxe_json": {"level": "n2", "prompt": ABLATION_PROMPTS / "n2_sintaxe_json.txt"},
    "ordem_invertida": {"level": "n3", "prompt": None},
}
MODEL_ORDER = ["llama", "dolphin", "mistral", "gemma", "alpaca", "qwen"]
MODEL_LABELS = {m: m.capitalize() for m in MODEL_ORDER}
PARAMS_B = {"llama": 8.0, "dolphin": 8.0, "mistral": 7.0, "gemma": 4.0, "alpaca": 13.0, "qwen": 0.5}


# ------------------------------------------------------------------ registro de uso

_USAGE: List[Dict[str, Any]] = []
_USAGE_LOCK = threading.Lock()


def _install_usage_hook() -> None:
    """Registra os contadores que o Ollama devolve ao final de cada geração."""
    from langchain_ollama import llms

    original = llms.OllamaLLM._stream_with_aggregation
    if getattr(original, "_robustez_hook", False):
        return

    def wrapped(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        chunk = original(self, *args, **kwargs)
        info = chunk.generation_info or {}
        with _USAGE_LOCK:
            _USAGE.append({
                "prompt_tokens": info.get("prompt_eval_count"),
                "output_tokens": info.get("eval_count"),
                "total_ns": info.get("total_duration"),
                "eval_ns": info.get("eval_duration"),
                "prompt_ns": info.get("prompt_eval_duration"),
                "load_ns": info.get("load_duration"),
            })
        return chunk

    wrapped._robustez_hook = True  # type: ignore[attr-defined]
    llms.OllamaLLM._stream_with_aggregation = wrapped


def _flush_usage(path: Path) -> None:
    with _USAGE_LOCK:
        rows = list(_USAGE)
        _USAGE.clear()
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


# ------------------------------------------------------------------ geração

def seed_path(level: str, model: str, seed: int) -> Path:
    return ROBUST_DIR / "sementes" / f"generate_{level}_{model}_s{seed}.json"


def ablation_path(variant: str, model: str) -> Path:
    level = ABLATIONS[variant]["level"]
    return ROBUST_DIR / "ablacao" / f"generate_{level}_{variant}_{model}.json"


def usage_path(prediction: Path) -> Path:
    return prediction.with_suffix(".uso.jsonl")


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _process_json_or_lines(text: str) -> Dict[str, str]:
    """Parser da variante sintaxe_json: lê o objeto JSON; se falhar, usa o parser de linhas."""
    from utils.n2.normalize_empty import normalize_empty
    from utils.n2.normalize_field_name import normalize_field_name
    from utils.n2.process_text import process_text

    match = _JSON_OBJECT.search(text)
    if match:
        try:
            obj = json.loads(match.group(0))
            out = {k: "" for k in ("aplicabilidade", "selecao", "execao", "requisito")}
            for key, value in obj.items():
                field = normalize_field_name(str(key))
                if field in out:
                    out[field] = normalize_empty(str(value or "").strip())
            return out
        except (json.JSONDecodeError, AttributeError):
            pass
    return process_text(text)


def _run_level(level: str, model: str, output: Path, prompt: Path | None = None, n3_variant: str | None = None) -> None:
    from config.models import MODELS
    import generates.generate_n1 as g1
    import generates.generate_n2 as g2
    import generates.generate_n3 as g3

    dataset = str(PROJECT_ROOT / "dataset.json")
    model_id = MODELS[model]
    output.parent.mkdir(parents=True, exist_ok=True)
    if level == "n1":
        g1.generate_n1(input_path=dataset, output_path=str(output), model_id=model_id)
    elif level == "n2":
        original_parser = g2.process_text
        if prompt is not None and prompt.name == "n2_sintaxe_json.txt":
            g2.process_text = _process_json_or_lines
        try:
            g2.generate_n2(
                input_path=dataset,
                output_path=str(output),
                model_id=model_id,
                prompt_path=prompt or g2.PROMPT_PATH,
            )
        finally:
            g2.process_text = original_parser
    else:
        original_paths = dict(g3.PROMPT_PATHS)
        if n3_variant:
            for key, name in N3_OPERATOR_FILES.items():
                g3.PROMPT_PATHS[key] = ABLATION_PROMPTS / f"{name}_{n3_variant}.txt"
        try:
            g3.generate_n3(input_path=dataset, output_path=str(output), model_id=model_id)
        finally:
            g3.PROMPT_PATHS.clear()
            g3.PROMPT_PATHS.update(original_paths)


def _job(level: str, model: str, output: Path, **kwargs: Any) -> None:
    from config.models import MODELS
    from utils.generates.unload_model import unload_model

    if output.exists() and usage_path(output).exists():
        done = load_json(output)
        if done.get("counts") == len(load_dataset()["datas"]):
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [PULADO] {output.name}: saída completa e uso registrado.", flush=True)
            return
    inicio = datetime.now()
    print(f"[{inicio:%Y-%m-%d %H:%M:%S}] [INÍCIO] {output.name} nível={level} modelo={model}", flush=True)
    _USAGE.clear()
    try:
        _run_level(level, model, output, **kwargs)
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [OK] {output.name} duração={datetime.now() - inicio}", flush=True)
    except Exception as exc:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [ERRO] {output.name}: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        _flush_usage(usage_path(output))
        try:
            unload_model(MODELS[model])
        except Exception:
            pass


def generate(study: str, models: List[str], seeds: List[int], levels: List[str], variants: List[str]) -> None:
    from utils.generates.check_ollama_installed import check_ollama_installed

    if not check_ollama_installed():
        return
    _install_usage_hook()
    os.chdir(PROJECT_ROOT)
    os.environ.setdefault("GENERATE_DEBUG", "0")
    for name in ("N1_TEMP_PAUSE", "N2_TEMP_PAUSE", "N3_TEMP_PAUSE"):
        os.environ[name] = "0"

    if study == "ablacao":
        for variant in variants:
            cfg = ABLATIONS[variant]
            for model in models:
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [PROGRESSO] ablação variante={variant} nível={cfg['level']} modelo={model}", flush=True)
                if cfg["level"] == "n2":
                    _job("n2", model, ablation_path(variant, model), prompt=cfg["prompt"])
                else:
                    _job("n3", model, ablation_path(variant, model), n3_variant=variant)
    else:
        for seed in seeds:
            os.environ["GEN_SEED"] = str(seed)
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [PROGRESSO] sementes semente={seed}/{seeds[-1]} níveis={','.join(levels)} modelos={','.join(models)}", flush=True)
            for level in levels:
                for model in models:
                    _job(level, model, seed_path(level, model, seed))
        os.environ["GEN_SEED"] = "42"


# ------------------------------------------------------------------ avaliação

class _Scorer:
    """Pares, similaridade Multilingual e acerto por item, como na validação principal."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        from utils.validates.build_pairs import build_pairs
        from utils.validates.build_pairs_n2 import build_pairs_n2
        from utils.validates.build_pairs_n3 import build_pairs_n3

        self.dataset = load_dataset()
        self.builders = {"n1": build_pairs, "n2": build_pairs_n2, "n3": build_pairs_n3}
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    def evaluate(self, level: str, predictions: Dict[str, Any]) -> Dict[str, Any]:
        from utils.validates.compute_multilingual_scores import compute_multilingual_scores
        from utils.validates.run_validation import compute_classification_metrics

        pairs = self.builders[level](self.dataset, predictions)
        targets = [p["target"] for p in pairs]
        predicted = [p["predicted"] for p in pairs]
        sims = compute_multilingual_scores(self.model, targets, predicted) if level != "n3" else [None] * len(pairs)
        data = {"x": {"pairs": pairs, "targets": targets, "predicted": predicted, "scores": {"multilingual": sims}}}
        cls = compute_classification_metrics(data, level)["x"]
        main = cls["overall"] if level == "n1" else cls["macro"]
        return {"f1": main["f1"], "classification": cls, "items": _item_outcomes(level, pairs, sims)}


def _item_outcomes(level: str, pairs: List[Dict[str, Any]], sims: List[float | None]) -> List[Dict[str, Any]]:
    """Desfecho de cada item avaliado, com o índice da norma para os testes pareados."""
    from utils.validates.compute_classification_scores import compute_confusion_from_exact

    out: List[Dict[str, Any]] = []
    if level == "n3":
        for i, p in enumerate(pairs):
            conf = compute_confusion_from_exact([p.get("target_properties", {})], [p.get("predicted_properties", {})])
            for field, c in conf.items():
                out.append({"norma": p.get("text_index"), "chave": f"{i}|{field}",
                            "grupo": field, "vp": c.get("tp", c.get("vp", 0)), "fp": c.get("fp", 0), "fn": c.get("fn", 0)})
        return out
    for i, p in enumerate(pairs):
        t, q, s = p["target"].strip(), p["predicted"].strip(), sims[i]
        vp = int(bool(t and q and s is not None and s >= THRESHOLD))
        fp = int(bool(q and not vp))
        fn = int(bool(t and not vp))
        key = str(i)
        out.append({"norma": p.get("text_index"), "chave": key, "grupo": p.get("operator", "regra"), "vp": vp, "fp": fp, "fn": fn})
    return out


def _f1(items: List[Dict[str, Any]], level: str, idx: np.ndarray | None = None) -> float:
    rows = items if idx is None else [items[i] for i in idx]
    groups: Dict[str, Dict[str, int]] = {}
    for it in rows:
        g = groups.setdefault(it["grupo"], {"vp": 0, "fp": 0, "fn": 0})
        for k in ("vp", "fp", "fn"):
            g[k] += it[k]

    def f1(c: Dict[str, int]) -> float:
        p = c["vp"] / (c["vp"] + c["fp"]) if c["vp"] + c["fp"] else 0.0
        r = c["vp"] / (c["vp"] + c["fn"]) if c["vp"] + c["fn"] else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    if level == "n1":
        total = {k: sum(g[k] for g in groups.values()) for k in ("vp", "fp", "fn")}
        return f1(total)
    return float(np.mean([f1(g) for g in groups.values()])) if groups else 0.0


def _correct(items: List[Dict[str, Any]]) -> Dict[str, int]:
    return {it["chave"]: int(it["fp"] == 0 and it["fn"] == 0) for it in items}


def _mcnemar(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, Any]:
    from scipy.stats import binomtest

    keys = sorted(set(a) & set(b))
    only_a = sum(1 for k in keys if a[k] and not b[k])
    only_b = sum(1 for k in keys if b[k] and not a[k])
    n = only_a + only_b
    return {"so_base_acerta": only_a, "so_variante_acerta": only_b, "p_valor": float(binomtest(only_a, n, 0.5).pvalue) if n else 1.0}


def _paired_bootstrap(base: List[Dict[str, Any]], variant: List[Dict[str, Any]], level: str, rng: np.random.Generator) -> List[float]:
    """IC 95% da diferença de F1 (variante - base), reamostrando normas."""
    norms = sorted({it["norma"] for it in base})
    by_norm_a = {n: [i for i, it in enumerate(base) if it["norma"] == n] for n in norms}
    by_norm_b = {n: [i for i, it in enumerate(variant) if it["norma"] == n] for n in norms}
    diffs = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = rng.choice(norms, len(norms))
        ia = np.array([i for n in sample for i in by_norm_a[n]], dtype=int)
        ib = np.array([i for n in sample for i in by_norm_b.get(n, [])], dtype=int)
        diffs.append(_f1(variant, level, ib) - _f1(base, level, ia))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def _per_norm_score(items: List[Dict[str, Any]]) -> Dict[Any, float]:
    out: Dict[Any, List[int]] = {}
    for it in items:
        out.setdefault(it["norma"], []).append(int(it["fp"] == 0 and it["fn"] == 0))
    return {n: float(np.mean(v)) for n, v in out.items()}


def _usage_summary(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("prompt_tokens") is not None]
    if not rows:
        return None
    pt = np.array([r["prompt_tokens"] for r in rows], dtype=float)
    ot = np.array([r["output_tokens"] or 0 for r in rows], dtype=float)
    total_s = sum((r["total_ns"] or 0) - (r.get("load_ns") or 0) for r in rows) / 1e9
    eval_s = sum(r["eval_ns"] or 0 for r in rows) / 1e9
    return {
        "chamadas": len(rows),
        "tokens_entrada_media": float(pt.mean()),
        "tokens_saida_media": float(ot.mean()),
        "tokens_entrada_total": int(pt.sum()),
        "tokens_saida_total": int(ot.sum()),
        "tempo_geracao_s": float(total_s),
        "tokens_saida_por_s": float(ot.sum() / eval_s) if eval_s else None,
    }


def evaluate(models: List[str]) -> Dict[str, Any]:
    from scipy.stats import wilcoxon

    from analyses.common import check_isolation, EXPERIMENTS

    scorer = _Scorer()
    rng = np.random.default_rng(42)
    result: Dict[str, Any] = {"limiar": THRESHOLD, "sementes": {}, "ablacao": {}, "uso": {}, "comparacoes_hipoteses": {}, "avisos": []}
    seed_items: Dict[Tuple[str, str], Dict[int, List[Dict[str, Any]]]] = {}

    # sementes
    for level in LEVELS:
        result["sementes"][level] = {}
        for model in models:
            per_seed: Dict[int, float] = {}
            for seed in SEEDS:
                path = seed_path(level, model, seed)
                if not path.exists():
                    continue
                preds = load_json(path)
                problem = check_isolation(EXPERIMENTS[f"en{level[1]}"], scorer.dataset, preds)
                if problem:
                    result["avisos"].append(f"{path.name}: {problem}")
                    continue
                ev = scorer.evaluate(level, preds)
                per_seed[seed] = ev["f1"]
                seed_items.setdefault((level, model), {})[seed] = ev["items"]
            if not per_seed:
                continue
            vals = np.array(list(per_seed.values()))
            entry: Dict[str, Any] = {
                "f1_por_semente": per_seed,
                "media": float(vals.mean()),
                "desvio": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
            original = PROJECT_ROOT / "predicts" / f"generate_{level}_{model}.json"
            if 42 in per_seed and original.exists():
                ev0 = scorer.evaluate(level, load_json(original))
                rerun = load_json(seed_path(level, model, 42))
                same, total = _identical_outputs(level, load_json(original), rerun)
                entry["f1_original"] = ev0["f1"]
                entry["reproducao_s42"] = {"saidas_identicas": same, "saidas": total}
            usages = [_usage_summary(usage_path(seed_path(level, model, s))) for s in per_seed]
            usages = [u for u in usages if u]
            if usages:
                entry["uso_medio"] = {k: float(np.mean([u[k] for u in usages if u.get(k) is not None])) for k in usages[0]}
            result["sementes"][level][model] = entry

    # comparações que sustentam as hipóteses sobre a seleção de modelos
    # (média por norma sobre as sementes; teste de Wilcoxon pareado por norma)
    pairs_to_test = [("llama", "dolphin"), ("llama", "mistral"), ("dolphin", "mistral"), ("gemma", "qwen"), ("llama", "alpaca"), ("gemma", "llama")]
    for level in LEVELS:
        result["comparacoes_hipoteses"][level] = {}
        for a, b in pairs_to_test:
            if (level, a) not in seed_items or (level, b) not in seed_items:
                continue
            sa = [_per_norm_score(v) for v in seed_items[(level, a)].values()]
            sb = [_per_norm_score(v) for v in seed_items[(level, b)].values()]
            norms = sorted(set.intersection(*[set(d) for d in sa + sb]))
            xa = np.array([np.mean([d[n] for d in sa]) for n in norms])
            xb = np.array([np.mean([d[n] for d in sb]) for n in norms])
            diff = xb - xa
            try:
                p = float(wilcoxon(xa, xb, zero_method="wilcox").pvalue) if np.any(diff != 0) else 1.0
            except ValueError:
                p = 1.0
            result["comparacoes_hipoteses"][level][f"{a}_vs_{b}"] = {
                "normas": len(norms),
                "acerto_medio_por_norma": [float(xa.mean()), float(xb.mean())],
                "normas_melhores": [int((diff < 0).sum()), int((diff > 0).sum())],
                "wilcoxon_p": p,
            }

    # ablações
    for variant, cfg in ABLATIONS.items():
        level = cfg["level"]
        result["ablacao"][variant] = {}
        for model in models:
            path = ablation_path(variant, model)
            if not path.exists():
                continue
            preds = load_json(path)
            problem = check_isolation(EXPERIMENTS[f"en{level[1]}"], scorer.dataset, preds)
            if problem:
                result["avisos"].append(f"{path.name}: {problem}")
                continue
            base_items = seed_items.get((level, model), {}).get(42)
            if base_items is None:
                base_items = scorer.evaluate(level, load_json(PROJECT_ROOT / "predicts" / f"generate_{level}_{model}.json"))["items"]
            ev = scorer.evaluate(level, preds)
            result["ablacao"][variant][model] = {
                "f1_base": _f1(base_items, level),
                "f1_variante": ev["f1"],
                "diferenca": ev["f1"] - _f1(base_items, level),
                "ic95_diferenca": _paired_bootstrap(base_items, ev["items"], level, rng),
                "mcnemar": _mcnemar(_correct(base_items), _correct(ev["items"])),
                "por_grupo": {g: m["f1"] for g, m in (ev["classification"].get("by_operator") or ev["classification"].get("by_field") or {}).items()},
                "uso": _usage_summary(usage_path(path)),
                "falhas_formato": _format_failures(variant, preds) if variant == "sintaxe_json" else None,
            }

    for w in result["avisos"]:
        print("AVISO:", w)
    write_json(OUTPUT_DIR / "robustez.json", result)
    print(f"Resultado salvo em {OUTPUT_DIR / 'robustez.json'}")
    return result


def _identical_outputs(level: str, a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[int, int]:
    def flat(d: Dict[str, Any]) -> List[str]:
        out = []
        for item in d.get("datas", []):
            if level == "n1":
                out.append(json.dumps([r.get("text_n1") for r in item.get("texts_n1", [])], ensure_ascii=False))
            else:
                for rule in item.get("texts_n1", []):
                    out.append(json.dumps(rule.get("operators_n2"), ensure_ascii=False, sort_keys=True))
        return out

    fa, fb = flat(a), flat(b)
    return sum(1 for x, y in zip(fa, fb) if x == y), max(len(fa), len(fb))


def _format_failures(variant: str, preds: Dict[str, Any]) -> Dict[str, int]:
    rules = [r for item in preds.get("datas", []) for r in item.get("texts_n1", [])]
    no_req = sum(1 for r in rules if not (r.get("operators_n2", {}).get("requeriments", {}) or {}).get("text_n2", "").strip())
    return {"regras": len(rules), "sem_requisito": no_req}


# ------------------------------------------------------------------ tabelas

def _fmt(value: float | None, digits: int = 3) -> str:
    return "---" if value is None else f"{value:.{digits}f}".replace(".", "{,}")


def _p(value: float) -> str:
    return "$<$0{,}001" if value < 0.001 else _fmt(value)


def tables() -> None:
    res = load_json(OUTPUT_DIR / "robustez.json")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1) estabilidade entre sementes
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{F1 principal em cinco execuções de cada experimento isolado (sementes 42 a 46): média, desvio-padrão amostral e amplitude. "
        r"N1: F1 sobre as regras; N2: média macro sobre os quatro operadores; N3: média macro sobre os cinco campos. "
        r"A coluna ``Orig.'' traz o F1 da execução original (semente 42) usada no restante do capítulo.}",
        r"\label{tab:sementes}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l" + "rrr" * 3 + "}",
        r"\hline",
        r" & \multicolumn{3}{c}{EN1} & \multicolumn{3}{c}{EN2} & \multicolumn{3}{c}{EN3} \\",
        r"Modelo & Média $\pm$ DP & Ampl. & Orig. & Média $\pm$ DP & Ampl. & Orig. & Média $\pm$ DP & Ampl. & Orig. \\",
        r"\hline",
    ]
    for model in MODEL_ORDER:
        cells = [MODEL_LABELS[model]]
        for level in LEVELS:
            e = res["sementes"].get(level, {}).get(model)
            if not e:
                cells += ["---"] * 3
                continue
            cells += [f"{_fmt(e['media'])} $\\pm$ {_fmt(e['desvio'])}", _fmt(e["max"] - e["min"]), _fmt(e.get("f1_original"))]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (TABLES_DIR / "sementes.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 2) ablações de prompt
    labels = {
        "sem_exemplo": "N2 sem exemplo (\\textit{zero-shot})",
        "exemplo_externo": "N2 com exemplo externo ao \\textit{dataset}",
        "sintaxe_json": "N2 com saída em JSON",
        "ordem_invertida": "N3 com exemplos em ordem inversa",
    }
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Ablações de \textit{prompt}: diferença de F1 macro entre cada variante e o \textit{prompt} da dissertação (variante $-$ original), "
        r"com intervalo de confiança de 95\% por \textit{bootstrap} pareado sobre as normas. Em negrito, as diferenças cujo intervalo não contém o zero.}",
        r"\label{tab:ablacao_prompt}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{>{\raggedright}p{3.6cm}" + "r" * len(MODEL_ORDER) + "}",
        r"\hline",
        "Variante & " + " & ".join(MODEL_LABELS[m] for m in MODEL_ORDER) + r" \\",
        r"\hline",
    ]
    for variant, label in labels.items():
        cells = [label]
        for model in MODEL_ORDER:
            e = res["ablacao"].get(variant, {}).get(model)
            if not e:
                cells.append("---")
                continue
            lo, hi = e["ic95_diferenca"]
            text = ("+" if e["diferenca"] > 0 else "") + _fmt(e["diferenca"])
            text = text.replace("+", "$+$").replace("-", "$-$")
            cells.append(r"\textbf{" + text + "}" if lo > 0 or hi < 0 else text)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (TABLES_DIR / "ablacao_prompt.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 3) custo em tokens e desempenho
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Custo por chamada e por experimento, médio sobre as cinco sementes: \textit{tokens} de entrada e de saída por chamada, "
        r"tempo de geração do experimento (sem a carga do modelo) e F1 médio. Medidas registradas pelo Ollama em cada chamada.}",
        r"\label{tab:custo_tokens}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        r"Exp. & Modelo & Parâm. & \textit{Tokens} entrada & \textit{Tokens} saída & Tempo (s) & F1 \\",
        r"\hline",
    ]
    for level in LEVELS:
        for model in MODEL_ORDER:
            e = res["sementes"].get(level, {}).get(model)
            if not e or "uso_medio" not in e:
                continue
            u = e["uso_medio"]
            lines.append(" & ".join([
                f"EN{level[1]}", MODEL_LABELS[model], _fmt(PARAMS_B[model], 1).replace("{,}0", "") + "B",
                f"{u['tokens_entrada_media']:.0f}", f"{u['tokens_saida_media']:.0f}",
                f"{u['tempo_geracao_s']:.0f}", _fmt(e["media"]),
            ]) + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}"]
    (TABLES_DIR / "custo_tokens.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Tabelas escritas em", TABLES_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--study", choices=["sementes", "ablacao"], required=True)
    g.add_argument("--models", nargs="+", default=None)
    g.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    g.add_argument("--levels", nargs="+", default=LEVELS, choices=LEVELS)
    g.add_argument("--variants", nargs="+", default=list(ABLATIONS), choices=list(ABLATIONS))
    e = sub.add_parser("evaluate")
    e.add_argument("--models", nargs="+", default=None)
    sub.add_parser("tables")
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.study, iter_models(args.models), args.seeds, args.levels, args.variants)
    elif args.command == "evaluate":
        evaluate(iter_models(args.models))
    else:
        tables()


if __name__ == "__main__":
    main()
