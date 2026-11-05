"""Sensibilidade ao limiar de similaridade (0,1 a 0,9) e curva ROC/AUC.

Parte 1 - varredura: recalcula VP, FP, FN, VN, precisao, sensibilidade (revocacao),
especificidade, F1 e acuracia para cada limiar, com a mesma convencao da validacao.

Parte 2 - ROC: nos pares em que referencia e saida estao preenchidas, a similaridade
Multilingual e tratada como escore de um classificador "o par esta correto?". O
gabarito vem de um criterio independente dos embeddings:
  - estrito: texto identico apos normalizacao;
  - relaxado: F1 de tokens >= 0,5 (criterio de sobreposicao usado no SQuAD);
  - manual: rotulos humanos em results/analises/rotulos_manuais.csv, se existir.

Uso:
    python -m analyses.threshold_analysis
    python -m analyses.threshold_analysis --export-sample 300
"""

import argparse
import csv
import random
from typing import Any, Dict, List, Tuple

import numpy as np

from analyses.common import (
    EXPERIMENTS,
    OUTPUT_DIR,
    PROJECT_ROOT,
    Similarity,
    check_isolation,
    exclude_contaminated,
    iter_models,
    load_dataset,
    load_predictions,
    normalize,
    token_f1,
    write_json,
)
from utils.validates.align_pairs import align_lists
from utils.validates.build_pairs_n2 import build_pairs_n2

THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]
RELAXED_F1 = 0.5
MANUAL_LABELS = OUTPUT_DIR / "rotulos_manuais.csv"
FIGURES_DIR = PROJECT_ROOT / "dissertacao" / "Imagens"


def _pairs_n1(dataset: Dict[str, Any], predictions: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    predicted_items = predictions.get("datas", [])
    for idx, item in enumerate(dataset.get("datas", [])):
        ref = [r.get("text_n1", "") for r in item.get("texts_n1", [])]
        pred_item = predicted_items[idx] if idx < len(predicted_items) else {}
        pred = [r.get("text_n1", "") for r in pred_item.get("texts_n1", [])]
        for t, p in align_lists(ref, pred):
            pairs.append((ref[t] if t is not None else "", pred[p] if p is not None else ""))
    return pairs


def _pairs_n2(dataset: Dict[str, Any], predictions: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    return [(p["target"], p["predicted"], p["operator"]) for p in build_pairs_n2(dataset, predictions)]


def _confusion(records: List[Dict[str, Any]], threshold: float) -> Dict[str, int]:
    vp = fp = fn = vn = 0
    for r in records:
        t, q, s = r["target"].strip(), r["predicted"].strip(), r["sim"]
        if t and q:
            if s is not None and s >= threshold:
                vp += 1
            else:
                fp += 1
                fn += 1
        elif t:
            fn += 1
        elif q:
            fp += 1
        else:
            vn += 1
    return {"vp": vp, "fp": fp, "fn": fn, "vn": vn}


def _div(a: float, b: float) -> float | None:
    return a / b if b > 0 else None


def _rates(c: Dict[str, int], has_negatives: bool) -> Dict[str, float | None]:
    precision = _div(c["vp"], c["vp"] + c["fp"])
    recall = _div(c["vp"], c["vp"] + c["fn"])
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precisao": precision,
        "sensibilidade": recall,
        "especificidade": _div(c["vn"], c["vn"] + c["fp"]) if has_negatives else None,
        "f1": f1,
        "acuracia": _div(c["vp"] + c["vn"], sum(c.values())),
    }


def _gold(target: str, predicted: str) -> Dict[str, int]:
    return {
        "estrito": int(normalize(target) == normalize(predicted)),
        "relaxado": int(token_f1(target, predicted) >= RELAXED_F1),
    }


def _manual_key(record: Dict[str, Any]) -> Tuple[str, ...]:
    return (record["experimento"], record["nivel"], record["modelo"], record["operador"], record["target"], record["predicted"])


def _load_manual() -> Dict[Tuple[str, ...], int]:
    """Rotulos humanos indexados pelo conteudo do par, nao pelo id.

    O id da amostra e posicional (experimento-nivel-modelo-indice) e deixa de apontar
    para o mesmo par sempre que a base de registros muda, como em --sem-contaminacao.
    """
    if not MANUAL_LABELS.exists():
        return {}
    labels: Dict[Tuple[str, ...], int] = {}
    with open(MANUAL_LABELS, "r", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file, delimiter=";"):
            value = (row.get("correto") or "").strip()
            if value not in {"0", "1"}:
                continue
            prefix, _index = row["id"].rsplit("-", 1)
            prefix, model = prefix.rsplit("-", 1)
            experiment, level = prefix.rsplit("-", 1)
            labels[(experiment, level, model, row["operador"], row["referencia"], row["previsto"])] = int(value)
    return labels


def _labeled(records: List[Dict[str, Any]], manual: Dict[Tuple[str, ...], int]) -> List[Tuple[Dict[str, Any], int]]:
    """Casa cada rotulo com o primeiro registro de mesmo conteudo, sem repetir pares."""
    seen = set()
    pairs = []
    for record in records:
        key = _manual_key(record)
        if key in manual and key not in seen:
            seen.add(key)
            pairs.append((record, manual[key]))
    return pairs


def _roc(scores: List[float], labels: List[int]) -> Dict[str, Any] | None:
    from sklearn.metrics import roc_auc_score, roc_curve

    if len(set(labels)) < 2:
        return None
    y = np.array(labels)
    s = np.array(scores)
    fpr, tpr, thr = roc_curve(y, s)
    youden = tpr - fpr
    best = int(np.argmax(youden))
    at_thresholds = {}
    for t in THRESHOLDS:
        pred = s >= t
        tp = int(np.sum(pred & (y == 1)))
        fn = int(np.sum(~pred & (y == 1)))
        tn = int(np.sum(~pred & (y == 0)))
        fp = int(np.sum(pred & (y == 0)))
        at_thresholds[str(t)] = {"sensibilidade": _div(tp, tp + fn), "especificidade": _div(tn, tn + fp)}
    return {
        "n": len(labels),
        "positivos": int(y.sum()),
        "auc": float(roc_auc_score(y, s)),
        "limiar_youden": float(min(thr[best], 1.0)),
        "sensibilidade_youden": float(tpr[best]),
        "especificidade_youden": float(1 - fpr[best]),
        "por_limiar": at_thresholds,
        "curva": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


def collect(experiment_keys: List[str], models: List[str], sim: Similarity) -> Tuple[List[Dict[str, Any]], List[str]]:
    dataset = load_dataset()
    records: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for key in experiment_keys:
        experiment = EXPERIMENTS[key]
        for model in models:
            predictions = load_predictions(experiment, model)
            if predictions is None:
                continue
            problem = check_isolation(experiment, dataset, predictions)
            if problem:
                warnings.append(f"{problem} (modelo {model})")
                continue
            if "n1" in experiment.levels:
                pairs = _pairs_n1(dataset, predictions)
                scores = sim.pairs(pairs)
                for i, ((t, q), s) in enumerate(zip(pairs, scores)):
                    records.append({"id": f"{key}-n1-{model}-{i}", "experimento": key, "nivel": "n1", "modelo": model, "operador": "", "target": t, "predicted": q, "sim": s})
            if "n2" in experiment.levels:
                pairs2 = _pairs_n2(dataset, predictions)
                scores = sim.pairs([(t, q) for t, q, _ in pairs2])
                for i, ((t, q, op), s) in enumerate(zip(pairs2, scores)):
                    records.append({"id": f"{key}-n2-{model}-{i}", "experimento": key, "nivel": "n2", "modelo": model, "operador": op, "target": t, "predicted": q, "sim": s})
    return records, warnings


def analyze(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    manual = _load_manual()
    output: Dict[str, Any] = {"limiares": THRESHOLDS, "varredura": {}, "roc": {}}
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["experimento"], r["nivel"]), []).append(r)

    for (exp, level), recs in groups.items():
        key = f"{exp}:{level}"
        has_neg = level != "n1"
        sweep: Dict[str, Any] = {}
        for model in sorted({r["modelo"] for r in recs}):
            mrecs = [r for r in recs if r["modelo"] == model]
            sweep[model] = {}
            for t in THRESHOLDS:
                c = _confusion(mrecs, t)
                entry: Dict[str, Any] = {**c, **_rates(c, has_neg)}
                if level == "n2":
                    per_op = []
                    for op in sorted({r["operador"] for r in mrecs}):
                        rates = _rates(_confusion([r for r in mrecs if r["operador"] == op], t), True)
                        per_op.append(rates["f1"] or 0.0)
                    entry["f1_macro_operadores"] = sum(per_op) / len(per_op) if per_op else None
                sweep[model][str(t)] = entry
        output["varredura"][key] = sweep

        filled = [r for r in recs if r["target"].strip() and r["predicted"].strip() and r["sim"] is not None]
        roc_out: Dict[str, Any] = {}
        for gold_name in ("estrito", "relaxado"):
            labels = [_gold(r["target"], r["predicted"])[gold_name] for r in filled]
            pooled = _roc([r["sim"] for r in filled], labels)
            per_model = {}
            for model in sorted({r["modelo"] for r in filled}):
                sub = [(r["sim"], lab) for r, lab in zip(filled, labels) if r["modelo"] == model]
                res = _roc([s for s, _ in sub], [lab for _, lab in sub])
                per_model[model] = {k: res[k] for k in ("n", "positivos", "auc")} if res else None
            roc_out[gold_name] = {"agregado": pooled, "por_modelo": per_model}
        labeled = _labeled(filled, manual)
        if labeled:
            roc_out["manual"] = {"agregado": _roc([r["sim"] for r, _ in labeled], [lab for _, lab in labeled])}
        output["roc"][key] = roc_out

    filled_all = [r for r in records if r["target"].strip() and r["predicted"].strip() and r["sim"] is not None]
    labeled_all = _labeled(filled_all, manual)
    if labeled_all:
        output["roc_manual"] = {}
        for level in ("n1", "n2", ""):
            sub = [(r, lab) for r, lab in labeled_all if not level or r["nivel"] == level]
            res = _roc([r["sim"] for r, _ in sub], [lab for _, lab in sub]) if sub else None
            if res:
                output["roc_manual"][level or "geral"] = res
    return output


def export_sample(records: List[Dict[str, Any]], size: int, seed: int = 42) -> None:
    filled = [r for r in records if r["target"].strip() and r["predicted"].strip() and r["sim"] is not None]
    rng = random.Random(seed)
    bins: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in filled:
        bins.setdefault((r["nivel"], min(int(r["sim"] * 10), 9)), []).append(r)
    per_bin = max(1, size // max(1, len(bins)))
    sample: List[Dict[str, Any]] = []
    for key in sorted(bins):
        items = bins[key]
        rng.shuffle(items)
        sample.extend(items[:per_bin])
    rng.shuffle(sample)
    path = OUTPUT_DIR / "amostra_para_rotulagem.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["id", "nivel", "operador", "referencia", "previsto", "correto"])
        for r in sample:
            writer.writerow([r["id"], r["nivel"], r["operador"], r["target"], r["predicted"], ""])
    print(f"Amostra com {len(sample)} pares salva em {path}. Preencha 'correto' com 1 ou 0 e salve como {MANUAL_LABELS.name} (separador ';').")


def plot(output: Dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    keys = [k for k in ("en1:n1", "en1n2:n2") if k in output["roc"]]
    if keys:
        fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.6), squeeze=False)
        titles = {"en1:n1": "N1 (EN1)", "en1n2:n2": "N2 (EN1N2)"}
        for ax, key in zip(axes[0], keys):
            level = key.split(":")[1]
            curves = [
                ("-", "#1A73E8", output["roc"][key]["relaxado"]["agregado"], "gabarito relaxado"),
                ("--", "#E8710A", output["roc"][key]["estrito"]["agregado"], "gabarito estrito"),
                ("-.", "#188038", output.get("roc_manual", {}).get(level), "gabarito manual (todos os experimentos)"),
            ]
            for style, color, res, name in curves:
                if res:
                    ax.plot(res["curva"]["fpr"], res["curva"]["tpr"], style, color=color,
                            label=f"{name} (AUC = {res['auc']:.3f})".replace(".", ","))
            ax.plot([0, 1], [0, 1], ":", color="#9AA0A6")
            ax.set_title(titles[key])
            ax.set_xlabel("1 − especificidade")
            ax.set_ylabel("sensibilidade")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "5.roc-limiar.png", dpi=200)
        plt.close(fig)

    sweep_keys = [k for k in ("en1:n1", "en1n2:n2") if k in output["varredura"]]
    if sweep_keys:
        fig, axes = plt.subplots(1, len(sweep_keys), figsize=(5.2 * len(sweep_keys), 4.2), squeeze=False)
        for ax, key in zip(axes[0], sweep_keys):
            metric = "f1" if key.endswith("n1") else "f1_macro_operadores"
            for model, values in output["varredura"][key].items():
                ax.plot(THRESHOLDS, [values[str(t)][metric] or 0 for t in THRESHOLDS], marker="o", markersize=3, label=model.capitalize())
            ax.axvline(0.7, color="#D93025", linestyle="--", linewidth=1)
            ax.set_title(("F1 em N1 (EN1)" if key.endswith("n1") else "F1 macro dos operadores em N2 (EN1N2)"))
            ax.set_xlabel("limiar de similaridade")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "5.varredura-limiar.png", dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--export-sample", type=int, default=0, help="Exporta N pares para rotulagem manual.")
    parser.add_argument("--sem-contaminacao", action="store_true", help="Exclui as normas usadas como exemplo nos prompts.")
    args = parser.parse_args()

    suffix = exclude_contaminated(args.sem_contaminacao)
    sim = Similarity()
    records, warnings = collect(args.experiments, iter_models(args.models), sim)
    for w in warnings:
        print("AVISO:", w)
    if args.export_sample:
        export_sample(records, args.export_sample)
    output = analyze(records)
    output["avisos"] = warnings
    path = OUTPUT_DIR / f"limiar_roc{suffix}.json"
    write_json(path, output)
    if not suffix:
        plot(output)
    print(f"Resultado salvo em {path}")


if __name__ == "__main__":
    main()
