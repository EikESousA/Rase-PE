"""Redundância e poder discriminante das métricas de similaridade.

Responde a três perguntas sobre o conjunto de métricas da validação:
  1. Quais métricas medem a mesma coisa? Correlação de Spearman, item a item,
     entre as nove similaridades, nos pares em que saída e referência estão
     preenchidas (N1 e N2) e em todos os pares (N3).
  2. Elas ordenam os modelos da mesma forma? Tau de Kendall entre a ordenação
     dos seis modelos por cada similaridade e a ordenação pelo F1.
  3. Quais distinguem um par correto de um incorreto? AUC de cada métrica
     contra os 264 pares rotulados à mão (results/analises/rotulos_manuais.csv).
Inclui o BM25 (Okapi, k1=1,2 e b=0,75), normalizado para [0,1], como
alternativa ao TF-IDF, para verificar se a troca mudaria alguma conclusão.

Uso:
    python -m analyses.redundancia_metricas
Saída: results/analises/redundancia_metricas.json e
       dissertacao/Conteudo/Tabelas/redundancia_metricas.tex
"""

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

from analyses.common import OUTPUT_DIR, PROJECT_ROOT, normalize, write_json

METRICS = ["fuzzywuzzy", "tfidf", "sbert", "bertimbau", "multilingual", "wmd_ft", "wmd_nilc", "bertscore", "rouge_l"]
LABELS = {
    "fuzzywuzzy": "FuzzyWuzzy", "tfidf": "TF-IDF", "bm25": "BM25", "sbert": "SBERT (PT)", "bertimbau": "BERTimbau",
    "multilingual": "Multilingual", "wmd_ft": "WMD\\_ft", "wmd_nilc": "WMD\\_nilc", "bertscore": "BERTScore", "rouge_l": "ROUGE-L",
}
MODELS = ["llama", "dolphin", "mistral", "gemma", "alpaca", "qwen"]
DETAILS = PROJECT_ROOT / "metrics" / "details"
TABLES_DIR = PROJECT_ROOT / "dissertacao" / "Conteudo" / "Tabelas"


# ------------------------------------------------------------------ BM25

class BM25:
    def __init__(self, corpus: List[str], k1: float = 1.2, b: float = 0.75) -> None:
        docs = [normalize(d).split() for d in corpus if d.strip()]
        self.k1, self.b = k1, b
        self.avgdl = float(np.mean([len(d) for d in docs])) if docs else 1.0
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.default_idf = math.log(1 + (n + 0.5) / 0.5)

    def _score(self, query: List[str], doc: List[str]) -> float:
        tf = Counter(doc)
        norm = self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
        return sum(self.idf.get(t, self.default_idf) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in query if t in tf)

    def similarity(self, a: str, b: str) -> float | None:
        """Média dos dois sentidos, cada um dividido pelo escore do documento contra si mesmo."""
        ta, tb = normalize(a).split(), normalize(b).split()
        if not ta or not tb:
            return None
        ab = self._score(ta, tb) / (self._score(tb, tb) or 1.0)
        ba = self._score(tb, ta) / (self._score(ta, ta) or 1.0)
        return float(min(1.0, (ab + ba) / 2))


# ------------------------------------------------------------------ dados

def _details(level: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model in MODELS:
        path = DETAILS / f"validate_{level}_{model}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                item["model"] = model
                rows.append(item)
    return rows


def _filled(rows: List[Dict[str, Any]], level: str) -> List[Dict[str, Any]]:
    if level == "n3":
        return [r for r in rows if all(r.get(m) is not None for m in METRICS)]
    return [r for r in rows if str(r.get("target", "")).strip() and str(r.get("predicted", "")).strip()
            and all(r.get(m) is not None for m in METRICS)]


def correlations(level: str) -> Dict[str, Any]:
    rows = _filled(_details(level), level)
    bm25 = BM25([r["target"] for r in rows])
    for r in rows:
        r["bm25"] = bm25.similarity(r["target"], r["predicted"]) or 0.0
    names = METRICS + ["bm25"]
    mat = np.array([[r[m] for m in names] for r in rows], dtype=float)
    rho = spearmanr(mat).correlation
    return {"pares": len(rows), "metricas": names, "spearman": [[float(x) for x in row] for row in rho]}


def ranking_agreement(level: str) -> Dict[str, Any]:
    with open(PROJECT_ROOT / "metrics" / f"validate_{level}.json", encoding="utf-8") as file:
        data = json.load(file)
    f1 = []
    for model in MODELS:
        cls = data["models"][model]["classification"]
        f1.append((cls.get("overall") if level == "n1" else cls.get("macro"))["f1"])
    out = {}
    for m in METRICS:
        values = [data["averages"][model][m] for model in MODELS]
        out[m] = float(kendalltau(values, f1).statistic)
    return out


def manual_auc() -> Dict[str, Any]:
    """Calcula as nove similaridades (e o BM25) sobre os pares rotulados à mão."""
    import utils.validates.run_validation as rv

    with open(PROJECT_ROOT / "results" / "analises" / "rotulos_manuais.csv", encoding="utf-8-sig") as file:
        labels = list(csv.DictReader(file, delimiter=";"))
    out: Dict[str, Any] = {}
    data: Dict[str, Dict[str, Any]] = {}
    for level in ("n1", "n2"):
        subset = [r for r in labels if r["nivel"] == level and r["referencia"].strip() and r["previsto"].strip()]
        data[level] = {"targets": [r["referencia"] for r in subset], "predicted": [r["previsto"] for r in subset],
                       "y": [int(r["correto"]) for r in subset], "scores": {m: [] for m in METRICS}}
    rv.compute_all_scores([data])
    for level, d in data.items():
        bm25 = BM25(d["targets"])
        d["scores"]["bm25"] = [bm25.similarity(t, p) for t, p in zip(d["targets"], d["predicted"])]
        y = np.array(d["y"])
        level_out = {"pares": len(y), "positivos": int(y.sum())}
        for m, values in d["scores"].items():
            v = np.array([np.nan if x is None else x for x in values], dtype=float)
            ok = ~np.isnan(v)
            level_out[m] = float(roc_auc_score(y[ok], v[ok])) if ok.sum() and len(set(y[ok])) > 1 else None
        out[level] = level_out
    return out


# ------------------------------------------------------------------ tabela

def _fmt(x: float | None) -> str:
    return "---" if x is None else f"{x:.2f}".replace(".", "{,}").replace("-", "$-$")


def write_table(result: Dict[str, Any]) -> None:
    order = ["fuzzywuzzy", "tfidf", "bm25", "sbert", "bertimbau", "multilingual", "wmd_ft", "wmd_nilc", "bertscore", "rouge_l"]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Redundância e poder discriminante das similaridades. $\rho$: correlação de Spearman, item a item, com o Multilingual "
        r"(N1 e N2, pares com os dois lados preenchidos). $\tau$: concordância de Kendall entre a ordenação dos seis modelos pela métrica e pelo F1. "
        r"AUC: capacidade de separar pares corretos de incorretos segundo os 264 pares rotulados à mão (0{,}5 equivale ao acaso). "
        r"O BM25 não integra a validação; entra como alternativa ao TF-IDF.}",
        r"\label{tab:redundancia_metricas}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r" & \multicolumn{3}{c}{N1} & \multicolumn{3}{c}{N2} \\",
        r"Métrica & $\rho$ & $\tau$ & AUC & $\rho$ & $\tau$ & AUC \\",
        r"\hline",
    ]
    for m in order:
        cells = [LABELS[m]]
        for level in ("n1", "n2"):
            corr = result["correlacao"][level]
            i, j = corr["metricas"].index(m), corr["metricas"].index("multilingual")
            cells += [_fmt(corr["spearman"][i][j]), _fmt(result["ordenacao"][level].get(m)), _fmt(result["auc_manual"][level].get(m))]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "redundancia_metricas.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = {
        "correlacao": {level: correlations(level) for level in ("n1", "n2", "n3")},
        "ordenacao": {level: ranking_agreement(level) for level in ("n1", "n2", "n3")},
        "auc_manual": manual_auc(),
    }
    write_json(OUTPUT_DIR / "redundancia_metricas.json", result)
    write_table(result)
    print(json.dumps({k: result[k] for k in ("ordenacao", "auc_manual")}, indent=1, ensure_ascii=False))
    for level in ("n1", "n2", "n3"):
        c = result["correlacao"][level]
        print(level, c["pares"])
        names = c["metricas"]
        for i, a in enumerate(names):
            print(f"{a:>12} " + " ".join(f"{c['spearman'][i][j]:5.2f}" for j in range(len(names))))


if __name__ == "__main__":
    main()
