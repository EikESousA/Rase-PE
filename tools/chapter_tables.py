"""Gera as tabelas do Capitulo 5 a partir de metrics/ e predicts/.

Escreve em dissertacao/Conteudo/Tabelas/ no mesmo formato usado pelo capitulo
(realce por coluna com \\melhor e \\pior, negrito no limiar 0,7), de modo que os
numeros do texto e das tabelas venham sempre da mesma rodada.

Uso:
    python -m tools.chapter_tables
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "metrics"
PREDICTS_DIR = PROJECT_ROOT / "predicts"
OUT_DIR = PROJECT_ROOT / "dissertacao" / "Conteudo" / "Tabelas"

MODELS = ["alpaca", "dolphin", "gemma", "llama", "mistral", "qwen"]
LABEL = {m: m.capitalize() for m in MODELS}
SIM_METRICS = ["fuzzywuzzy", "tfidf", "sbert", "bertimbau", "multilingual",
               "wmd_ft", "wmd_nilc", "bertscore", "rouge_l"]
CLF_KEYS = ["accuracy", "precision", "recall", "f1"]
THRESHOLD = 0.7
N3_FIELDS = ("object", "property", "comparation", "target", "unit")

SIM_HEADER = [
    "\\hline",
    " & \\multicolumn{2}{c}{Lexical} & \\multicolumn{3}{c}{Semântica} & "
    "\\multicolumn{2}{c}{WMD} & \\multicolumn{2}{c}{Geração} \\\\",
    "\\cline{2-3}\\cline{4-6}\\cline{7-8}\\cline{9-10}",
    "Modelo & FW & TF-IDF & SBERT & BERTimbau & Multi. & WMD\\_ft & WMD\\_nilc & "
    "BERTScore & ROUGE-L \\\\",
    "\\hline",
]
CLF_HEADER = [
    "Modelo  & \\multicolumn{2}{c}{Acurácia} & \\multicolumn{2}{c}{Precisão} & "
    "\\multicolumn{2}{c}{Revocação} & \\multicolumn{3}{c}{F1} \\\\",
    "\\hline",
]


# ---------------------------------------------------------------- formatacao

def num(value: Optional[float], decimals: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{decimals}f}".replace(".", "{,}")


def seconds(value: float) -> str:
    inteiro, _, decimal = f"{value:.2f}".partition(".")
    if len(inteiro) > 3:
        inteiro = f"{inteiro[:-3]}\\,{inteiro[-3:]}"
    return f"{inteiro}{{,}}{decimal}"


def hms(value: float) -> str:
    total = int(round(value))
    return f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def decorate(cells: List[Optional[float]], bold: bool = True,
             lower_is_better: bool = False) -> List[str]:
    """Aplica \\melhor no melhor valor da coluna, \\pior no pior e negrito no limiar."""
    valid = [round(c, 3) for c in cells if c is not None]
    best = (min(valid) if lower_is_better else max(valid)) if valid else None
    worst = (max(valid) if lower_is_better else min(valid)) if valid else None
    out = []
    for cell in cells:
        if cell is None:
            out.append("-")
            continue
        text = num(cell)
        if bold and cell >= THRESHOLD:
            text = f"\\textbf{{{text}}}"
        value = round(cell, 3)
        if best is not None and value == best and best != worst:
            text = f"\\melhor{{{text}}}"
        elif worst is not None and value == worst and best != worst:
            text = f"\\pior{{{text}}}"
        out.append(text)
    return out


def table(caption: str, label: str, size: str, colspec: str, body: Sequence[str]) -> str:
    return "\n".join([
        "\\begin{table}[ht]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
        f"\\begin{{tabular}}{{{colspec}}}",
        *body,
        "\\hline", "\\end{tabular}", "\\end{table}", ""])


def write(name: str, content: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(content, encoding="utf-8", newline="\r\n")
    print(f"Gerada {OUT_DIR / name}")


# ------------------------------------------------------------------- leitura

def load(name: str) -> Dict[str, Any]:
    return json.loads((METRICS_DIR / name).read_text(encoding="utf-8"))


def averages(data: Dict[str, Any], section: Optional[str] = None) -> Dict[str, Any]:
    if section:
        return data[section]["averages"]
    return data["averages"]


EXPERIMENTS = [
    ("EN1", "validate_n1.json", None),
    ("EN2", "validate_n2.json", None),
    ("EN1N2", "validate_n1n2.json", "n1n2"),
    ("EN3", "validate_n3.json", None),
    ("EN1N2N3", "validate_n1n2n3.json", "n1n2n3"),
]


def all_averages() -> Dict[str, Dict[str, Any]]:
    return {name: averages(load(f), sec) for name, f, sec in EXPERIMENTS}


# -------------------------------------------------------- tabelas por experimento

def _sim_block(avg: Dict[str, Any]) -> List[str]:
    columns = [decorate([avg[m].get(k) for m in MODELS]) for k in SIM_METRICS]
    return [f"{LABEL[m]:<7} & " + " & ".join(col[i] for col in columns) + " \\\\"
            for i, m in enumerate(MODELS)]


def _clf_block(avg: Dict[str, Any]) -> List[str]:
    columns = [decorate([(avg[m].get("classification_macro") or {}).get(k) for m in MODELS])
               for k in CLF_KEYS]
    spans = ["\\multicolumn{2}{c}", "\\multicolumn{2}{c}", "\\multicolumn{2}{c}",
             "\\multicolumn{3}{c}"]
    rows = []
    for i, m in enumerate(MODELS):
        cells = [f"{span}{{{col[i]}}}" for span, col in zip(spans, columns)]
        rows.append(f"{LABEL[m]:<7} & " + " & ".join(cells) + " \\\\")
    return rows


def experiment_table(name: str, avg: Dict[str, Any], caption: str,
                     clf_title: Optional[str]) -> str:
    body = [*SIM_HEADER, *_sim_block(avg)]
    if clf_title is not None:
        body += ["\\hline",
                 f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{Métricas de classificação}}{clf_title}}} \\\\",
                 *CLF_HEADER, *_clf_block(avg)]
    return table(caption, f"tab:resultados_{name.lower()}", "\\scriptsize",
                 "lccccccccc", body)


# ------------------------------------------------------------ tabelas agregadas

METRIC_ROWS = [
    ("Similaridade lexical", [("FuzzyWuzzy", "fuzzywuzzy"), ("TF-IDF", "tfidf")]),
    ("Similaridade semântica", [("SBERT", "sbert"), ("BERTimbau", "bertimbau"),
                                ("Multilingual", "multilingual")]),
    ("Distância semântica (WMD)", [("WMD\\_ft", "wmd_ft"), ("WMD\\_nilc", "wmd_nilc")]),
    ("Métricas voltadas para geração", [("BERTScore", "bertscore"), ("ROUGE-L", "rouge_l")]),
]
CLF_ROWS = [("Acurácia", "accuracy"), ("Precisão", "precision"),
            ("Revocação", "recall"), ("F1-score", "f1")]
AGG_ORDER = ["EN1", "EN2", "EN1N2", "EN3", "EN1N2N3"]
NO_CLF = {"EN1N2", "EN1N2N3"}


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def metricas_resultados(avgs: Dict[str, Dict[str, Any]]) -> str:
    sim = {exp: {k: mean([avgs[exp][m].get(k) for m in MODELS]) for k in SIM_METRICS}
           for exp in AGG_ORDER}
    clf = {exp: {k: mean([(avgs[exp][m].get("classification_macro") or {}).get(k)
                          for m in MODELS]) for k in CLF_KEYS}
           for exp in AGG_ORDER if exp not in NO_CLF}

    # o realce e por coluna (experimento), sobre todas as linhas da tabela
    cols = {}
    for exp in AGG_ORDER:
        values = [sim[exp][k] for _, metrics in METRIC_ROWS for _, k in metrics]
        values += [clf[exp][k] for _, k in CLF_ROWS] if exp in clf else [None] * len(CLF_ROWS)
        cols[exp] = decorate(values)

    body = ["\\hline",
            "Métrica       & EN1   & EN2   & EN1N2 & EN3   & EN1N2N3 \\\\", "\\hline"]
    i = 0
    for group, metrics in METRIC_ROWS:
        body.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{group}}}}} \\\\")
        for label, _ in metrics:
            body.append(f"\\hspace{{1em}}{label} & " +
                        " & ".join(cols[exp][i] for exp in AGG_ORDER) + " \\\\")
            i += 1
        body.append("\\hline")
    body.append("\\multicolumn{6}{@{}l}{\\textit{Métricas de classificação}} \\\\")
    for label, _ in CLF_ROWS:
        cells = [cols[exp][i] if exp in clf else "N/A" for exp in AGG_ORDER]
        body.append(f"\\hspace{{1em}}{label} & " + " & ".join(cells) + " \\\\")
        i += 1
    return table("Médias por métrica nos cinco experimentos, agregadas sobre os seis modelos.",
                 "tab:metricas_resultados", "\\small", "lccccc", body)


def ranking_global(avgs: Dict[str, Dict[str, Any]]) -> str:
    per_model = {m: {exp: mean([avgs[exp][m].get(k) for k in SIM_METRICS])
                     for exp in AGG_ORDER} for m in MODELS}
    for m in MODELS:
        per_model[m]["global"] = mean([per_model[m][exp] for exp in AGG_ORDER])
    order = sorted(MODELS, key=lambda m: per_model[m]["global"], reverse=True)
    cols = {c: decorate([per_model[m][c] for m in order]) for c in AGG_ORDER + ["global"]}
    body = ["\\hline",
            "Modelo  & EN1   & EN2   & EN1N2 & EN3   & EN1N2N3 & Média Global \\\\",
            "\\hline"]
    for i, m in enumerate(order):
        cells = [cols[c][i] for c in AGG_ORDER + ["global"]]
        body.append(f"{LABEL[m]:<7} & " + " & ".join(cells) + " \\\\")
    return table("Média global por modelo (nove métricas de similaridade $\\times$ cinco experimentos).",
                 "tab:ranking_global", "\\small", "lcccccc", body), per_model, order


# ---------------------------------------------------------------- tempo e N3

TIME_TABLES = [
    ("tempo_n1", "n1", "Tempo total de geração por modelo em N1."),
    ("tempo_n2", "n2", "Tempo total de geração por modelo em N2."),
    ("tempo_n1n2", "n1n2", "Tempo total de geração por modelo em N1N2."),
    ("tempo_n3", "n3", "Tempo total de geração por modelo em N3 (EN3 isolado)."),
    ("tempo_n1n2n3", "n1n2n3",
     "Tempo total de geração por modelo em N1N2N3 (\\textit{pipeline} completo)."),
]


def read_times(level: str) -> Dict[str, float]:
    times = {}
    for m in MODELS:
        path = PREDICTS_DIR / f"generate_{level}_{m}.json"
        if path.exists():
            times[m] = float(json.loads(path.read_text(encoding="utf-8")).get("time", 0.0))
    return times


def tempo_table(name: str, level: str, caption: str) -> str:
    times = read_times(level)
    valid = [times[m] for m in MODELS if m in times]
    best, worst = min(valid), max(valid)
    body = ["\\hline", "Modelo & Tempo (s) & Tempo (h:mm:ss) \\\\", "\\hline"]
    for m in MODELS:
        if m not in times:
            continue
        s, h = seconds(times[m]), hms(times[m])
        if times[m] == best:
            s, h = f"\\melhor{{{s}}}", f"\\melhor{{{h}}}"
        elif times[m] == worst:
            s, h = f"\\pior{{{s}}}", f"\\pior{{{h}}}"
        body.append(f"{LABEL[m]:<7} & {s} & {h} \\\\")
    return table(caption, f"tab:{name}", "", "lrr", body), times


def erros_n3() -> str:
    """VP, falsos positivos com referencia vazia e com valor divergente, por modelo."""
    stats, ref_filled, total = {}, None, None
    for m in MODELS:
        path = METRICS_DIR / "details" / f"validate_n3_{m}.jsonl"
        if not path.exists():
            continue
        vp = fp_vazio = fp_divergente = 0
        preenchidos = posicoes = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                alvo = row.get("target_properties") or {}
                saida = row.get("predicted_properties") or {}
                for field in N3_FIELDS:
                    a = (alvo.get(field) or "").strip()
                    p = (saida.get(field) or "").strip()
                    posicoes += 1
                    if a:
                        preenchidos += 1
                    if a and p == a:
                        vp += 1
                    elif p and not a:
                        fp_vazio += 1
                    elif p and a and p != a:
                        fp_divergente += 1
        stats[m] = (vp, fp_vazio, fp_divergente)
        ref_filled, total = preenchidos, posicoes

    def br(n: int) -> str:
        return f"{n:,}".replace(",", ".")

    body = ["\\hline",
            "Modelo & VP & FP com referência vazia & FP com valor divergente \\\\",
            "\\hline"]
    for m in MODELS:
        if m in stats:
            vp, fv, fd = stats[m]
            body.append(f"{LABEL[m]:<7} & {br(vp)} & {br(fv)} & {br(fd)} \\\\")
    caption = (f"Decomposição dos acertos e falsos positivos por modelo em EN3 (soma dos cinco "
               f"campos avaliados; a referência tem {br(ref_filled)} posições preenchidas em "
               f"{br(total)}).")
    return table(caption, "tab:erros_n3", "\\small", "lrrr", body), stats, ref_filled, total


# -------------------------------------------------- composicao do dataset (cap. 4)

OPERATORS = [("requeriments", "Requisito (R)"), ("aplicability", "Aplicabilidade (A)"),
             ("selection", "Seleção (S)"), ("exception", "Exceção (E)")]


def dataset_stats() -> Dict[str, Any]:
    data = json.loads((PROJECT_ROOT / "dataset.json").read_text(encoding="utf-8"))
    itens = data["datas"]
    regras = [r for x in itens for r in x["texts_n1"]]
    presentes = {op: 0 for op, _ in OPERATORS}
    campos = {f: 0 for f in N3_FIELDS}
    campos_presentes = {f: 0 for f in N3_FIELDS}
    tipo = 0
    tipo_presentes = 0
    orfaos = 0
    for r in regras:
        for op, _ in OPERATORS:
            o = r["operators_n2"].get(op) or {}
            tem_texto = bool((o.get("text_n2") or "").strip())
            if tem_texto:
                presentes[op] += 1
            p = o.get("properties_n3") or {}
            if (p.get("type") or "").strip():
                tipo += 1
                if tem_texto:
                    tipo_presentes += 1
            cheio = False
            for f in N3_FIELDS:
                if (p.get(f) or "").strip():
                    campos[f] += 1
                    cheio = True
                    if tem_texto:
                        campos_presentes[f] += 1
            # operador sem trecho, mas com atributos anotados
            if cheio and not tem_texto:
                orfaos += 1
    return {
        "normas": len(itens), "regras": len(regras), "registros": 4 * len(regras),
        "posicoes": 5 * 4 * len(regras), "presentes": presentes,
        "total_presentes": sum(presentes.values()), "campos": campos,
        "total_campos": sum(campos.values()), "tipo": tipo,
        "campos_presentes": campos_presentes, "tipo_presentes": tipo_presentes,
        "orfaos": orfaos,
        "regras_por_norma": {k: sum(1 for x in itens if len(x["texts_n1"]) == k)
                             for k in sorted({len(x["texts_n1"]) for x in itens})},
        "origem": {o: sum(1 for x in itens if x.get("origem") == o)
                   for o in {x.get("origem") for x in itens}},
    }


def operadores_dataset(st: Dict[str, Any]) -> str:
    body = ["\\hline", "Operador & Ocorrências & Proporção das regras \\\\", "\\hline"]
    for op, label in OPERATORS:
        n = st["presentes"][op]
        body.append(f"{label} & {n} & {num(100 * n / st['regras'], 1)}\\% \\\\")
    return table(f"Distribuição dos operadores RASE nas {st['regras']} regras atômicas do \\textit{{dataset}}.",
                 "tab:operadores_dataset", "\\small", "lrr", body)


def campos_n3_dataset(st: Dict[str, Any]) -> str:
    body = ["\\hline",
            f"Campo & Preenchidos & Nos {st['total_presentes']} presentes & "
            f"\\% dos {st['total_presentes']} presentes & "
            f"\\% dos {st['registros']} registros \\\\", "\\hline"]
    linhas = [("\\textit{type}", st["tipo"], st["tipo_presentes"])] + \
             [(f"\\textit{{{f}}}", st["campos"][f], st["campos_presentes"][f])
              for f in N3_FIELDS]
    for label, n, npres in linhas:
        body.append(f"{label} & {n} & {npres} & "
                    f"{num(100 * npres / st['total_presentes'], 1)}\\% & "
                    f"{num(100 * n / st['registros'], 1)}\\% \\\\")
    return table("Preenchimento dos campos do nível N3 no \\textit{dataset}. "
                 f"A coluna ``Preenchidos'' conta os {st['registros']} registros; "
                 f"a coluna seguinte restringe a contagem aos {st['total_presentes']} operadores "
                 f"com trecho anotado no N2. A diferença vem de {st['orfaos']} operadores sem "
                 "trecho no N2 cujos atributos N3 foram anotados a partir do Apêndice~B.",
                 "tab:campos_n3_dataset", "\\small", "lrrrr", body)


# --------------------------------------------- analises complementares (cap. 5)

ANALISES_DIR = PROJECT_ROOT / "results" / "analises"
COT_VARIANTS = [("direto", "Direto"), ("cot_estruturado", "Estrut."),
                ("cot_explicito", "Explíc.")]


def _p(value: float) -> str:
    if value is None:
        return "---"
    if value < 0.001:
        return "$<$0{,}001"
    return num(value, 3)


def ablacao_cot() -> str:
    d = json.loads((ANALISES_DIR / "ablacao_cot.json").read_text(encoding="utf-8"))
    body = ["\\hline",
            " & \\multicolumn{3}{c}{F1 macro} & \\multicolumn{3}{c}{Trocas de operador} & "
            "\\multicolumn{3}{c}{Tempo (s)} & \\multicolumn{2}{c}{$p$ (McNemar)} \\\\",
            "\\cline{2-4}\\cline{5-7}\\cline{8-10}\\cline{11-12}",
            "Modelo & Direto & Estrut. & Explíc. & Direto & Estrut. & Explíc. & "
            "Direto & Estrut. & Explíc. & D$\\times$E & E$\\times$X \\\\", "\\hline"]
    for m in MODELS:
        v = d["modelos"].get(m)
        if not v:
            continue
        var = v["variantes"]
        f1 = decorate([var[k]["f1"]["macro"] for k, _ in COT_VARIANTS])
        trocas = [str(var[k]["trocas_de_operador"]) for k, _ in COT_VARIANTS]
        tempos = [f"{var[k]['tempo_s']:.0f}" for k, _ in COT_VARIANTS]
        c = v["comparacoes"]
        p1 = _p(c["direto_vs_cot_estruturado"]["mcnemar"]["p_valor"])
        p2 = _p(c["cot_estruturado_vs_cot_explicito"]["mcnemar"]["p_valor"])
        body.append(f"{LABEL[m]:<7} & " + " & ".join(f1 + trocas + tempos + [p1, p2]) + " \\\\")
    return table("Ablação do \\textit{Chain-of-Thought} no N2: F1 macro dos operadores, trocas de "
                 "operador, tempo de geração e significância das diferenças (N1 de referência "
                 "como entrada).",
                 "tab:ablacao_cot", "\\footnotesize\n\\setlength{\\tabcolsep}{3pt}",
                 "lrrrrrrrrrrr", body)


JUIZ_NIVEIS = [("en1:n1", "N1"), ("en1n2:n2", "N2"), ("en1n2n3:n3", "N3")]


def llm_juiz() -> str:
    d = json.loads((ANALISES_DIR / "llm_juiz.json").read_text(encoding="utf-8"))
    r = d["resumo"]
    body = ["\\hline",
            "Nível & Modelo & $n$ & Nota média & Aprovados (juiz) & Aprovados (0,7) & "
            "Concordância & $\\kappa$ \\\\", "\\hline"]
    for i, (chave, nivel) in enumerate(JUIZ_NIVEIS):
        if i:
            body.append("\\hline")
        for m in MODELS:
            v = r.get(f"{chave}:{m}")
            if not v:
                continue
            body.append(
                f"{nivel} & {LABEL[m]} & {v['n']} & {num(v['nota_media'])} & "
                f"{num(v['aprovados_juiz'])} & {num(v['aprovados_limiar'])} & "
                f"{num(v['concordancia'])} & {num(v['kappa_cohen'])} \\\\")
    juiz = d["juiz"].replace("_", "\\_")
    return table(f"LLM-juiz ({juiz}): nota média atribuída, proporção de pares aprovados pelo juiz "
                 f"e pelo limiar de 0,7, concordância entre os dois e kappa de Cohen. "
                 "N1 no EN1, N2 no EN1N2 e N3 no EN1N2N3.",
                 "tab:llm_juiz", "\\footnotesize", "llrrrrrr", body)


def roc_manual_caption() -> Dict[str, Any]:
    d = json.loads((ANALISES_DIR / "limiar_roc.json").read_text(encoding="utf-8"))
    return d.get("roc_manual", {})


# ----------------------------------------------------------------------- main

def main() -> None:
    avgs = all_averages()
    st = dataset_stats()
    write("ablacao_cot.tex", ablacao_cot())
    write("llm_juiz.tex", llm_juiz())
    write("operadores_dataset.tex", operadores_dataset(st))
    write("campos_n3_dataset.tex", campos_n3_dataset(st))

    write("resultados_en1.tex", experiment_table(
        "EN1", avgs["EN1"], "Resultados por modelo em EN1 (similaridade e classificação).", ""))
    write("resultados_en2.tex", experiment_table(
        "EN2", avgs["EN2"],
        "Resultados por modelo em EN2 (similaridade e classificação macro por operador).",
        " (macro por operador)"))
    write("resultados_en1n2.tex", experiment_table(
        "EN1N2", avgs["EN1N2"], "Resultados por modelo em EN1N2 (similaridade).", None))
    write("resultados_n3.tex", experiment_table(
        "N3", avgs["EN3"],
        "Resultados por modelo em EN3 (similaridade e classificação macro por campo).",
        " (macro por campo)"))
    write("resultados_en1n2n3.tex", experiment_table(
        "EN1N2N3", avgs["EN1N2N3"],
        "Resultados por modelo em EN1N2N3 (similaridade sobre o JSON estruturado).", None))

    write("metricas_resultados.tex", metricas_resultados(avgs))
    ranking, per_model, order = ranking_global(avgs)
    write("ranking_global.tex", ranking)

    tempos = {}
    for name, level, caption in TIME_TABLES:
        content, times = tempo_table(name, level, caption)
        write(f"{name}.tex", content)
        tempos[level] = times

    content, stats, ref_filled, total = erros_n3()
    write("erros_n3.tex", content)

    resumo = {
        "similaridade_por_experimento": {
            exp: {k: mean([avgs[exp][m].get(k) for m in MODELS]) for k in SIM_METRICS}
            for exp in AGG_ORDER},
        "classificacao_por_experimento": {
            exp: {k: mean([(avgs[exp][m].get("classification_macro") or {}).get(k)
                           for m in MODELS]) for k in CLF_KEYS}
            for exp in AGG_ORDER if exp not in NO_CLF},
        "media_por_modelo": per_model,
        "ordem_global": order,
        "tempos": tempos,
        "erros_n3": {m: {"vp": v[0], "fp_vazio": v[1], "fp_divergente": v[2]}
                     for m, v in stats.items()},
        "n3_referencia": {"preenchidos": ref_filled, "posicoes": total},
        "por_modelo_por_experimento": {
            exp: {m: dict(avgs[exp][m]) for m in MODELS} for exp in AGG_ORDER},
    }
    resumo["dataset"] = st
    path = PROJECT_ROOT / "results" / "analises" / "resumo_capitulo5.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Resumo numerico salvo em {path}")


if __name__ == "__main__":
    main()
