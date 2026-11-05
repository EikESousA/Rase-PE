"""Gera as tabelas LaTeX das analises complementares a partir dos JSONs em results/analises.

Uso:
    python -m analyses.report_tables
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from analyses.common import (
    DATASET_PATH,
    FIELDS,
    OPERATORS,
    OPERATOR_LABELS,
    OUTPUT_DIR,
    PROJECT_ROOT,
    contaminated_items,
    fmt,
    load_json,
)

TABLES_DIR = PROJECT_ROOT / "dissertacao" / "Conteudo" / "Tabelas"
MODELS = ["alpaca", "dolphin", "gemma", "llama", "mistral", "qwen"]


def _label(m: str) -> str:
    return m.capitalize()


def _pct(a: float, b: float) -> str:
    return fmt(100 * a / b, 1) + "\\%" if b else "---"


def _dataset_sizes() -> Dict[str, int]:
    """Tamanhos do dataset usados nas legendas, com e sem as normas contaminadas."""
    data = load_json(DATASET_PATH)
    itens = data.get("datas", [])
    excluidos = set(contaminated_items(data))
    limpos = [x for i, x in enumerate(itens) if i not in excluidos]
    return {
        "normas": len(itens),
        "regras": sum(len(x.get("texts_n1", [])) for x in itens),
        "contaminadas": len(excluidos),
        "normas_limpas": len(limpos),
        "regras_limpas": sum(len(x.get("texts_n1", [])) for x in limpos),
    }


def _write(name: str, body: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / name).write_text(body, encoding="utf-8")
    print("Gerada", TABLES_DIR / name)


def table_segmentation(tax: Dict[str, Any]) -> None:
    t = _dataset_sizes()
    rows = []
    for m in MODELS:
        n = tax["experimentos"]["en1"][m]["n1"]
        c, s, k = n["contagens"], n["segmentacao_por_item"], n["regras_inventadas_por_tipo"]
        rows.append(
            f"{_label(m)} & {c['regras_previstas']} & {s.get('igual', 0)} & {s.get('supersegmentacao', 0)} & {s.get('subsegmentacao', 0)} & "
            f"{c.get('inventada', 0)} & {k.get('fragmento_de_regra_de_referencia', 0)} & {k.get('trecho_da_norma_sem_regra_correspondente', 0)} & "
            f"{k.get('conteudo_ausente_da_norma', 0)} & {c.get('omitida', 0)} \\\\"
        )
    _write("segmentacao_n1.tex", "\n".join([
        "\\begin{table}[ht]", "\\centering",
        "\\caption{Segmentação em N1 (EN1): regras previstas, itens por tipo de segmentação e destino das regras sem par. "
        f"A referência tem {t['regras']} regras em {t['normas']} normas.}}",
        "\\label{tab:segmentacao_n1}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lrrrrrrrrr}", "\\hline",
        " & & \\multicolumn{3}{c}{Normas} & \\multicolumn{4}{c}{Regras previstas sem par} & \\\\",
        "\\cline{3-5}\\cline{6-9}",
        "Modelo & Regras & Iguais & Super & Sub & Total & Fragmento & Sem regra & Fora da norma & Omitidas \\\\",
        "\\hline", *rows, "\\hline", "\\end{tabular}", "\\end{table}", ""]))


def table_omissions(tax: Dict[str, Any]) -> None:
    rows = []
    for m in MODELS:
        n1 = tax["experimentos"]["en1"][m]["n1"]
        n2 = tax["experimentos"]["en1n2"][m]["n2"]
        n3 = tax["experimentos"]["en1n2n3"][m]["n3"]
        invented = sum(v["categorias"].get("alucinacao:regra_inventada", 0) for v in n2["por_operador"].values())
        missing = sum(v["categorias"].get("omissao:regra_ausente", 0) for v in n2["por_operador"].values())
        rows.append(
            f"{_label(m)} & {n1['omissoes']} & {n1['alucinacoes']} & {n1['conteudo_errado']} & "
            f"{n2['omissoes']} ({missing}) & {n2['alucinacoes']} ({invented}) & {n2['conteudo_errado']} & "
            f"{n3['omissoes']} & {n3['alucinacoes']} & {n3['conteudo_errado']} \\\\"
        )
    _write("omissoes_alucinacoes.tex", "\n".join([
        "\\begin{table}[ht]", "\\centering",
        "\\caption{Omissões (Om.), alucinações (Al.) e conteúdo errado (Err.) por nível. N1 no EN1, N2 no EN1N2 e N3 no EN1N2N3. Entre parênteses, no N2, a parte que decorre de regras ausentes ou inventadas no N1.}",
        "\\label{tab:omissoes_alucinacoes}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lrrrrrrrrr}", "\\hline",
        " & \\multicolumn{3}{c}{N1 (regras)} & \\multicolumn{3}{c}{N2 (operadores)} & \\multicolumn{3}{c}{N3 (campos)} \\\\",
        "\\cline{2-4}\\cline{5-7}\\cline{8-10}",
        "Modelo & Om. & Al. & Err. & Om. & Al. & Err. & Om. & Al. & Err. \\\\",
        "\\hline", *rows, "\\hline", "\\end{tabular}", "\\end{table}", ""]))


def table_taxonomy_n2(tax: Dict[str, Any]) -> None:
    labels = [
        ("acerto", "Acerto"),
        ("acerto_por_ausencia", "Acerto por ausência"),
        ("omissao:operador_nao_extraido", "Omissão: operador não extraído"),
        ("omissao:regra_ausente", "Omissão: regra ausente no N1"),
        ("alucinacao:operador_inexistente_na_regra", "Alucinação: operador inexistente"),
        ("alucinacao:troca_de_operador", "Alucinação: troca de operador"),
        ("alucinacao:regra_inventada", "Alucinação: regra inventada no N1"),
        ("alucinacao:conteudo_ausente_da_norma", "Alucinação: conteúdo fora da norma"),
        ("divergente:troca_de_operador", "Divergente: troca de operador"),
        ("divergente:trecho_longo_demais", "Divergente: trecho longo demais"),
        ("divergente:trecho_curto_demais", "Divergente: trecho curto demais"),
        ("divergente:trecho_errado_da_norma", "Divergente: trecho errado da norma"),
        ("divergente:conteudo_ausente_da_norma", "Divergente: conteúdo fora da norma"),
    ]
    totals = {op: Counter() for op in OPERATORS}
    swaps: Counter = Counter()
    for m in MODELS:
        n2 = tax["experimentos"]["en1n2"][m]["n2"]
        for op in OPERATORS:
            totals[op].update(n2["por_operador"][op]["categorias"])
        swaps.update(n2["trocas"])
    rows = []
    for key, label in labels:
        values = [totals[op].get(key, 0) for op in OPERATORS]
        rows.append(f"{label} & " + " & ".join(str(v) for v in values) + f" & {sum(values)} \\\\")
    header = " & ".join(OPERATOR_LABELS[op] for op in OPERATORS)
    _write("taxonomia_n2.tex", "\n".join([
        "\\begin{table}[ht]", "\\centering",
        "\\caption{Taxonomia dos resultados do N2 por operador, somada sobre os seis modelos (EN1N2, regras emparelhadas pelo texto N1).}",
        "\\label{tab:taxonomia_n2}", "\\footnotesize",
        "\\begin{tabular}{lrrrrr}", "\\hline",
        f"Categoria & {header} & Total \\\\", "\\hline", *rows, "\\hline", "\\end{tabular}", "\\end{table}", ""]))
    names = {op: OPERATOR_LABELS[op] for op in OPERATORS}
    top = ", ".join(f"{names[a]} $\\rightarrow$ {names[b]} ({n})" for (ab, n) in swaps.most_common(4) for a, b in [ab.split("->")])
    (TABLES_DIR / "taxonomia_n2_trocas.txt").write_text(top, encoding="utf-8")


def table_taxonomy_n3(tax: Dict[str, Any]) -> None:
    labels = [
        ("acerto", "Acerto"),
        ("acerto_por_ausencia", "Acerto por ausência"),
        ("omissao", "Omissão"),
        ("preenchimento_indevido:operador_vazio_na_referencia", "Alucinação: operador vazio na referência"),
        ("preenchimento_indevido:campo_vazio_na_referencia", "Alucinação: campo vazio na referência"),
        ("valor_divergente:parcial", "Divergente: parcial"),
        ("valor_divergente:diferente", "Divergente: diferente"),
    ]
    totals = {f: Counter() for f in FIELDS}
    for m in MODELS:
        n3 = tax["experimentos"]["en1n2n3"][m]["n3"]
        for f in FIELDS:
            totals[f].update(n3["por_campo"][f]["categorias"])
    rows = []
    for key, label in labels:
        values = [totals[f].get(key, 0) for f in FIELDS]
        rows.append(f"{label} & " + " & ".join(str(v) for v in values) + f" & {sum(values)} \\\\")
    _write("taxonomia_n3.tex", "\n".join([
        "\\begin{table}[ht]", "\\centering",
        "\\caption{Taxonomia dos resultados do N3 por campo, somada sobre os seis modelos (EN1N2N3).}",
        "\\label{tab:taxonomia_n3}", "\\footnotesize",
        "\\begin{tabular}{lrrrrrr}", "\\hline",
        "Categoria & \\textit{object} & \\textit{property} & \\textit{comparation} & \\textit{target} & \\textit{unit} & Total \\\\",
        "\\hline", *rows, "\\hline", "\\end{tabular}", "\\end{table}", ""]))


def table_roc(roc: Dict[str, Any]) -> None:
    rows = []
    names = {"en1:n1": "N1 (EN1)", "en1n2:n2": "N2 (EN1N2)"}
    for key, name in names.items():
        for gold in ("estrito", "relaxado"):
            a = roc["roc"][key][gold]["agregado"]
            at7 = a["por_limiar"]["0.7"]
            rows.append(
                f"{name} & {gold} & {a['n']} & {a['positivos']} & {fmt(a['auc'])} & {fmt(a['limiar_youden'], 2)} & "
                f"{fmt(a['sensibilidade_youden'])} & {fmt(a['especificidade_youden'])} & {fmt(at7['sensibilidade'])} & {fmt(at7['especificidade'])} \\\\"
            )
    for level, name in (("n1", "N1 (todos os exp.)"), ("n2", "N2 (todos os exp.)")):
        a = roc.get("roc_manual", {}).get(level)
        if not a:
            continue
        at7 = a["por_limiar"]["0.7"]
        rows.append(
            f"{name} & manual & {a['n']} & {a['positivos']} & {fmt(a['auc'])} & {fmt(a['limiar_youden'], 2)} & "
            f"{fmt(a['sensibilidade_youden'])} & {fmt(a['especificidade_youden'])} & {fmt(at7['sensibilidade'])} & {fmt(at7['especificidade'])} \\\\"
        )
    _write("roc_limiar.tex", "\n".join([
        "\\begin{table}[ht]", "\\centering",
        "\\caption{Similaridade Multilingual como classificador de pares corretos: AUC, limiar ótimo pelo índice de Youden e desempenho no limiar adotado (0,7).}",
        "\\label{tab:roc_limiar}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{llrrrrrrrr}", "\\hline",
        " & & & & & \\multicolumn{3}{c}{Limiar de Youden} & \\multicolumn{2}{c}{Limiar 0,7} \\\\",
        "\\cline{6-8}\\cline{9-10}",
        "Nível & Gabarito & Pares & Corretos & AUC & Limiar & Sens. & Espec. & Sens. & Espec. \\\\",
        "\\hline", *rows, "\\hline", "\\end{tabular}", "\\end{table}", ""]))


def table_sweep(roc: Dict[str, Any]) -> None:
    thresholds = ["0.3", "0.5", "0.7", "0.9"]
    rows = []
    for key, metric, name in (("en1:n1", "f1", "N1"), ("en1n2:n2", "f1_macro_operadores", "N2")):
        for m in MODELS:
            vals = roc["varredura"][key][m]
            cells = []
            for t in thresholds:
                v = vals[t]
                cells.append(fmt(v[metric]))
            prec = fmt(vals["0.7"]["precisao"])
            sens = fmt(vals["0.7"]["sensibilidade"])
            spec = fmt(vals["0.7"]["especificidade"])
            rows.append(f"{name} & {_label(m)} & " + " & ".join(cells) + f" & {prec} & {sens} & {spec} \\\\")
        rows.append("\\hline")
    _write("varredura_limiar.tex", "\n".join([
        "\\begin{table}[ht]", "\\centering",
        "\\caption{F1 em diferentes limiares (F1 em N1; F1 macro dos operadores em N2) e precisão, sensibilidade e especificidade no limiar de 0,7. N1 no EN1 e N2 no EN1N2.}",
        "\\label{tab:varredura_limiar}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llrrrrrrr}", "\\hline",
        " & & \\multicolumn{4}{c}{F1 no limiar} & \\multicolumn{3}{c}{Limiar 0,7} \\\\",
        "\\cline{3-6}\\cline{7-9}",
        "Nível & Modelo & 0,3 & 0,5 & 0,7 & 0,9 & Prec. & Sens. & Espec. \\\\",
        "\\hline", *rows, "\\end{tabular}", "\\end{table}", ""]))


def table_libraries(lib: Dict[str, Any]) -> None:
    rows = []
    for key, level, name in (("en1", "n1", "N1"), ("en1n2", "n2", "N2"), ("en1n2n3", "n3", "N3")):
        for m in MODELS:
            v = lib["experimentos"][key][m][level]
            lo, hi = v["chrf++_ic95"]
            schema = "---"
            if level == "n3":
                s = lib["experimentos"][key][m]["n3_esquema"]
                schema = _pct(s["validos"], s["registros"])
            rows.append(
                f"{name} & {_label(m)} & {fmt(v['chrf++_sentenca_media'])} [{fmt(lo)}; {fmt(hi)}] & {fmt(v['bleu'])} & {fmt(v['ter'])} & "
                f"{fmt(v['levenshtein_norm'])} & {fmt(v['token_set_ratio'])} & {schema} \\\\"
            )
        rows.append("\\hline")
    _write("metricas_bibliotecas.tex", "\n".join([
        "\\begin{table}[ht]", "\\centering",
        "\\caption{Métricas de bibliotecas de referência: chrF++ médio com IC de 95\\%, BLEU e TER de corpus (sacreBLEU), similaridade de Levenshtein e \\textit{token set ratio} (RapidFuzz) e registros N3 válidos pelo esquema (Pydantic). N1 no EN1, N2 no EN1N2 e N3 no EN1N2N3.}",
        "\\label{tab:metricas_bibliotecas}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{llrrrrrr}", "\\hline",
        "Nível & Modelo & chrF++ [IC 95\\%] & BLEU & TER $\\downarrow$ & Levenshtein & Token set & Esquema válido \\\\",
        "\\hline", *rows, "\\end{tabular}", "\\end{table}", ""]))


def table_contamination(full: Dict[str, Any], clean: Dict[str, Any], roc_full: Dict[str, Any], roc_clean: Dict[str, Any]) -> None:
    t = _dataset_sizes()

    def n3_vp(tax: Dict[str, Any], m: str) -> int:
        return sum(v["categorias"].get("acerto", 0) for v in tax["experimentos"]["en1n2n3"][m]["n3"]["por_campo"].values())

    def n3_pos(tax: Dict[str, Any], m: str) -> int:
        total = 0
        for v in tax["experimentos"]["en1n2n3"][m]["n3"]["por_campo"].values():
            c = v["categorias"]
            total += c.get("acerto", 0) + c.get("omissao", 0) + sum(n for k, n in c.items() if k.startswith("valor_divergente"))
        return total

    rows = []
    for m in MODELS:
        f1_full = roc_full["varredura"]["en1:n1"][m]["0.7"]["f1"]
        f1_clean = roc_clean["varredura"]["en1:n1"][m]["0.7"]["f1"]
        f2_full = roc_full["varredura"]["en1n2:n2"][m]["0.7"]["f1_macro_operadores"]
        f2_clean = roc_clean["varredura"]["en1n2:n2"][m]["0.7"]["f1_macro_operadores"]
        rows.append(
            f"{_label(m)} & {fmt(f1_full)} & {fmt(f1_clean)} & {fmt(f2_full)} & {fmt(f2_clean)} & "
            f"{n3_vp(full, m)}/{n3_pos(full, m)} & {n3_vp(clean, m)}/{n3_pos(clean, m)} \\\\"
        )
    _write("contaminacao.tex", "\n".join([
        "\\begin{table}[ht]", "\\centering",
        "\\caption{Efeito da exclusão das " + str(t['contaminadas']) + " normas usadas como exemplo nos \\textit{prompts}: F1 em N1 (EN1), F1 macro em N2 (EN1N2) e campos N3 acertados sobre campos preenchidos na referência (EN1N2N3).}",
        "\\label{tab:contaminacao}", "\\footnotesize",
        "\\begin{tabular}{lrrrrrr}", "\\hline",
        " & \\multicolumn{2}{c}{F1 N1} & \\multicolumn{2}{c}{F1 macro N2} & \\multicolumn{2}{c}{Acertos N3} \\\\",
        "\\cline{2-3}\\cline{4-5}\\cline{6-7}",
        f"Modelo & {t['normas']} normas & {t['normas_limpas']} normas & {t['normas']} normas & {t['normas_limpas']} normas & {t['normas']} normas & {t['normas_limpas']} normas " "\\\\",
        "\\hline", *rows, "\\hline", "\\end{tabular}", "\\end{table}", ""]))


def main() -> None:
    tax = load_json(OUTPUT_DIR / "taxonomia_erros.json")
    roc = load_json(OUTPUT_DIR / "limiar_roc.json")
    lib = load_json(OUTPUT_DIR / "metricas_bibliotecas.json")
    table_segmentation(tax)
    table_omissions(tax)
    table_taxonomy_n2(tax)
    table_taxonomy_n3(tax)
    table_roc(roc)
    table_sweep(roc)
    table_libraries(lib)
    clean_tax = OUTPUT_DIR / "taxonomia_erros_sem_contaminacao.json"
    clean_roc = OUTPUT_DIR / "limiar_roc_sem_contaminacao.json"
    if clean_tax.exists() and clean_roc.exists():
        table_contamination(tax, load_json(clean_tax), roc, load_json(clean_roc))


if __name__ == "__main__":
    main()
