"""Gera diagramas conceituais (pipeline e experimentos) em PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "dissertacao" / "Imagens"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, facecolor="#E8F0FE", edgecolor="#1A73E8", fontsize=10, weight="normal"):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight=weight)


def arrow(ax, x1, y1, x2, y2, label=None, color="#5F6368", style="->", curve=0.0, label_dy=0.18, label_fs=8):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        color=color,
        linewidth=1.6,
        mutation_scale=18,
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(arr)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mx,
            my + label_dy,
            label,
            ha="center",
            va="bottom",
            fontsize=label_fs,
            color=color,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"),
        )


def plot_pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    # Engine bar (top) — caixa container Ollama
    container = FancyBboxPatch(
        (2.7, 4.2),
        10.6,
        2.4,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        facecolor="#F8F9FA",
        edgecolor="#9AA0A6",
        linewidth=1.2,
        linestyle="--",
    )
    ax.add_patch(container)
    ax.text(8.0, 6.35, "Ollama — 6 LLMs (Llama, Dolphin, Gemma, Mistral, Alpaca, Qwen)",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#5F6368")

    # Linha principal
    y_main = 5.2
    box(ax, 1.5, y_main, 2.0, 1.3, "dataset.json\n(76 normas)\ntexto bruto", facecolor="#FFF3E0", edgecolor="#E8710A", fontsize=9)
    box(ax, 5.0, y_main, 2.3, 1.5, "N1 — Segmentação\nAtômica\n(Prompt Direct)", facecolor="#E8F0FE", edgecolor="#1A73E8", fontsize=9)
    box(ax, 8.0, y_main, 2.3, 1.5, "N2 — Operadores\nRASE (R/A/S/E)\n(Chain-of-Thought)", facecolor="#E8F0FE", edgecolor="#1A73E8", fontsize=9)
    box(ax, 11.0, y_main, 2.3, 1.5, "N3 — JSON\nestruturado\n(Few-Shot)", facecolor="#E8F0FE", edgecolor="#1A73E8", fontsize=9)
    box(ax, 14.3, y_main, 1.8, 1.3, "JSON RASE\nfinal", facecolor="#E6F4EA", edgecolor="#188038", fontsize=9, weight="bold")

    arrow(ax, 2.55, y_main, 3.8, y_main, label="text", label_dy=0.25)
    arrow(ax, 6.2, y_main, 6.8, y_main, label="texts_n1", label_dy=0.25)
    arrow(ax, 9.2, y_main, 9.8, y_main, label="operators_n2", label_dy=0.25)
    arrow(ax, 12.2, y_main, 13.4, y_main, label="properties_n3", label_dy=0.25)

    # Validação (parte inferior)
    box(
        ax,
        8.0,
        1.4,
        13.5,
        1.6,
        "Validação par a par (saída do modelo $\\times$ referência)\n"
        "9 métricas de similaridade: FuzzyWuzzy, TF-IDF, SBERT, BERTimbau, Multilingual, WMD_ft, WMD_nilc, BERTScore, ROUGE-L\n"
        "4 métricas de classificação: Acurácia, Precisão, Revocação, F1   $\\rightarrow$   metrics/validate_*.json",
        facecolor="#FCE8E6",
        edgecolor="#D93025",
        fontsize=9,
    )

    arrow(ax, 5.0, 4.45, 5.0, 2.25, color="#D93025")
    arrow(ax, 8.0, 4.45, 8.0, 2.25, color="#D93025")
    arrow(ax, 11.0, 4.45, 11.0, 2.25, color="#D93025")

    ax.set_title("Pipeline de conversão de normas técnicas para o formato RASE", fontsize=12, fontweight="bold", pad=10)
    out = OUT_DIR / "4.1-pipeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_experiments() -> Path:
    fig, ax = plt.subplots(figsize=(15.5, 14.5))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 16)
    ax.set_aspect("equal")
    ax.axis("off")

    styles = {
        "raw": dict(facecolor="#FFF3E0", edgecolor="#E8710A", weight="normal"),
        "ref": dict(facecolor="#E6F4EA", edgecolor="#188038", weight="normal"),
        "model": dict(facecolor="#E8F0FE", edgecolor="#1A73E8", weight="bold"),
        "pred": dict(facecolor="#FFFFFF", edgecolor="#5F6368", weight="normal"),
        "compare": dict(facecolor="#FCE8E6", edgecolor="#D93025", weight="normal"),
    }

    def draw_row(y, title, steps, x_left=0.7, x_right=17.3, box_h=1.0):
        n = len(steps)
        cell = (x_right - x_left) / n
        bw = cell * 0.72
        fontsize = 10 if n <= 4 else (9 if n <= 6 else 8)
        ax.text(x_left - 0.1, y + box_h / 2 + 0.45, title, fontsize=11,
                fontweight="bold", color="#1A73E8", ha="left", va="bottom")
        centers = [x_left + cell * (i + 0.5) for i in range(n)]
        for cx, (label, kind) in zip(centers, steps):
            st = styles[kind]
            box(ax, cx, y, bw, box_h, label, facecolor=st["facecolor"],
                edgecolor=st["edgecolor"], fontsize=fontsize, weight=st["weight"])
        for i in range(n - 1):
            color = "#D93025" if i == n - 2 else "#5F6368"
            arrow(ax, centers[i] + bw / 2, y, centers[i + 1] - bw / 2, y, color=color)

    rows = [
        (14.4, "EN1 — Segmentação N1 isolada", [
            ("text\n(bruto)", "raw"),
            ("Modelo: N1", "model"),
            ("texts_n1_pred", "pred"),
            ("compara com\ntexts_n1_ref", "compare"),
        ]),
        (11.2, "EN2 — Identificação RASE isolada (N1 de referência como entrada)", [
            ("texts_n1\n(referência)", "ref"),
            ("Modelo: N2", "model"),
            ("operators_n2_pred", "pred"),
            ("compara com\noperators_n2_ref", "compare"),
        ]),
        (8.0, "EN3 — Estruturação N3 isolada (N2 de referência como entrada)", [
            ("operators_n2\n(referência)", "ref"),
            ("Modelo: N3", "model"),
            ("properties_n3_pred", "pred"),
            ("compara com\nproperties_n3_ref", "compare"),
        ]),
        (4.8, "EN1N2 — Pipeline encadeado (N1$\\rightarrow$N2 do mesmo modelo)", [
            ("text\n(bruto)", "raw"),
            ("Modelo: N1", "model"),
            ("texts_n1_pred", "pred"),
            ("Modelo: N2", "model"),
            ("operators_n2_pred", "pred"),
            ("compara\ncom ref.", "compare"),
        ]),
        (1.6, "EN1N2N3 — Pipeline completo encadeado (N1$\\rightarrow$N2$\\rightarrow$N3 do mesmo modelo)", [
            ("text\n(bruto)", "raw"),
            ("Modelo: N1", "model"),
            ("texts_n1_\npred", "pred"),
            ("Modelo: N2", "model"),
            ("operators_n2_\npred", "pred"),
            ("Modelo: N3", "model"),
            ("properties_n3_\npred", "pred"),
            ("compara\ncom ref.", "compare"),
        ]),
    ]
    for y, title, steps in rows:
        draw_row(y, title, steps)

    ax.set_title("Os cinco experimentos: EN1, EN2 e EN3 (isolados); EN1N2 e EN1N2N3 (encadeados)",
                 fontsize=13, fontweight="bold", pad=12)
    out = OUT_DIR / "4.2-experimentos.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pipeline_conversao() -> list[Path]:
    fig, ax = plt.subplots(figsize=(16, 7.2))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    xs = [2.0, 6.0, 10.0, 14.0, 18.0]
    y_top, top_h, bw = 6.3, 1.9, 3.0
    y_ex, ex_h = 2.55, 3.1

    band = FancyBboxPatch(
        (xs[1] - bw / 2 - 0.3, 0.75), xs[3] - xs[1] + bw + 0.6, 7.65,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor="#F1F5FE", edgecolor="#9AA0A6", linewidth=1.2, linestyle="--",
    )
    ax.add_patch(band)
    ax.text(xs[2], 8.0, "Ollama  ·  6 LLMs open-source  ·  extração de informação em 3 níveis",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#5F6368")

    stages = [
        ("Texto bruto", "norma NBR original", "#FFF3E0", "#E8710A", "-"),
        ("N1", "Prompt Direct\nextração de regras", "#E8F0FE", "#1A73E8", "-"),
        ("N2", "Chain-of-Thought\nextração de operadores", "#E8F0FE", "#1A73E8", "-"),
        ("N3", "Few-Shot\nextração de atributos", "#E8F0FE", "#1A73E8", "-"),
        ("Verificação", "modelo BIM\nSantos e Sampaio (2021)", "#F1F3F4", "#5F6368", "--"),
    ]
    for x, (title, sub, fc, ec, ls) in zip(xs, stages):
        patch = FancyBboxPatch(
            (x - bw / 2, y_top - top_h / 2), bw, top_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=2.0, linestyle=ls,
        )
        ax.add_patch(patch)
        ax.text(x, y_top + 0.42, title, ha="center", va="center", fontsize=15, fontweight="bold", color=ec)
        ax.text(x, y_top - 0.38, sub, ha="center", va="center", fontsize=10, color="#3C4043", linespacing=1.3)

    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + bw / 2 + 0.05, y_top, xs[i + 1] - bw / 2 - 0.05, y_top)
    for i, label in enumerate(["sentenças atômicas", "operadores R/A/S/E", "JSON estruturado"]):
        ax.text((xs[i + 1] + xs[i + 2]) / 2, y_top - top_h / 2 - 0.35, label,
                ha="center", va="center", fontsize=9, style="italic", color="#188038", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#E6F4EA", edgecolor="#188038", linewidth=0.8))

    examples = [
        "As áreas de qualquer\nespaço ou edificação\nde uso público ou\ncoletivo devem ser\nservidas de uma ou\nmais rotas acessíveis.",
        "1 regra atômica:\n\n“As áreas de qualquer\nespaço ou edificação\n(...) devem ser servidas\nde (...) rotas acessíveis.”",
        "A: As áreas de qualquer\n   espaço ou edificação\nS: uso público ou\n   coletivo\nE: \"\"\nR: devem ser servidas\n   de uma ou mais\n   rotas acessíveis",
        "S → {\n \"type\": \"selecao\",\n \"object\": \"espaço\",\n \"property\": \"uso\",\n \"comparation\": \"=\",\n \"target\": \"público\n    ou coletivo\",\n \"unit\": \"\" }",
        "Cada elemento do\nmodelo é conferido\ncontra o JSON:\n\nverde = conforme\nvermelho = não conforme",
    ]
    mono = [False, False, True, True, False]
    for x, text, is_mono in zip(xs, examples, mono):
        card = FancyBboxPatch(
            (x - bw / 2, y_ex - ex_h / 2), bw, ex_h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor="white", edgecolor="#DADCE0", linewidth=1.0,
        )
        ax.add_patch(card)
        ax.text(x, y_ex, text, ha="center" if not is_mono else "left", va="center",
                fontsize=8.0 if is_mono else 9.2, color="#3C4043",
                family="monospace" if is_mono else None, linespacing=1.35,
                position=(x - bw / 2 + 0.2 if is_mono else x, y_ex))
    ax.text(0.5, y_ex + ex_h / 2 + 0.3, "Exemplo (NBR 9050):", ha="left", va="center",
            fontsize=9.5, fontweight="bold", color="#5F6368")

    outs = []
    for ext in ("png", "svg"):
        out = OUT_DIR / f"4.3-pipeline-conversao.{ext}"
        fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
        outs.append(out)
    plt.close(fig)
    return outs


def main() -> None:
    p1 = plot_pipeline()
    p2 = plot_experiments()
    p3 = plot_pipeline_conversao()
    print("Gerados:")
    print(" -", p1)
    print(" -", p2)
    for p in p3:
        print(" -", p)


if __name__ == "__main__":
    main()
