"""Prepara results/analises/rotulos_para_revisao.csv para conferencia par a par.

Ordena os pares por quanto o julgamento pesou (prioridade 1 = casos limite) e
pre-preenche a coluna `seu_rotulo` com o rotulo proposto, para que a conferencia seja de
ler e corrigir o que discordar. Depois: python -m tools.aplicar_revisao --conferido NOME

Uso:
    python -m tools.preparar_revisao
"""

import csv, re, sys, unicodedata
sys.stdout.reconfigure(encoding='utf-8')

def norm(t):
    t = unicodedata.normalize('NFKD', (t or '').lower())
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return [w for w in re.findall(r'\w+', t) if w]

def f1(a, b):
    A, B = norm(a), norm(b)
    if not A or not B:
        return 0.0
    common = 0
    tmp = list(B)
    for w in A:
        if w in tmp:
            tmp.remove(w); common += 1
    if not common:
        return 0.0
    p, r = common / len(A), common / len(B)
    return 2 * p * r / (p + r)

SRC = 'results/analises/rotulos_manuais.csv'
DST = 'results/analises/rotulos_para_revisao.csv'
rows = list(csv.DictReader(open(SRC, encoding='utf-8-sig'), delimiter=';'))

out = []
for r in rows:
    s = f1(r['referencia'], r['previsto'])
    lab = int(r['correto'])
    # prioridade: casos em que o julgamento pesou, nao os obvios
    if lab == 1:
        pri, motivo = 1, 'marquei CORRETO'
    elif s >= 0.55:
        pri, motivo = 1, 'marquei incorreto, mas os textos sao muito parecidos'
    elif s >= 0.3:
        pri, motivo = 2, 'marquei incorreto; sobreposicao parcial'
    else:
        pri, motivo = 3, 'marquei incorreto; textos claramente diferentes'
    out.append({**r, 'prioridade': pri, 'sobreposicao': f'{s:.2f}',
                'por_que_conferir': motivo, 'seu_rotulo': lab})

out.sort(key=lambda r: (r['prioridade'], -float(r['sobreposicao'])))
with open(DST, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, delimiter=';', fieldnames=[
        'prioridade', 'seu_rotulo', 'correto', 'por_que_conferir', 'sobreposicao',
        'id', 'nivel', 'operador', 'referencia', 'previsto'])
    w.writeheader()
    for r in out:
        w.writerow({k: r[k] for k in w.fieldnames})

from collections import Counter
c = Counter(r['prioridade'] for r in out)
print(f'{DST}: {len(out)} pares')
for p in sorted(c):
    print(f'  prioridade {p}: {c[p]} pares')
