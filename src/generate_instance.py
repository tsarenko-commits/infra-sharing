"""
Генератор экземпляра задачи infra-sharing.

Воспроизводит пилотный экземпляр pilot_14_13_5 (см. specs/pilot_14_13_5_spec.txt):
  - сеть из 100 узлов, 3 ЦОД;
  - 4 класса заявок: 50x0.05, 50x0.4, 20x10, 3x400 Гбит;
  - привязка БС к узлам по правилу floor по списку узлов вне ЦОД
    (гарантирует ровно 50 периферийных узлов без конфликта с ЦОД);
  - маршруты заявок - до ближайшего ЦОД по кратчайшей стороне.

Выход: instance.json со всеми данными экземпляра.
Запуск:  python generate_instance.py [выходной_файл]
"""

import json
import sys
from collections import deque

# --------------------- Зафиксированные константы ---------------------

SEED_NAME = "pilot_14_13_5"
N_NODES = 100
DC_NODES = [0, 24, 71]          # три центра обработки данных
CAPACITY_GBIT = 900.0           # ёмкость сегмента C
N_BS = 50                       # периферийных базовых станций

TRAFFIC_CLASSES = [
    # (имя класса, количество заявок, объём одной заявки, Гбит)
    ("access", 50, 0.05),
    ("aggr",   50, 0.40),
    ("trunk",  20, 10.0),
    ("dc",      3, 400.0),
]

TOTAL_WORK_LOAD = sum(cnt * vol for _, cnt, vol in TRAFFIC_CLASSES)  # 1422.5

# Контрольные числа из spec_core.txt, раздел 4:
#   total_work_load      - сверяется жёстко (assert);
#   nominal_max_segment  - сверяется на чистом прогоне (печатается);
#   target_at_C900       - сверяется решателем milp_reduced.py.
CONTROL = {
    "total_work_load": 1422.5,
    "nominal_max_segment": 443.6,
    "target_at_C900": 3664625550,
}

# ------------------------- Построение сети ---------------------------

def build_edges(n):
    """Детерминированная топология: кольцо + фиксированные хорды."""
    edges = set()
    for i in range(n):
        edges.add(tuple(sorted((i, (i + 1) % n))))
        edges.add(tuple(sorted((i, (i * 37 + 11) % n))))
    return sorted(edges)


def bfs_parents(n, edges, src):
    """Родители BFS от src: кратчайшие пути по числу рёбер."""
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    parent = [-1] * n
    parent[src] = src
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if parent[v] == -1:
                parent[v] = u
                q.append(v)
    return parent


def path_to(parent, src, dst):
    """Путь от dst к src по родителям, развёрнутый src -> dst."""
    path = [dst]
    while path[-1] != src:
        if parent[path[-1]] == -1:
            raise ValueError(f"узел {path[-1]} недостижим от {src}")
        path.append(parent[path[-1]])
    return path[::-1]

# --------------------- Привязка и заявки ------------------------------

def bind_bs(n, dc_nodes, n_bs=N_BS):
    """Правило floor по списку узлов вне ЦОД:
    позиции floor(k * len(свободных) / n_bs), k = 0..n_bs-1.
    Строго возрастают, поэтому узлы не повторяются и не задевают ЦОД."""
    free = [v for v in range(n) if v not in dc_nodes]
    assert len(free) >= n_bs
    bound = [free[(k * len(free)) // n_bs] for k in range(n_bs)]
    assert len(set(bound)) == n_bs
    assert not (set(bound) & set(dc_nodes))
    return bound


def nearest_dc(src, dc_nodes, parents):
    """Ближайший ЦОД по числу рёбер; при равенстве - меньший индекс."""
    best, best_d = None, None
    for d in dc_nodes:
        dist = len(path_to(parents[d], d, src)) - 1
        if best is None or dist < best_d:
            best, best_d = d, dist
    return best


def build_demands(n, dc_nodes, bound, parents):
    """Заявки четырёх классов с детерминированным размещением."""
    non_dc = [v for v in range(n) if v not in dc_nodes]
    demands = []
    did = 0

    # access: ровно по одной заявке с каждой привязанной БС
    for node in bound:
        dst = nearest_dc(node, dc_nodes, parents)
        demands.append({"id": did, "class": "access", "src": node,
                        "dst": dst, "gbits": 0.05})
        did += 1

    # aggr, trunk: модульное распределение по узлам вне ЦОД
    for name, cnt, vol in (("aggr", 50, 0.40), ("trunk", 20, 10.0)):
        step = max(len(non_dc) // cnt, 1)
        for k in range(cnt):
            node = non_dc[(k * step) % len(non_dc)]
            dst = nearest_dc(node, dc_nodes, parents)
            demands.append({"id": did, "class": name, "src": node,
                            "dst": dst, "gbits": vol})
            did += 1

    # dc: три заявки 400 Гбит по кольцу ЦОД 0 -> 24 -> 71 -> 0
    for a, b in ((0, 24), (24, 71), (71, 0)):
        demands.append({"id": did, "class": "dc", "src": a,
                        "dst": b, "gbits": 400.0})
        did += 1

    return demands

# ----------------------------- Сборка ---------------------------------

def build_instance():
    edges = build_edges(N_NODES)
    parents = {s: bfs_parents(N_NODES, edges, s) for s in range(N_NODES)}
    bound = bind_bs(N_NODES, DC_NODES)
    demands = build_demands(N_NODES, DC_NODES, bound, parents)

    total = round(sum(d["gbits"] for d in demands), 2)
    assert abs(total - TOTAL_WORK_LOAD) < 1e-6, (total, TOTAL_WORK_LOAD)
    assert abs(total - CONTROL["total_work_load"]) < 1e-6

    # номинальная загрузка сегментов: заявка считается на каждом ребре пути
    seg_load = {}
    for d in demands:
        p = path_to(parents[d["dst"]], d["dst"], d["src"])
        for a, b in zip(p, p[1:]):
            key = tuple(sorted((a, b)))
            seg_load[key] = seg_load.get(key, 0.0) + d["gbits"]

    nominal_max = max(seg_load.values()) if seg_load else 0.0

    return {
        "meta": {"seed_name": SEED_NAME, "n_nodes": N_NODES,
                 "n_edges": len(edges), "n_demands": len(demands)},
        "dc_nodes": DC_NODES,
        "capacity_gbit": CAPACITY_GBIT,
        "bound_bs_nodes": bound,
        "edges": [list(e) for e in edges],
        "demands": demands,
        "diagnostics": {
            "total_work_load_gbit": total,
            "nominal_max_segment_gbit": round(nominal_max, 1),
            "control": CONTROL,
        },
    }


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "instance.json"
    inst = build_instance()
    diag = inst["diagnostics"]
    print(f"Экземпляр: {inst['meta']['n_nodes']} узлов, "
          f"{inst['meta']['n_edges']} рёбер, "
          f"{inst['meta']['n_demands']} заявок")
    print(f"Привязано БС: {len(inst['bound_bs_nodes'])} "
          f"(без конфликтов с ЦОД {inst['dc_nodes']})")
    print(f"Суммарная нагрузка: {diag['total_work_load_gbit']} Гбит "
          f"(контроль {CONTROL['total_work_load']}) - OK")
    print(f"Номинальная макс. загрузка сегмента: "
          f"{diag['nominal_max_segment_gbit']} Гбит "
          f"(контроль {CONTROL['nominal_max_segment']} - сверить на прогоне)")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(inst, f, ensure_ascii=False, indent=2)
    print(f"Записан файл: {out}")


if __name__ == "__main__":
    main()
