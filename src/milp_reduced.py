"""Сведённая MILP-модель размещения арендуемых каналов (infra-sharing).

Вход:  instance.json (выход generate_instance.py).
Выход: решение HiGHS + сводка по загрузкам; метаданные -> solution_nominal.json.

Модель (сведённая, по разделу 5 спецификации):
  Для каждой заявки d заранее построены k=2 маршрута-кандидата:
    до ближайшего ЦОД и до альтернативного.
  Переменные:
    x_{d,p} in [0,1] — доля заявки d на маршруте p (сумма по p = 1);
    y_e in {0,1}     — арендуемый модуль ёмкости C на ребре e.
  Ограничения:
    (F1) sum_p x_{d,p} = 1                         для каждой заявки d;
    (F3) sum_{d,p: e in p} b_d * x_{d,p} <= C*y_e  для каждого ребра e.
  Целевая (базовая семантика — номинальный режим):
    min sum_e C*y_e — минимизация суммарной арендуемой ёмкости (Гбит).
  Семантику сценариев отказа (1+1, переключения) сверяем после первого
  прогона по контрольному числу из spec_core.txt.
"""

import heapq
import json
from pathlib import Path

import highspy

CAPACITY_C = 900.0   # Гбит, ёмкость арендуемого модуля
K_CAND = 2           # число маршрутов-кандидатов на заявку


def load_instance(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def k_shortest_paths(adj: dict, src: int, dst: int, k: int = K_CAND):
    """k простых кратчайших путей (упрощённый Йен, достаточно для пилота)."""
    def dijkstra(start, banned):
        dist = {start: 0.0}
        prev = {}
        pq = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float("inf")):
                continue
            for v in adj.get(u, ()):
                if (min(u, v), max(u, v)) in banned:
                    continue
                nd = d + 1
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    paths = []
    banned = set()
    for _ in range(k):
        dist, prev = dijkstra(src, banned)
        if dst not in dist:
            break
        path, u = [dst], dst
        while u != src:
            p = prev.get(u)
            if p is None:
                break
            path.append(p)
            u = p
        if u != src:
            break
        path.reverse()
        paths.append(path)
        if len(path) >= 2:
            banned.add((min(path[0], path[1]), max(path[0], path[1])))
    return paths


def build_and_solve(inst: dict, out_dir: Path) -> dict:
    n = inst["n_nodes"]
    edges = [tuple(e) for e in inst["edges"]]
    edge_idx = {tuple(sorted(e)): i for i, e in enumerate(edges)}
    demands = inst["demands"]
    dcs = set(inst["datacenters"])

    adj = {u: [] for u in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # маршруты-кандидаты: (индекс заявки, ширина, набор рёбер)
    cand = []
    for d_id, dm in enumerate(demands):
        targets = [dm["dc_nearest"]] + [
            dc for dc in sorted(dcs) if dc != dm["dc_nearest"]
        ][:1]
        for tgt in targets:
            for path in k_shortest_paths(adj, dm["src"], tgt):
                es = [edge_idx[tuple(sorted((path[i], path[i + 1])))]
                      for i in range(len(path) - 1)]
                cand.append((d_id, float(dm["b_gbit"]), es))

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    inf = highspy.kHighsInf

    nx, ny = len(cand), len(edges)
    h.addVars(nx + ny, 0.0, 1.0)

    # (F1): сумма по кандидатам заявки = 1
    by_demand = {}
    for j, (d_id, _, _) in enumerate(cand):
        by_demand.setdefault(d_id, []).append(j)
    for d_id, js in by_demand.items():
        h.addRow(1.0, 1.0, len(js),
                 [highspy.HighsSparseMatrix.index_t(j) for j in js],
                 [1.0] * len(js))

    # (F3): load_e - C*y_e <= 0
    by_edge = {}
    for j, (_, _, es) in enumerate(cand):
        for e in set(es):
            by_edge.setdefault(e, []).append(j)
    for e in range(ny):
        js = by_edge.get(e, [])
        cols = js + [nx + e]
        coefs = [cand[j][1] for j in js] + [-CAPACITY_C]
        h.addRow(-inf, 0.0, len(cols),
                 [highspy.HighsSparseMatrix.index_t(c) for c in cols], coefs)

    # цель: min sum_e C*y_e
    h.changeColsCost(
        ny,
        [highspy.HighsSparseMatrix.index_t(nx + e) for e in range(ny)],
        [CAPACITY_C] * ny,
    )
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)
    h.run()

    status = h.getModelStatus()
    col = list(h.getSolution().col_value)
    y = col[nx:]
    used = [e for e in range(ny) if y[e] > 0.5]
    loads = {}
    for j, (d_id, b, es) in enumerate(cand):
        if col[j] > 1e-6:
            for e in set(es):
                loads[e] = loads.get(e, 0.0) + b

    total_cap = CAPACITY_C * len(used)
    max_load = max(loads.values()) if loads else 0.0
    print(f"Статус: {h.modelStatusToString(status)}")
    print(f"Арендованных модулей: {len(used)} из {ny} рёбер")
    print(f"Суммарная арендуемая ёмкость: {total_cap:.1f} Гбит")
    print(f"Максимальная загрузка ребра: {max_load:.1f} Гбит "
          f"({max_load / CAPACITY_C * 100:.1f}% от модуля)")

    out = {
        "status": h.modelStatusToString(status),
        "modules_used": len(used),
        "edges_used": [edges[e] for e in used],
        "total_capacity_gbit": total_cap,
        "max_edge_load_gbit": max_load,
        "edge_loads": {str(edges[e]): v for e, v in loads.items()},
    }
    with open(out_dir / "solution_nominal.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    return out


def main():
    here = Path(__file__).resolve().parent
    inst = load_instance(here / "instance.json")
    build_and_solve(inst, here)


if __name__ == "__main__":
    main()
