"""Сведённая MILP-модель размещения арендуемых каналов (infra-sharing).

Вход:  instance.json (выход generate_instance.py).
Выход: решение HiGHS (через scipy) + сводка; метаданные -> solution_nominal.json.

Модель:
  Для каждой заявки d заранее построены k = K_CAND маршрутов-кандидатов
    до каждого из двух ЦОД: целевого (dst) и ближайшего альтернативного.
  Переменные:
    x_{d,p} in [0,1] — доля заявки d на маршруте p (сумма по p = 1);
    y_e in {0,1}     — арендуемый модуль ёмкости C на ребре e.
  Ограничения:
    (F1) sum_p x_{d,p} = 1                         для каждой заявки d;
    (F3) sum_{d,p: e in p} b_d * x_{d,p} <= C*y_e  для каждого ребра e;
    (F5) x_{d,p} <= y_e                            для каждой пары (d,p), e in p.
  Целевая (номинальный режим): min sum_e C*y_e.
"""

import heapq
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

CAPACITY_C_DEFAULT = 900.0   # Гбит, ёмкость арендуемого модуля (если нет в экземпляре)
K_CAND = 4                   # число маршрутов-кандидатов на заявку

STATUS_NAMES = {
    0: "Optimal",
    1: "Достигнут лимит времени/итераций",
    2: "Infeasible (ограничения несовместны)",
    3: "Unbounded",
    4: "Прочее (решение не найдено)",
}


def _as_id(x):
    """Возвращает идентификатор узла: и число, и словарь с ключом 'id'."""
    return x if isinstance(x, int) else x["id"]


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
    seen = set()
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
        if len(path) >= 2:
            banned.add((min(path[0], path[1]), max(path[0], path[1])))
        key = frozenset(path)
        if key in seen:
            continue
        seen.add(key)
        if paths and len(path) > 2 * len(paths[0]):
            continue
        paths.append(path)
    return paths



def _bfs_dist(adj: dict, src: int) -> dict:
    """Расстояния от src до всех узлов (BFS по числу рёбер)."""
    dist = {src: 0}
    queue = [src]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist

def build_and_solve(inst: dict, out_dir: Path) -> dict:
    edges = [tuple(e) for e in inst["edges"]]
    edge_idx = {tuple(sorted(e)): i for i, e in enumerate(edges)}
    demands = inst["demands"]
    dcs = sorted({_as_id(x) for x in inst["dc_nodes"]})

    # множество узлов: ЦОД + привязанные БС + всё, что встречается в рёбрах/заявках
    nodes = set(dcs) | {_as_id(x) for x in inst["bound_bs_nodes"]}
    for u, v in edges:
        nodes.update((u, v))
    for dm in demands:
        nodes.add(dm["src"])
        nodes.add(dm["dst"])
    n = max(nodes) + 1

    adj = {u: [] for u in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    cap = float(inst.get("capacity_gbit", CAPACITY_C_DEFAULT))

    # маршруты-кандидаты: (индекс заявки, ширина, набор рёбер)
    cand = []
    for d_id, dm in enumerate(demands):
        primary = dm["dst"]  # ближайший ЦОД уже назначен генератором
        # альтернативный ЦОД: ближайший по фактической длине пути от src
        dist = _bfs_dist(adj, dm["src"])
        others = [dc for dc in dcs if dc != primary]
        alt = min(others, key=lambda dc: dist.get(dc, float("inf"))) if others else primary
        targets = [primary, alt] if others else [primary]
        for tgt in targets:
            for path in k_shortest_paths(adj, dm["src"], tgt):
                es = [edge_idx[tuple(sorted((path[i], path[i + 1])))]
                      for i in range(len(path) - 1)]
                if es:
                    cand.append((d_id, float(dm["gbits"]), es))

    nx, ny = len(cand), len(edges)
    n_demands = len(demands)

    # (F1): сумма по кандидатам заявки = 1
    by_demand = {}
    for j, (d_id, _, _) in enumerate(cand):
        by_demand.setdefault(d_id, []).append(j)

    # (F3): load_e - C*y_e <= 0
    by_edge = {}
    for j, (_, _, es) in enumerate(cand):
        for e in set(es):
            by_edge.setdefault(e, []).append(j)

    rows, cols, vals = [], [], []
    for d_id, js in by_demand.items():
        for j in js:
            rows.append(d_id)
            cols.append(j)
            vals.append(1.0)
    for e, js in by_edge.items():
        for j in js:
            rows.append(n_demands + e)
            cols.append(j)
            vals.append(cand[j][1])
        rows.append(n_demands + e)
        cols.append(nx + e)
        vals.append(-cap)

    # 5: разрезы y_e >= x_{d,p} — усиление нижней границы.
    # Строка на КАЖДУЮ пару (кандидат j, ребро e): x_{d,p} - y_e <= 0,
    # т.е. поток по ребру возможен только при арендованном ребре.
    n_cuts = sum(len(set(es)) for _, _, es in cand)
    cut_r = n_demands + ny
    for j, (_, _, es) in enumerate(cand):
        for e in set(es):
            rows.append(cut_r)
            cols.append(j)
            vals.append(1.0)
            rows.append(cut_r)
            cols.append(nx + e)
            vals.append(-1.0)
            cut_r += 1

    A = sp.csr_matrix((vals, (rows, cols)),
                      shape=(n_demands + ny + n_cuts, nx + ny))
    lb = np.concatenate([np.ones(n_demands),
                         np.full(ny, -np.inf),
                         np.full(n_cuts, -np.inf)])
    ub = np.concatenate([np.ones(n_demands),
                         np.zeros(ny),
                         np.zeros(n_cuts)])
    constraints = LinearConstraint(A, lb, ub)

    integrality = np.concatenate([np.zeros(nx), np.ones(ny)])  # x — непрерывные, y — целые
    cost = np.concatenate([np.zeros(nx), np.full(ny, cap)])

    # 4б: рёбра, не входящие ни в один маршрут-кандидат, исключаем из аренды
    used_edges = set()
    for _, _, es in cand:
        used_edges.update(es)

    var_ub = np.ones(nx + ny)
    for e in range(ny):
        if e not in used_edges:
            var_ub[nx + e] = 0.0

    res = milp(c=cost, constraints=constraints, integrality=integrality,
               bounds=Bounds(np.zeros(nx + ny), var_ub),
               options={"time_limit": 3600})

    status = STATUS_NAMES.get(res.status, f"код {res.status}")
    dual = getattr(res, "mip_dual_bound", None)
    gap = getattr(res, "mip_gap", None)
    if dual is not None:
        gap_txt = f"{gap * 100:.2f}%" if gap is not None else "н/д"
        print(f"Нижняя граница: {dual:.1f} Гбит; зазор: {gap_txt}")
    if res.x is None:
        print(f"Статус: {status}")
        print("[ERR] решения нет — пришлите этот вывод целиком")
        return {"status": status}

    col = list(res.x)
    y = col[nx:]
    used = [e for e in range(ny) if y[e] > 0.5]
    loads = {}
    for j, (d_id, b, es) in enumerate(cand):
        for e in set(es):
            loads[e] = loads.get(e, 0.0) + b * col[j]


    total_cap = cap * len(used)
    max_load = max(loads.values()) if loads else 0.0
    print(f"Статус: {status}")
    print(f"Арендованных модулей: {len(used)} из {ny} рёбер")
    print(f"Суммарная арендуемая ёмкость: {total_cap:.1f} Гбит")
    print(f"Максимальная загрузка ребра: {max_load:.1f} Гбит "
          f"({max_load / cap * 100:.1f}% от модуля)")

    out = {
        "status": status,
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
