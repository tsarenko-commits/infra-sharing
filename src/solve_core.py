"""
solve_core.py — оркестратор пилотного ядра infra-sharing.

Пайплайн:
  1. generate_instance.build_instance() -> экземпляр (dict) + instance.json
  2. milp_reduced.build_and_solve(instance) -> номинальное решение + solution_nominal.json
  3. сводка на stdout: число модулей, суммарная ёмкость, макс. загрузка ребра

Запуск:  python solve_core.py
Зависимости: см. requirements.txt (highspy>=1.7); стандартная библиотека для генератора.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

INSTANCE_PATH = HERE / "instance.json"
SOLUTION_PATH = HERE / "solution_nominal.json"

CAPACITY_MODULE = 900  # Гбит — ёмкость одного арендуемого модуля


def main() -> int:
    # --- 1. Генерация экземпляра ---
    try:
        from generate_instance import build_instance
    except ImportError as e:
        print("[ERR] не найден generate_instance.py рядом с solve_core.py:", e)
        return 1

    instance = build_instance()
    INSTANCE_PATH.write_text(json.dumps(instance, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    n_nodes = len(instance["nodes"])
    n_edges = len(instance["edges"])
    n_demands = len(instance["demands"])
    total_load = sum(d["rate"] for d in instance["demands"])
    print(f"[OK] экземпляр: узлов={n_nodes}, рёбер={n_edges}, заявок={n_demands}, "
          f"суммарная нагрузка={total_load}")
    print(f"[OK] сохранён {INSTANCE_PATH.name}")

    # --- 2. Решение сведённой MILP ---
    try:
        from milp_reduced import build_and_solve
    except ImportError as e:
        print("[ERR] не найден milp_reduced.py рядом с solve_core.py:", e)
        return 1

    solution = build_and_solve(instance)
    SOLUTION_PATH.write_text(json.dumps(solution, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"[OK] сохранён {SOLUTION_PATH.name}")

    # --- 3. Сводка ---
    modules = solution.get("modules", {})
    total_capacity = CAPACITY_MODULE * sum(modules.values())
    edge_loads = solution.get("edge_loads", {})
    max_load = max(edge_loads.values()) if edge_loads else 0.0
    max_edge = max(edge_loads, key=edge_loads.get) if edge_loads else "-"
    print("---- сводка (номинальный сценарий) ----")
    print(f"арендовано модулей: {sum(modules.values())} (по узлам: {modules})")
    print(f"суммарная арендуемая ёмкость: {total_capacity} Гбит")
    print(f"максимальная загрузка ребра: {max_load} Гбит (ребро {max_edge})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
