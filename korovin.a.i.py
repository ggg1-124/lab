"""SimpleAnalysis — количественный анализ случайного набора данных.

Установка дополнительных библиотек:
    python -m pip install pandas matplotlib

Или из файла зависимостей:
    python -m pip install -r requirements.txt

Запуск:
    python korovin.a.i.py
"""

from __future__ import annotations

import random
from numbers import Integral
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


DATA_SIZE = 1000
LOWER_BOUND = -10_000
UPPER_BOUND = 10_000
CONTROL_SEED = 42


def generate_data(
    count: int = DATA_SIZE,
    lower: int = LOWER_BOUND,
    upper: int = UPPER_BOUND,
    seed: int | None = None,
) -> list[int]:
    """Сгенерировать список целых чисел в заданном диапазоне."""
    if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
        raise ValueError("Количество элементов должно быть положительным целым числом.")
    if lower > upper:
        raise ValueError("Нижняя граница диапазона не может быть больше верхней.")

    rng = random.Random(seed)
    return [rng.randint(lower, upper) for _ in range(int(count))]


def clean_data(
    data: Iterable[object],
    lower: int = LOWER_BOUND,
    upper: int = UPPER_BOUND,
) -> tuple[list[int], dict[str, int]]:
    """Очистить данные от цифрового мусора.

    Допускаются только целые числа (bool исключается) внутри [lower; upper].
    Строки, дробные значения, None и выходящие за диапазон значения отклоняются.
    Дубликаты не удаляются, так как для данной задачи они являются частью выборки.
    """
    cleaned: list[int] = []
    rejected = {"not_integer": 0, "out_of_range": 0}

    for value in data:
        if isinstance(value, bool) or not isinstance(value, Integral):
            rejected["not_integer"] += 1
            continue

        number = int(value)
        if not lower <= number <= upper:
            rejected["out_of_range"] += 1
            continue

        cleaned.append(number)

    if not cleaned:
        raise ValueError("После очистки не осталось корректных данных.")

    return cleaned, rejected


def calculate_statistics(series: pd.Series) -> dict[str, int | float]:
    """Рассчитать основные числовые характеристики набора данных."""
    if series.empty:
        raise ValueError("Нельзя рассчитывать характеристики пустого набора данных.")

    return {
        "Количество наблюдений": int(series.count()),
        "Минимальное значение": int(series.min()),
        "Максимальное значение": int(series.max()),
        "Сумма": int(series.sum()),
        "Среднее арифметическое": float(series.mean()),
        "Медиана": float(series.median()),
        "Дисперсия": float(series.var()),
        "Среднеквадратическое отклонение": float(series.std()),
        "Первый квартиль (Q1)": float(series.quantile(0.25)),
        "Третий квартиль (Q3)": float(series.quantile(0.75)),
        "Размах": int(series.max() - series.min()),
        "Количество повторов": int(series.duplicated().sum()),
    }


def build_dataframe(series: pd.Series) -> pd.DataFrame:
    """Сформировать DataFrame из исходного и двух отсортированных рядов."""
    up = series.sort_values().reset_index(drop=True)
    down = series.sort_values(ascending=False).reset_index(drop=True)

    return pd.DataFrame(
        {
            "Исходные данные": series.reset_index(drop=True),
            "По возрастанию": up,
            "По убыванию": down,
        }
    )


def create_plots(
    series: pd.Series,
    dataframe: pd.DataFrame,
    output_dir: str | Path = "results",
    show: bool = True,
) -> list[Path]:
    """Построить и сохранить три графика анализа."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    plt.figure(figsize=(10, 5))
    plt.plot(series)
    plt.title("Исходные данные")
    plt.xlabel("Номер элемента")
    plt.ylabel("Значение")
    plt.grid()
    path = output_path / "01_source_series.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    saved.append(path)

    rounded = series.round(-2)
    plt.figure(figsize=(10, 5))
    plt.hist(rounded, bins=25)
    plt.title("Гистограмма округленных данных")
    plt.xlabel("Значение")
    plt.ylabel("Частота")
    plt.grid()
    path = output_path / "02_histogram.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    saved.append(path)

    plt.figure(figsize=(10, 5))
    plt.plot(dataframe["По возрастанию"], label="По возрастанию")
    plt.plot(dataframe["По убыванию"], label="По убыванию")
    plt.title("Сравнение сортировок")
    plt.xlabel("Индекс")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid()
    path = output_path / "03_sorted_series.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    saved.append(path)

    if show:
        plt.show()
    else:
        plt.close("all")

    return saved


def run_analysis(seed: int = CONTROL_SEED, show_plots: bool = True):
    """Выполнить полный конвейер: генерация → очистка → анализ → визуализация."""
    raw_data = generate_data(seed=seed)
    cleaned_data, rejected = clean_data(raw_data)

    series = pd.Series(cleaned_data, dtype="int64")
    statistics = calculate_statistics(series)
    dataframe = build_dataframe(series)
    plot_files = create_plots(series, dataframe, show=show_plots)

    return series, statistics, dataframe, rejected, plot_files


def main() -> None:
    series, statistics, dataframe, rejected, plot_files = run_analysis(
        seed=CONTROL_SEED,
        show_plots=True,
    )

    print(f"Контрольный seed = {CONTROL_SEED}")
    print(f"Отклонено нецелых значений: {rejected['not_integer']}")
    print(f"Отклонено значений вне диапазона: {rejected['out_of_range']}")
    print("\nЧисловые характеристики:")
    for name, value in statistics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.2f}")
        else:
            print(f"{name}: {value}")

    print("\nПервые 10 строк DataFrame:")
    print(dataframe.head(10))
    print("\nГрафики сохранены:")
    for path in plot_files:
        print(path)


if __name__ == "__main__":
    main()
