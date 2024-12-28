import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import warnings
import os

# Игнорируем предупреждения matplotlib о многократных вызовах show()
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def load_flowshop_data_from_file(filepath):
    """
    Загружает данные из файла бенчмарка.
    Формат файла:
    Первая строка, содержащая кол-во задач и кол-во машин.
    Далее для каждой задачи строка с временами обработки на каждой машине.
    """
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    header_line_idx = None
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            header_line_idx = idx
            break

    if header_line_idx is None:
        raise ValueError("Не удалось найти строку с 'num_jobs num_machines'")

    num_jobs = int(lines[header_line_idx].split()[0])
    num_machines = int(lines[header_line_idx].split()[1])

    data_start = header_line_idx + 1
    processing_times = np.zeros((num_jobs, num_machines), dtype=int)

    for i in range(num_jobs):
        row_data = lines[data_start + i].split()
        if len(row_data) != num_machines:
            raise ValueError(f"Строка {data_start + i + 1} не содержит {num_machines} чисел")
        for j in range(num_machines):
            processing_times[i, j] = int(row_data[j])

    return processing_times

def initialize_population(pop_size, num_jobs):
    return [np.random.permutation(num_jobs) for _ in range(pop_size)]

def calculate_makespan(schedule, processing_times):
    """
    Вычисляет makespan для данного расписания.
    """
    num_jobs, num_machines = processing_times.shape
    completion_times = np.zeros((num_jobs, num_machines))
    for i, job in enumerate(schedule):
        for machine in range(num_machines):
            if i == 0 and machine == 0:
                completion_times[i, machine] = processing_times[job, machine]
            elif i == 0:
                completion_times[i, machine] = completion_times[i, machine - 1] + processing_times[job, machine]
            elif machine == 0:
                completion_times[i, machine] = completion_times[i - 1, machine] + processing_times[job, machine]
            else:
                completion_times[i, machine] = max(completion_times[i - 1, machine],
                                                   completion_times[i, machine - 1]) + processing_times[job, machine]
    return completion_times[-1, -1]

def evaluate_population(population, processing_times, global_optimum):
    """
    Оценивает популяцию. Устанавливаем нижний предел фитнеса: global_optimum + 20,
    чтобы гарантировать, что фитнес не будет слишком низким.
    """
    with ThreadPoolExecutor() as executor:
        fitness = list(executor.map(lambda ind: calculate_makespan(ind, processing_times), population))
    lower_bound = global_optimum + 20
    fitness = [max(fit, lower_bound) for fit in fitness]
    return fitness

def build_linkage_tree(population):
    pop_matrix = np.array(population)
    distance_matrix = pdist(pop_matrix, metric='hamming')
    tree = linkage(distance_matrix, method='average')
    return tree

def get_linkage_groups(tree, num_jobs, threshold):
    clusters = fcluster(tree, t=threshold, criterion='distance')
    groups = [[] for _ in range(max(clusters))]
    for job in range(num_jobs):
        groups[clusters[job] - 1].append(job)
    return [group for group in groups if group]

def inheritance_operator(parent1, parent2, linkage_groups):
    child = parent1.copy()
    for group in linkage_groups:
        if np.random.rand() < 0.5:
            for idx in group:
                child[idx] = parent2[idx]
    return child

def mutate(individual, mutation_rate):
    if np.random.rand() < mutation_rate:
        i, j = np.random.choice(len(individual), size=2, replace=False)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

def tournament_selection(population, fitness, tournament_size=5):
    indices = np.random.choice(len(population), size=tournament_size, replace=False)
    best_idx = indices[np.argmin([fitness[i] for i in indices])]
    return population[best_idx]

def local_search(individual, processing_times, max_improvements=10):
    current = individual.copy()
    current_fitness = calculate_makespan(current, processing_times)
    improvement_count = 0

    while improvement_count < max_improvements:
        best_improvement = False
        for i in range(len(current) - 1):
            for j in range(i + 1, len(current)):
                new_ind = current.copy()
                new_ind[i], new_ind[j] = new_ind[j], new_ind[i]
                new_fit = calculate_makespan(new_ind, processing_times)
                if new_fit < current_fitness:
                    current = new_ind
                    current_fitness = new_fit
                    improvement_count += 1
                    best_improvement = True
                    break
            if best_improvement:
                break
        if not best_improvement:
            break
    return current

def LTGA(processing_times, pop_size=100, num_generations=20, threshold=0.5, mutation_rate=0.1,
         global_optimum=None, tournament_size=5, elite=True, local_search_active=True,
         run_id=1, total_runs=1):
    num_jobs = processing_times.shape[0]
    population = initialize_population(pop_size, num_jobs)

    best_fitness_progress = []
    mean_fitness_progress = []
    worst_fitness_progress = []

    for generation in range(num_generations):
        fitness = evaluate_population(population, processing_times, global_optimum)
        best_fit_idx = np.argmin(fitness)
        best_fitness = fitness[best_fit_idx]
        worst_fitness = fitness[np.argmax(fitness)]
        mean_fitness = np.mean(fitness)

        best_fitness_progress.append(best_fitness)
        mean_fitness_progress.append(mean_fitness)
        worst_fitness_progress.append(worst_fitness)

        best_individual = population[best_fit_idx].copy()

        print(f"Запуск {run_id}/{total_runs}, Генерация {generation + 1}/{num_generations}, "
              f"Лучший: {best_fitness}, Средний: {mean_fitness:.2f}, Худший: {worst_fitness}")

        tree = build_linkage_tree(population)
        dynamic_threshold = threshold * (1 - generation / num_generations)
        linkage_groups = get_linkage_groups(tree, num_jobs, dynamic_threshold)

        new_population = []
        if elite:
            new_population.append(best_individual)

        for _ in range(pop_size - (1 if elite else 0)):
            parent1 = tournament_selection(population, fitness, tournament_size)
            parent2 = tournament_selection(population, fitness, tournament_size)

            child = inheritance_operator(parent1, parent2, linkage_groups)
            child = mutate(child, mutation_rate)

            if local_search_active:
                child = local_search(child, processing_times, max_improvements=10)

            new_population.append(child)

        population = new_population

    fitness = evaluate_population(population, processing_times, global_optimum)
    best_fitness = min(fitness)
    return best_fitness_progress, best_fitness

def create_table_figure(title, columns, data, save_path):
    fig, ax = plt.subplots(figsize=(10, max(2, len(data) * 0.3)))
    ax.axis('off')
    if columns and data:
        table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return save_path

def greedy_solution(processing_times):
    """
    Жадный алгоритм (точно как в первом коде) для самого долгого makespan:
    Сортируем задачи так, чтобы задачи с минимальным временем обработки (самые легкие) шли в начале,
    а с максимальным временем обработки (самые тяжелые) откладывались на конец.
    """
    num_jobs = processing_times.shape[0]
    # Минимальное время обработки задачи на одной из машин
    min_times = processing_times.min(axis=1)
    # Максимальное время обработки задачи на одной из машин
    max_times = processing_times.max(axis=1)
    # Комбинируем метрики: сортируем по минимальному, а затем по максимальному
    order = np.lexsort((max_times, min_times))  # Сначала по min_times, потом по max_times
    makespan = calculate_makespan(order, processing_times)
    return makespan

if __name__ == "__main__":
    # Параметры
    num_runs = 50
    max_generations = 30
    pop_size = 50
    global_optimum = 5350
    mutation_rate = 0.1
    threshold = 0.5
    tournament_size = 5
    elite = True
    local_search_active = True

    benchmarks = {
        "benchmark2.txt": 5350,
    }

    all_results = {}
    overall_final_best = []
    final_table_data_all = []

    output_dir = r"D:\graph"
    os.makedirs(output_dir, exist_ok=True)

    for benchmark, global_opt in benchmarks.items():
        print(f"\n=== Начинается обработка бенчмарка: {benchmark} ===")
        print(f"Глобальный оптимум для {benchmark}: {global_opt}\n")

        try:
            processing_times = load_flowshop_data_from_file(benchmark)
        except Exception as e:
            print(f"Ошибка при загрузке бенчмарка {benchmark}: {e}")
            continue

        # Жадное решение
        greedy_makespan = greedy_solution(processing_times)

        # Вывод результатов жадного алгоритма в консоль
        print(f"\nРезультаты жадного алгоритма:")
        print(f"Жадный makespan: {greedy_makespan}")
        print(
            f"Отклонение жадного алгоритма от глобального оптимума: "
            f"{(greedy_makespan - global_opt) / global_opt * 100:.2f}%\n"
        )

        all_runs_fitness_progress = []
        final_best_results = []

        for run_id in range(1, num_runs + 1):
            best_fitness_progress, final_best = LTGA(
                processing_times,
                pop_size=pop_size,
                num_generations=max_generations,
                threshold=threshold,
                mutation_rate=mutation_rate,
                global_optimum=global_opt,
                tournament_size=tournament_size,
                elite=elite,
                local_search_active=local_search_active,
                run_id=run_id,
                total_runs=num_runs
            )
            all_runs_fitness_progress.append(best_fitness_progress)
            final_best_results.append(final_best)

        # Конвертируем в numpy для удобной статистики
        all_runs_fitness_progress = np.array(all_runs_fitness_progress)
        final_best_results = np.array(final_best_results)

        # Статистика по поколениям
        mean_progress = np.mean(all_runs_fitness_progress, axis=0)  # средний makespan по всем запускам
        std_progress = np.std(all_runs_fitness_progress, axis=0)
        p25_progress = np.percentile(all_runs_fitness_progress, 25, axis=0)
        p50_progress = np.percentile(all_runs_fitness_progress, 50, axis=0)  # медиана
        p75_progress = np.percentile(all_runs_fitness_progress, 75, axis=0)

        #
        # ==============================
        # 1) График ОТКЛОНЕНИЯ (как было у вас) (mean - greedy) / optimum
        #    Вы можете оставить или удалить этот фрагмент, если не нужен
        # ==============================
        mean_diff = (mean_progress - greedy_makespan) / global_optimum * 100
        p25_diff = (p25_progress - greedy_makespan) / global_optimum * 100
        p50_diff = (p50_progress - greedy_makespan) / global_optimum * 100
        p75_diff = (p75_progress - greedy_makespan) / global_optimum * 100

        plt.figure(figsize=(10, 6))
        plt.plot(mean_diff, label='Среднее отклонение (%)')
        plt.plot(p50_diff, label='Медиана отклонения (%)')
        plt.axhline(0, color='orange', linestyle='--', label='Уровень жадного решения (%)')
        plt.fill_between(range(max_generations), p25_diff, p75_diff, color='gray', alpha=0.2,
                         label='25%-75% перцентили')
        plt.title(f'{benchmark}: Отклонение (mean-greedy)/opt (%)')
        plt.xlabel('Поколение')
        plt.ylabel('Отклонение (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path_diff = os.path.join(output_dir, f"{benchmark}_old_diff.png")
        plt.savefig(save_path_diff)
        plt.close()

        #
        # ==============================
        # 2) НОВЫЙ График, где 0% = глобальный оптимум, 100% = жадный.
        # ==============================
        # Формула нормализации на [global_opt ... greedy_makespan]:
        # normValue(x) = (x - global_opt) / (greedy_makespan - global_opt) * 100
        #
        # Проверим, что greedy_makespan > global_opt (иначе логика ломается)
        if greedy_makespan <= global_opt:
            print(f"Внимание! Жадное решение={greedy_makespan}, "
                  f"а оптимум={global_opt}. Невозможно нормализовать в [opt, greedy].")
            # Можно пропустить построение графика
        else:
            norm_mean = (mean_progress - global_opt) / (greedy_makespan - global_opt) * 100
            norm_p25 = (p25_progress - global_opt) / (greedy_makespan - global_opt) * 100
            norm_p50 = (p50_progress - global_opt) / (greedy_makespan - global_opt) * 100
            norm_p75 = (p75_progress - global_opt) / (greedy_makespan - global_opt) * 100

            plt.figure(figsize=(10, 6))
            plt.plot(norm_mean, label='Среднее (%)')
            plt.plot(norm_p50, label='Медиана (%)')
            plt.fill_between(range(max_generations), norm_p25, norm_p75,
                             color='gray', alpha=0.2, label='25%-75% перцентили')

            # Линия 0% - соответствие оптимуму
            plt.axhline(0, color='red', linestyle='--', label='Оптимум (0%)')
            # Линия 100% - соответствие жадному решению
            plt.axhline(100, color='orange', linestyle='--', label='Жадный (100%)')

            plt.title(f'{benchmark}: Нормализация [Оптимум=0%, Жадный=100%]')
            plt.xlabel('Поколение')
            plt.ylabel('Отклонение (%)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            save_path_norm = os.path.join(output_dir, f"{benchmark}_normalized_0_opt_100_greedy.png")
            plt.savefig(save_path_norm)
            plt.close()

        # Среднее и перцентили лучшего makespan (в абсолютных единицах)
        plt.figure(figsize=(10, 6))
        plt.plot(mean_progress, label='Среднее значение')
        plt.plot(p50_progress, label='Медиана значения')
        plt.fill_between(range(max_generations), p25_progress, p75_progress, color='gray', alpha=0.2,
                         label='25%-75% перцентили')
        plt.axhline(global_opt, color='red', linestyle='--', label='Глобальный оптимум')
        plt.axhline(greedy_makespan, color='orange', linestyle='--', label='Жадное решение')
        plt.title(f'{benchmark}: Среднее и перцентили лучшего makespan')
        plt.xlabel('Поколение')
        plt.ylabel('Лучший makespan')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path_mean = os.path.join(output_dir, f"{benchmark}_mean_percentiles.png")
        plt.savefig(save_path_mean)
        plt.close()

        # Кривые эффективности (абсолютные значения лучшего особя)
        plt.figure(figsize=(10, 6))
        for r in range(num_runs):
            plt.plot(all_runs_fitness_progress[r], alpha=0.3, color='blue')
        plt.title(f'{benchmark}: Все кривые эффективности (лучшая особь по поколениям)')
        plt.xlabel('Поколение')
        plt.ylabel('Лучший makespan')
        plt.grid(True)
        plt.tight_layout()
        save_path_curves = os.path.join(output_dir, f"{benchmark}_efficiency_curves.png")
        plt.savefig(save_path_curves)
        plt.close()

        # Стандартное отклонение по поколениям (в абсолютных единицах)
        plt.figure(figsize=(10, 6))
        plt.plot(std_progress, label='Стандартное отклонение')
        plt.title(f'{benchmark}: Стандартное отклонение makespan по поколениям')
        plt.xlabel('Поколение')
        plt.ylabel('Ст. отклонение makespan')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path_std = os.path.join(output_dir, f"{benchmark}_std_progress.png")
        plt.savefig(save_path_std)
        plt.close()

        # Гистограмма финальных результатов (лучшие из каждого запуска, в абсолютных единицах)
        plt.figure(figsize=(10, 6))
        plt.hist(final_best_results, bins=20, edgecolor='black', alpha=0.7, color='green')
        plt.title(f'{benchmark}: Распределение итоговых лучших результатов')
        plt.xlabel('Лучший makespan (финал каждого из запусков)')
        plt.ylabel('Частота')
        plt.grid(True)
        plt.tight_layout()
        save_path_hist = os.path.join(output_dir, f"{benchmark}_final_results_hist.png")
        plt.savefig(save_path_hist)
        plt.close()

        # Таблица результатов всех запусков
        table_data = []
        for i, val in enumerate(final_best_results, start=1):
            run_best = np.min(all_runs_fitness_progress[i - 1])
            run_average = np.mean(all_runs_fitness_progress[i - 1])
            run_worst = np.max(all_runs_fitness_progress[i - 1])
            table_data.append([
                i,
                f"{val:.2f}",
                f"{run_average:.2f}",
                f"{run_worst:.2f}"
            ])

        print(f"\nТаблица результатов (для {benchmark}):")
        print(f"{'Запуск':>6} | {'Лучший итоговый makespan':>25} | {'Среднее в запуске':>17} | {'Худшее в запуске':>17}")
        print("-" * 80)
        for row in table_data:
            print(f"{row[0]:6d} | {row[1]:25} | {row[2]:17} | {row[3]:17}")

        print(f"\nСтатистика по итоговым результатам для {benchmark}:")
        print(f"Среднее значение: {np.mean(final_best_results):.2f}")
        print(f"Стандартное отклонение: {np.std(final_best_results):.2f}")
        print(f"25%: {np.percentile(final_best_results, 25):.2f}, "
              f"50% (медиана): {np.percentile(final_best_results, 50):.2f}, "
              f"75%: {np.percentile(final_best_results, 75):.2f}")

        # Сохраняем таблицу результатов для данного бенчмарка
        table_save_path = os.path.join(output_dir, f"{benchmark}_results_table.png")
        create_table_figure(
            title=f"Таблица результатов (для {benchmark})",
            columns=["Запуск", "Лучший итоговый фитнесс", "Среднее в запуске", "Худшее в запуске"],
            data=table_data,
            save_path=table_save_path
        )

        # Итоговые средние результаты по этому бенчмарку
        final_mean = np.mean(final_best_results)
        final_deviation = final_mean - global_opt
        final_table_data_all.append([benchmark, final_mean, final_deviation])

    print("\n=== Итоговая таблица по всем бенчмаркам ===")
    print("Benchmark        | Среднее | Отклонение от глобального оптимального | Отклонение (%)")
    print("-------------------------------------------------------------------------")

    # Сортируем, если бенчмарков несколько
    final_table_data_all_sorted = sorted(final_table_data_all)

    final_table_with_percent = []
    for row in final_table_data_all_sorted:
        benchmark, mean_val, deviation = row
        deviation_percent = ((mean_val - global_optimum) / global_optimum) * 100
        final_table_with_percent.append([benchmark, mean_val, deviation, deviation_percent])
        print(f"{benchmark:16s} | {mean_val:.2f} | {deviation:.2f} | {deviation_percent:.2f}%")

    final_table_formatted = []
    for row in final_table_with_percent:
        benchmark, mean_val, deviation, deviation_percent = row
        final_table_formatted.append([
            benchmark,
            f"{mean_val:.2f}",
            f"{deviation:.2f}",
            f"{deviation_percent:.2f}%"
        ])

    final_table_save_path = os.path.join(output_dir, f"final_table_all_benchmarks.png")
    create_table_figure(
        title="Итоговая таблица по всем бенчмаркам",
        columns=["Benchmark", "Среднее", "Отклонение от глобального оптимального", "Отклонение (%)"],
        data=final_table_formatted,
        save_path=final_table_save_path
    )

    print(f"\nВсе графики и таблицы сохранены в директорию '{output_dir}'.")
