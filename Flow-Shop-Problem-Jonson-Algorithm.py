import random
import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------------------------------------
# Пояснение по фрагментам кода:
# -----------------------------------------------------------
# Часть: Генерация данных.
# Мы создаем N задач, для каждой задачи генерируем время обработки на M1 и M2.
# Используем random.randint для получения случайных значений.
# Затем сохраняем их в списки или DataFrame для удобства.
#
# Переменные:
# N - количество задач
# tasks - список кортежей (id, t1, t2), где id - номер задачи, t1 - время на M1, t2 - время на M2.
# -----------------------------------------------------------

def generate_tasks(N, low=1, high=10):
    tasks = []
    for i in range(N):
        t1 = random.randint(low, high)
        t2 = random.randint(low, high)
        tasks.append((i + 1, t1, t2))
    return tasks


# -----------------------------------------------------------
# Часть: Алгоритм Джонсона.
# Функция принимает список задач вида (id, t1, t2).
# 1. Разделяем задачи на две группы:
#    G1: t1 <= t2 (сортировать по t1 по возрастанию)
#    G2: t1 > t2 (сортировать по t2 по убыванию)
# 2. Конечный порядок = G1 + G2
#
# Возвращается оптимальный порядок задач.
# -----------------------------------------------------------

def johnson_algorithm(tasks):
    # Разбиваем на группы
    G1 = [task for task in tasks if task[1] <= task[2]]
    G2 = [task for task in tasks if task[1] > task[2]]

    # Сортируем G1 по возрастанию t1
    G1.sort(key=lambda x: x[1])
    # Сортируем G2 по убыванию t2
    G2.sort(key=lambda x: x[2], reverse=True)

    # Финальный порядок
    return G1 + G2


# -----------------------------------------------------------
# Часть: Расчет расписания.
# Имея порядок задач, нужно вычислить временные интервалы:
# - Когда каждая задача начинает и заканчивает обработку на M1.
# - Когда каждая задача начинает и заканчивает обработку на M2.
#
# Правила:
# - Первая задача начинает на M1 в момент 0, M2 может начать только после окончания M1 этой же задачи.
# - Каждая следующая задача может начать на M1 только после того, как M1 освободится.
# - На M2 задача может начать, только когда:
#   a) M2 освободится, и
#   b) задача завершит обработку на M1.
#
# В итоге получаем два списка интервалов или DataFrame со стартом и концом для каждой задачи.
# -----------------------------------------------------------

def compute_schedule(order):
    # order - список задач в порядке выполнения [(id, t1, t2), ...]

    # Время, когда M1 освободится для новой задачи
    time_m1 = 0
    # Время, когда M2 освободится для новой задачи
    time_m2 = 0

    schedule = []
    for (task_id, t1, t2) in order:
        # Задача начинает на M1, как только M1 станет свободен
        start_m1 = time_m1
        finish_m1 = start_m1 + t1
        time_m1 = finish_m1  # M1 будет свободен после этого времени

        # На M2 задача может начаться не раньше, чем задача закончится на M1 (finish_m1),
        # и не раньше, чем M2 освободится (time_m2)
        start_m2 = max(finish_m1, time_m2)
        finish_m2 = start_m2 + t2
        time_m2 = finish_m2  # M2 будет свободен после этого времени

        schedule.append({
            'Task': task_id,
            'M1_start': start_m1,
            'M1_finish': finish_m1,
            'M2_start': start_m2,
            'M2_finish': finish_m2
        })
    return schedule


# -----------------------------------------------------------
# Часть: Визуализация.
# Создадим простой график (диаграмму Ганта) для визуализации расписания.
#
# Мы используем matplotlib.pyplot.barh для отображения интервалов задач на M1 и M2.
# Задачи будем отображать горизонтальными полосками.
#
# Также можно вывести результаты в таблице (через pandas DataFrame).
# -----------------------------------------------------------

def plot_schedule(schedule):
    # Преобразуем schedule в DataFrame для удобства
    df = pd.DataFrame(schedule)
    # Задачи идут по порядку их выполнения
    df = df.sort_values(by='M1_start')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Диаграмма для M1
    for i, row in df.iterrows():
        ax1.barh(y=row['Task'], width=row['M1_finish'] - row['M1_start'], left=row['M1_start'], color='skyblue')
        ax1.text(row['M1_start'] + (row['M1_finish'] - row['M1_start']) / 2, row['Task'],
                 f"J{row['Task']}", ha='center', va='center', color='black')
    ax1.set_title('Machine M1')
    ax1.set_ylabel('Task')
    ax1.invert_yaxis()  # Чтобы задачи шли сверху вниз
    ax1.grid(True, axis='x')

    # Диаграмма для M2
    for i, row in df.iterrows():
        ax2.barh(y=row['Task'], width=row['M2_finish'] - row['M2_start'], left=row['M2_start'], color='orange')
        ax2.text(row['M2_start'] + (row['M2_finish'] - row['M2_start']) / 2, row['Task'],
                 f"J{row['Task']}", ha='center', va='center', color='black')
    ax2.set_title('Machine M2')
    ax2.set_xlabel('Time')
    ax2.invert_yaxis()  # Чтобы задачи шли сверху вниз
    ax2.grid(True, axis='x')

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# Основной блок выполнения:
# 1. Генерируем данные.
# 2. Применяем алгоритм Джонсона.
# 3. Считаем расписание.
# 4. Выводим таблицу с результатами и строим график.
# -----------------------------------------------------------

if __name__ == "__main__":
    # Количество задач
    N = 8
    # Генерируем задачи
    tasks = generate_tasks(N, low=1, high=10)

    # Вывод сгенерированных задач
    print("Generated tasks (id, t1, t2):")
    for t in tasks:
        print(t)

    # Применяем алгоритм Джонсона
    order = johnson_algorithm(tasks)
    print("\nOptimal order by Johnson's algorithm:")
    print([o[0] for o in order])  # выводим только ID задач

    # Считаем расписание
    schedule = compute_schedule(order)

    # Выводим расписание в табличном виде
    df_schedule = pd.DataFrame(schedule)
    print("\nSchedule:")
    print(df_schedule)

    # Строим график (диаграмма Ганта)
    plot_schedule(schedule)
