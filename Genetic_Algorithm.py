import random
import numpy as np
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter  # 引入TensorBoard

# 输入数据
classrooms = ["2101", "2102", "2103"]
classes = ["1班", "2班", "3班"]
courses = {
    "52005": "高等数学",
    "53202": "线性代数",
    "53215": "概率统计",
    "53230": "运筹学",
    "54007": "计算机应用基础",
    "54200": "微机原理与接口技术",
    "54459": "数据结构",
    "54569": "数字图像处理",
    "54830": "人工智能基础",
    "54976": "嵌入式智能系统"
}
teachers = {
    "T1": {"name": "冯桥", "courses": ["52005", "53202"]},
    "T2": {"name": "赵思", "courses": ["53215"]},
    "T3": {"name": "赵宏生", "courses": ["54007"]},
    "T4": {"name": "赵国", "courses": ["54007"]},
    "T5": {"name": "董南振", "courses": ["53230", "54200"]},
    "T6": {"name": "祝毅勇", "courses": ["53230"]},
    "T7": {"name": "李兴", "courses": ["54459", "54569"]},
    "T8": {"name": "赵荣正", "courses": ["54830"]},
    "T9": {"name": "王磊宏", "courses": ["54976"]},
    "T10": {"name": "周友国", "courses": ["54830"]},
    "T11": {"name": "孙文", "courses": ["54976"]},
    "T12": {"name": "项峰炎", "courses": ["54830"]},
    "T13": {"name": "项健德", "courses": ["54569", "54976"]},
    "T14": {"name": "李艺祥", "courses": ["54830", "54976"]},
    "T15": {"name": "祝桥", "courses": ["54830", "54976"]},
    "T16": {"name": "郑宏文", "courses": ["54569", "54976"]},
    "T17": {"name": "赵国健", "courses": ["54976"]},
    "T18": {"name": "郑军锦", "courses": ["52005", "53202"]},
    "T19": {"name": "董友裕", "courses": ["53230", "54200"]},
    "T20": {"name": "童仁兴", "courses": ["53230"]},
    "T21": {"name": "陈南坚", "courses": ["54459"]},
    "T22": {"name": "项友志", "courses": ["54200", "54976"]}
}
schedule_requirements = {
    "1班": {
        "52005": 1,
        "53202": 1,
        "53215": 1,
        "53230": 1,
        "54007": 2,
        "54200": 0,
        "54459": 0,
        "54569": 1,
        "54830": 1,
        "54976": 1
    },
    "2班": {
        "52005": 1,
        "53202": 1,
        "53215": 1,
        "53230": 1,
        "54007": 0,
        "54200": 2,
        "54459": 0,
        "54569": 1,
        "54830": 1,
        "54976": 1
    },
    "3班": {
        "52005": 1,
        "53202": 1,
        "53215": 1,
        "53230": 1,
        "54007": 0,
        "54200": 0,
        "54459": 2,
        "54569": 1,
        "54830": 1,
        "54976": 1
    }
}
time_slots = {
    1: {"start": "8:00", "end": "9:50"},
    2: {"start": "10:10", "end": "12:00"},
    3: {"start": "14:00", "end": "15:50"},
    4: {"start": "16:10", "end": "18:00"},
    5: {"start": "19:30", "end": "21:20"}
}
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

# 遗传算法参数
POPULATION_SIZE = 500   # 种群大小
MUTATION_RATE = 0.01    # 变异率
CROSSOVER_RATE = 0.7    # 交叉率
GENERATIONS = 10000     # 最大代数
TOURNAMENT_SIZE = 7     # 锦标赛选择的大小
ELITE_SIZE = 2          # 精英保留策略，保留前2个优质个体


# 课程安排类
class CourseArrangement:
    def __init__(self, class_name, course_id, teacher_id, classroom, day, time_slot):
        self.class_name = class_name
        self.course_id = course_id
        self.teacher_id = teacher_id
        self.classroom = classroom
        self.day = day
        self.time_slot = time_slot

    def __str__(self):
        return f"{self.class_name} {courses[self.course_id]} \
	        {teachers[self.teacher_id]['name']}{self.classroom} {days[self.day - 1]} \
	        {time_slots[self.time_slot]['start']}-{time_slots[self.time_slot]['end']}"


# 生成初始种群
def generate_initial_population():
    population = []
    for _ in range(POPULATION_SIZE):
        schedule = []
        for class_name in classes:
            for course_id, weekly_hours in schedule_requirements[class_name].items():
                for _ in range(weekly_hours):
                    possible_teachers = [t \
                                         for t in teachers if course_id in teachers[t]["courses"]]
                    teacher_id = random.choice(possible_teachers)
                    classroom = random.choice(classrooms)
                    day = random.randint(1, 6)
                    time_slot = random.randint(1, 5)
                    schedule.append(CourseArrangement(class_name, course_id, teacher_id, \
                                                      classroom, day, time_slot))
        population.append(schedule)
    return population


# 计算适应度
def calculate_fitness(individual):
    conflict_count = 0
    course_spread = 0
    class_spread = 0

    teacher_schedule = {t: set() for t in teachers}
    for arr in individual:
        time_key = (arr.day, arr.time_slot)
        if time_key in teacher_schedule[arr.teacher_id]:
            conflict_count += 1
        else:
            teacher_schedule[arr.teacher_id].add(time_key)

    classroom_schedule = {c: set() for c in classrooms}
    for arr in individual:
        time_key = (arr.day, arr.time_slot)
        if time_key in classroom_schedule[arr.classroom]:
            conflict_count += 1
        else:
            classroom_schedule[arr.classroom].add(time_key)

    class_schedule = {cls: set() for cls in classes}
    for arr in individual:
        time_key = (arr.day, arr.time_slot)
        if time_key in class_schedule[arr.class_name]:
            conflict_count += 1
        else:
            class_schedule[arr.class_name].add(time_key)

    course_days = {}
    for arr in individual:
        if (arr.class_name, arr.course_id) not in course_days:
            course_days[(arr.class_name, arr.course_id)] = set()
        course_days[(arr.class_name, arr.course_id)].add(arr.day)

    for key in course_days:
        days_used = sorted(course_days[key])
        for i in range(1, len(days_used)):
            course_spread += (days_used[i] - days_used[i - 1] - 1)

    class_day_counts = {cls: [0] * 7 for cls in classes}
    for arr in individual:
        class_day_counts[arr.class_name][arr.day] += 1

    for cls in classes:
        avg = sum(class_day_counts[cls][1:7]) / 6
        for day in range(1, 7):
            class_spread += abs(class_day_counts[cls][day] - avg)

    fitness = 1 / (conflict_count + 1)
    fitness += course_spread * 0.01
    fitness += class_spread * 0.01

    return fitness


# 选择操作（锦标赛选择）
def select_parent(population):
    selected = random.sample(population, TOURNAMENT_SIZE)
    return max(selected, key=lambda x: x[1])


# 交叉操作
def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1

    child = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])

    # 冲突解决：教师时间冲突
    teacher_schedule = {t: set() for t in teachers}
    for arr in child:
        time_key = (arr.day, arr.time_slot)
        attempt = 0
        max_attempts = 100  # 设置最大尝试次数
        while time_key in teacher_schedule[arr.teacher_id] and attempt < max_attempts:
            arr.day = random.randint(1, 6)
            arr.time_slot = random.randint(1, 5)
            time_key = (arr.day, arr.time_slot)
            attempt += 1
        teacher_schedule[arr.teacher_id].add(time_key)

    # 冲突解决：教室时间冲突
    classroom_schedule = {c: set() for c in classrooms}
    for arr in child:
        time_key = (arr.day, arr.time_slot)
        attempt = 0
        while time_key in classroom_schedule[arr.classroom] and attempt < max_attempts:
            arr.classroom = random.choice(classrooms)
            time_key = (arr.day, arr.time_slot)
            attempt += 1
        classroom_schedule[arr.classroom].add(time_key)

    # 冲突解决：班级时间冲突
    class_schedule = {cls: set() for cls in classes}
    for arr in child:
        time_key = (arr.day, arr.time_slot)
        attempt = 0
        while time_key in class_schedule[arr.class_name] and attempt < max_attempts:
            arr.day = random.randint(1, 6)
            arr.time_slot = random.randint(1, 5)
            time_key = (arr.day, arr.time_slot)
            attempt += 1
        class_schedule[arr.class_name].add(time_key)

    return child


# 变异操作
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            possible_teachers = [t for t in teachers \
                                 if individual[i].course_id in teachers[t]["courses"]]
            individual[i].teacher_id = random.choice(possible_teachers)
            individual[i].classroom = random.choice(classrooms)
            individual[i].day = random.randint(1, 6)
            individual[i].time_slot = random.randint(1, 5)

            # 冲突解决：教师时间冲突
            teacher_schedule = {t: set() for t in teachers}
            time_key = (individual[i].day, individual[i].time_slot)
            attempt = 0
            max_attempts = 100  # 设置最大尝试次数
            while time_key in teacher_schedule[individual[i].teacher_id] \
                    and attempt < max_attempts:
                individual[i].day = random.randint(1, 6)
                individual[i].time_slot = random.randint(1, 5)
                time_key = (individual[i].day, individual[i].time_slot)
                attempt += 1
            teacher_schedule[individual[i].teacher_id].add(time_key)

            # 冲突解决：教室时间冲突
            classroom_schedule = {c: set() for c in classrooms}
            time_key = (individual[i].day, individual[i].time_slot)
            attempt = 0
            while time_key in classroom_schedule[individual[i].classroom] \
                    and attempt < max_attempts:
                individual[i].classroom = random.choice(classrooms)
                time_key = (individual[i].day, individual[i].time_slot)
                attempt += 1
            classroom_schedule[individual[i].classroom].add(time_key)

            # 冲突解决：班级时间冲突
            class_schedule = {cls: set() for cls in classes}
            time_key = (individual[i].day, individual[i].time_slot)
            attempt = 0
            while time_key in class_schedule[individual[i].class_name] \
                    and attempt < max_attempts:
                individual[i].day = random.randint(1, 6)
                individual[i].time_slot = random.randint(1, 5)
                time_key = (individual[i].day, individual[i].time_slot)
                attempt += 1
            class_schedule[individual[i].class_name].add(time_key)

    return individual

# 遗传算法主循环
def genetic_algorithm():
    population = generate_initial_population()
    fitness_scores = [calculate_fitness(ind) for ind in population]

    # 初始化TensorBoard
    writer = SummaryWriter('runs/scheduling_experiment')

    # 用于保存每100轮的最大适应度值
    max_fitness_every_100 = []

    # 记录过程中最大适应度值及其对应的个体
    max_fitness = max(fitness_scores)
    max_fitness_individual = population[np.argmax(fitness_scores)]

    # 用于判断适应度值波动是否小于等于0.01
    fitness_history = [max_fitness]

    for generation in range(GENERATIONS):
        # 每100代输出一次进度
        if (generation + 1) % 100 == 0:
            best_index = np.argmax(fitness_scores)
            print(
                f"Generation {generation + 1}, Best Fitness: {fitness_scores[best_index]:.4f}, Remaining Generations: {GENERATIONS - (generation + 1)}")
            max_fitness_every_100.append((generation + 1, fitness_scores[best_index]))

        # 记录每一代的最佳适应度值到TensorBoard
        current_best_fitness = max(fitness_scores)
        writer.add_scalar('Best Fitness', current_best_fitness, generation)

        # 更新过程中最大适应度值及其对应的个体
        if current_best_fitness > max_fitness:
            max_fitness = current_best_fitness
            max_fitness_individual = population[np.argmax(fitness_scores)]

        # 添加到适应度历史记录
        fitness_history.append(current_best_fitness)
        if len(fitness_history) > 10:  # 保留最近10代的适应度值
            fitness_history.pop(0)

        # 计算适应度值的波动
        if len(fitness_history) >= 10:
            fitness_std = np.std(fitness_history)
            if fitness_std <= 0.01:
                print(f"Adaptation value fluctuation is less than or equal to 0.01, stop at generation {generation + 1}")
                break

        # 精英保留策略：保留前ELITE_SIZE个优质个体
        elite_indices = np.argsort(fitness_scores)[-ELITE_SIZE:]
        elite_individuals = [population[i] for i in elite_indices]

        # 选择下一代
        new_population = elite_individuals.copy()  # 添加精英个体到新种群
        while len(new_population) < POPULATION_SIZE:
            parent1 = select_parent(list(zip(population, fitness_scores)))[0]
            parent2 = select_parent(list(zip(population, fitness_scores)))[0]
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
        fitness_scores = [calculate_fitness(ind) for ind in population]

        # 如果达到10000轮次，停止
        if generation + 1 >= 10000:
            print("Reached 10000 generations, stop")
            break

    # 关闭TensorBoard
    writer.close()

    # 将每100轮的最大适应度值保存到文件
    with open('max_fitness_every_100.txt', 'w') as f:
        for gen, fitness in max_fitness_every_100:
            f.write(f"Generation {gen}, Best Fitness: {fitness:.4f}\n")

    # 确保最终答案的适应度不小于过程中最大适应度
    final_best_index = np.argmax(fitness_scores)
    final_best_fitness = fitness_scores[final_best_index]
    if final_best_fitness >= max_fitness:
        best_individual = population[final_best_index]
    else:
        best_individual = max_fitness_individual

    return best_individual, max_fitness, max_fitness_individual


# 格式化输出课表
def format_schedule(schedule):
    sorted_schedule = sorted(schedule, key=lambda x: (x.class_name, x.day, x.time_slot))

    table = PrettyTable()
    table.field_names = ["班级", "课程", "教师", "教室", "星期", "时间"]

    for arr in sorted_schedule:
        time_str = f"{time_slots[arr.time_slot]['start']}-{time_slots[arr.time_slot]['end']}"
        table.add_row([arr.class_name, courses[arr.course_id],
                       teachers[arr.teacher_id]['name'],
                       arr.classroom, days[arr.day - 1], time_str])

    return table


# 主函数
if __name__ == "__main__":
    best_schedule, max_fitness, max_fitness_individual = genetic_algorithm()
    best_fitness = calculate_fitness(best_schedule)  # 计算最佳排课方案的适应度值

    # 确保最终答案的适应度不小于过程中最大适应度
    if best_fitness < max_fitness:
        best_schedule = max_fitness_individual
        best_fitness = max_fitness

    print("\n最佳排课方案：")
    schedule_table = format_schedule(best_schedule)
    print(schedule_table)
    print(f"\n最佳适应度值：{best_fitness:.4f}")  # 打印最佳适应度值

    # 将表格和适应度值保存到文件
    with open('scheduling_result.txt', 'w') as f:
        f.write(str(schedule_table))
        f.write(f"\n最佳适应度值：{best_fitness:.4f}")