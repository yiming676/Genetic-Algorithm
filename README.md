# Genetic-Algorithm
这是一个基于遗传算法（GA）的排课程序

## 实验原理

### 遗传算法
遗传算法（Genetic Algorithm, GA）是一种受到生物进化启发的全局优化搜索算法，由美国学者约翰·霍兰德（John Holland）于20世纪70年代提出。它通过模拟自然选择和遗传机制，在解空间中搜索最优解。遗传算法的主要特点是直接对结构对象进行操作，不需要梯度信息，具有内在的并行性和全局搜索能力。

### 排课问题描述
课程表生成是一个典型的组合优化问题，需要在满足一系列约束条件下，为课程、教师、教室和时间分配合理的组合。具体约束包括：

#### 硬约束
- **时间约束：** 同一教师不能在同一时间教多门课程，同一教室不能在同一时间容纳多门课程，同一班级不能在同一时间上多门课程。
- **空间约束：** 每门课程需要分配到一个教室，教室资源有限。

#### 软约束
- 课程应合理分布在不同时间段，避免过度集中或分散。

### 解题逻辑

#### 编码
将课程安排表示为一个个体（染色体），每个基因代表一门课程的安排，包括班级、课程ID、教师ID、教室、星期和时间段。

#### 遗传参数选择
```python
POPULATION_SIZE = 500   # 种群大小
MUTATION_RATE = 0.01    # 变异率
CROSSOVER_RATE = 0.7    # 交叉率
GENERATIONS = 10000     # 最大代数
TOURNAMENT_SIZE = 7     # 锦标赛选择的大小
ELITE_SIZE = 2          # 精英保留策略，保留前2个优质个体
```

#### 初始种群生成
随机生成500个个体，构成初始种群。每个个体代表一种可能的课程安排方案。
```python
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
```

#### 适应度函数
定义适应度函数来评估每个个体的优劣。适应度函数综合考虑了时间冲突、课程分布合理性等因素。

```python
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
```

 ### 选择操作
 采用锦标赛选择（Tournament Selection）（每轮锦标赛挑选7个个体），从当前种群中选择适应度 
 较高的个体作为父代，用于产生下一代。

 ```python
 def select_parent(population):
     selected = random.sample(population, TOURNAMENT_SIZE)
     return max(selected, key=lambda x: x[1])
 ```
 ### 交叉操作
 以 0.9 的概率（交叉率）对两个父代个体进行交叉，生成新的子代个体。交叉过程中，确保新生成 
 的课程安排满足基本的时间和空间约束。

 ```python
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
 ```

 ### 变异操作
 以 0.01 的概率（变异率）对个体中的某些基因进行变异，改变课程的教师、教室、时间等属性，增加种群的多样性。

 ```python
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
```

 ### 进化流程（主循环）
 重复选择、交叉和变异操作，不断进化种群，直到达到预设的代数（10000代），或者当适应度变化不超过0.01。每100代输出一次当前最大适应度值。最佳答案的适应度不能小于整个过程中的最大适应度（即结束时如果小于过程中最大值，则选择过程中最大值作为最佳结果。）
 ```python
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
```

 ### 进化结果
 进化结束后，输出最佳适应度对应的课程安排方案，并以表格形式展示。
 ```python
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
```
## 实验结果
### 适应度值
最终停止在第 7288 代。
每100轮次最大适应度值如下表：
| Generation | Best Fitness | Generation | Best Fitness |
|------------|--------------|------------|--------------|
| 100        | 1.2100       | 3700       | 1.2200       |
| 200        | 1.1600       | 3800       | 1.2100       |
| 300        | 1.1600       | 3900       | 1.2100       |
| 400        | 0.7000       | 4000       | 1.1600       |
| 500        | 0.6300       | 4100       | 1.1500       |
| 600        | 0.4833       | 4200       | 1.1000       |
| 700        | 1.2000       | 4300       | 0.6800       |
| 800        | 0.6400       | 4400       | 1.2100       |
| 900        | 1.2100       | 4500       | 1.1900       |
| 1000       | 1.1400       | 4600       | 1.2000       |
| 1100       | 1.2100       | 4700       | 0.6900       |
| 1200       | 1.1700       | 4800       | 1.1600       |
| 1300       | 1.2200       | 4900       | 1.1400       |
| 1400       | 0.6500       | 5000       | 1.2200       |
| 1500       | 1.1900       | 5100       | 1.1500       |
| 1600       | 1.1600       | 5200       | 1.2100       |
| 1700       | 1.1400       | 5300       | 0.5233       |
| 1800       | 1.1800       | 5400       | 1.1300       |
| 1900       | 1.2100       | 5500       | 1.1500       |
| 2000       | 0.6700       | 5600       | 1.2300       |
| 2100       | 1.1400       | 5700       | 1.1900       |
| 2200       | 1.1600       | 5800       | 1.2300       |
| 2300       | 0.7100       | 5900       | 1.1900       |
| 2400       | 1.2000       | 6000       | 1.1600       |
| 2500       | 1.1500       | 6100       | 0.6600       |
| 2600       | 1.1600       | 6200       | 1.2000       |
| 2700       | 0.6500       | 6300       | 1.1600       |
| 2800       | 1.2300       | 6400       | 1.1800       |
| 2900       | 1.2200       | 6500       | 1.1500       |
| 3000       | 1.1900       | 6600       | 1.2200       |
| 3100       | 1.1900       | 6700       | 1.2200       |
| 3200       | 1.1900       | 6800       | 1.2300       |
| 3300       | 1.1900       | 6900       | 1.2100       |
| 3400       | 1.1100       | 7000       | 1.1900       |
| 3500       | 1.2000       | 7100       | 1.1700       |
| 3600       | 0.6600       | 7200       | 1.1800       |

### 最佳排课方案
该最佳方案的适应度值为 1.32
| 班级 | 课程                  | 教师    | 教室  | 星期 | 时间           |
|------|-----------------------|---------|-------|------|----------------|
| 1班  | 概率统计              | 赵思    | 2101  | Mon  | 8:00-9:50      |
| 1班  | 计算机应用基础        | 赵宏生  | 2103  | Mon  | 10:10-12:00    |
| 1班  | 运筹学                | 童仁兴  | 2103  | Mon  | 14:00-15:50    |
| 1班  | 人工智能基础          | 项峰炎  | 2101  | Mon  | 19:30-21:20    |
| 1班  | 数字图像处理          | 项健德  | 2103  | Tue  | 16:10-18:00    |
| 1班  | 嵌入式智能系统        | 项友志  | 2103  | Wed  | 10:10-12:00    |
| 1班  | 高等数学              | 郑军锦  | 2103  | Thu  | 19:30-21:20    |
| 1班  | 线性代数              | 冯桥    | 2103  | Fri  | 14:00-15:50    |
| 1班  | 计算机应用基础        | 赵宏生  | 2102  | Fri  | 16:10-18:00    |
| 2班  | 数字图像处理          | 项健德  | 2103  | Mon  | 16:10-18:00    |
| 2班  | 人工智能基础          | 周友国  | 2102  | Wed  | 10:10-12:00    |
| 2班  | 运筹学                | 童仁兴  | 2102  | Wed  | 19:30-21:20    |
| 2班  | 概率统计              | 赵思    | 2101  | Thu  | 10:10-12:00    |
| 2班  | 高等数学              | 冯桥    | 2101  | Thu  | 14:00-15:50    |
| 2班  | 嵌入式智能系统        | 孙文    | 2101  | Fri  | 8:00-9:50      |
| 2班  | 微机原理与接口技术    | 董南振  | 2102  | Fri  | 14:00-15:50    |
| 2班  | 线性代数              | 郑军锦  | 2103  | Fri  | 16:10-18:00    |
| 2班  | 微机原理与接口技术    | 董友裕  | 2102  | Sat  | 8:00-9:50      |
| 3班  | 数据结构              | 李兴    | 2101  | Mon  | 14:00-15:50    |
| 3班  | 运筹学                | 童仁兴  | 2101  | Mon  | 16:10-18:00    |
| 3班  | 数据结构              | 李兴    | 2103  | Tue  | 8:00-9:50      |
| 3班  | 人工智能基础          | 赵荣正  | 2103  | Tue  | 19:30-21:20    |
| 3班  | 数字图像处理          | 郑宏文  | 2102  | Wed  | 8:00-9:50      |
| 3班  | 高等数学              | 冯桥    | 2101  | Thu  | 8:00-9:50      |
| 3班  | 线性代数              | 郑军锦  | 2101  | Fri  | 10:10-12:00    |
| 3班  | 概率统计              | 赵思    | 2101  | Fri  | 19:30-21:20    |
| 3班  | 嵌入式智能系统        | 祝桥    | 2103  | Sat  | 8:00-9:50      |
