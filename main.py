import copy
import datetime
import random
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from utils import initUtils
from function import function1, fun3
from function import function2


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


# 保存原始的 stdout
original_stdout = sys.stdout
f = open("./log/log.txt", "a")
# 将 stdout 重定向到文件
sys.stdout = f

# 将 stdout 重定向到一个 Tee 对象，将输出同时发送到控制台和文件中
sys.stdout = Tee(original_stdout, f)

# 恢复 stdout
# sys.stdout = original_stdout


# vms = init.nodes

# containers = init.containers
containers = initUtils.tasks
vms = initUtils.nodes
# cost_expect = 40
pop_size = 80
MAX_ITER = 100


# 基因交叉编码，产生子代
def cross(individual1, individual2):
    child1 = copy.deepcopy(individual1)
    child2 = copy.deepcopy(individual2)
    length = len(child1["nodes"])
    crossover_point1 = random.randint(0, length - 1)
    child1["nodes"][crossover_point1], child2["nodes"][crossover_point1] = child2["nodes"][crossover_point1], \
                                                                           child1["nodes"][crossover_point1]
    # 将交叉点之后的节点进行位置调整
    for i in range(len(child1["nodes"])):
        if i != crossover_point1:
            # 判断是否重叠
            if is_overlap(child1["nodes"][i], child1["nodes"][crossover_point1]):
                child1["nodes"][i] = move_away(child1["nodes"][i], child1["nodes"][crossover_point1])
            if is_overlap(child2["nodes"][i], child2["nodes"][crossover_point1]):
                child2["nodes"][i] = move_away(child2["nodes"][i], child2["nodes"][crossover_point1])
    return child1, child2


# def cross(individual1, individual2):
#     child1 = copy.deepcopy(individual1)
#     child2 = copy.deepcopy(individual2)
#     length = len(child1["nodes"])
#     crossover_point1 = random.randint(0, length - 1)
#     crossover_point2 = random.randint(0, length - 1)
#     if crossover_point1 == crossover_point2:
#         child1["nodes"][crossover_point1], child2["nodes"][crossover_point1] = child2["nodes"][crossover_point1], \
#                                                                                child1["nodes"][crossover_point1]
#         for i in range(len(child1["nodes"])):
#             if i != crossover_point1:
#                 if is_overlap(child1["nodes"][i], child1["nodes"][crossover_point1]):
#                     child1["nodes"][i] = move_away(child1["nodes"][i], child1["nodes"][crossover_point1])
#                 if is_overlap(child2["nodes"][i], child2["nodes"][crossover_point1]):
#                     child2["nodes"][i] = move_away(child2["nodes"][i], child2["nodes"][crossover_point1])
#     else:
#         child1["nodes"][crossover_point1], child2["nodes"][crossover_point1] = child2["nodes"][crossover_point1], \
#                                                                                child1["nodes"][crossover_point1]
#         child1["nodes"][crossover_point2], child2["nodes"][crossover_point2] = child2["nodes"][crossover_point2], \
#                                                                                child1["nodes"][crossover_point2]
#     for i in range(len(child1["nodes"])):
#         if i != crossover_point1:
#             if is_overlap(child1["nodes"][i], child1["nodes"][crossover_point1]):
#                 child1["nodes"][i] = move_away(child1["nodes"][i], child1["nodes"][crossover_point1])
#             if is_overlap(child2["nodes"][i], child2["nodes"][crossover_point1]):
#                 child2["nodes"][i] = move_away(child2["nodes"][i], child2["nodes"][crossover_point1])
#     return child1, child2


def mute(individual):
    child = copy.deepcopy(individual)
    length = len(child["nodes"])
    crossover_point1 = random.randint(0, length - 1)
    child["nodes"][crossover_point1] = vms[crossover_point1]
    correct_solution(child)
    return child


# def cross(individual1, individual2):
#     child1 = copy.deepcopy(individual1)
#     child2 = copy.deepcopy(individual2)
#     length = len(child1)
#     crossover_point1 = random.randint(1, length - 1)
#     for i in range(crossover_point1, length):
#         child1["nodes"][crossover_point1], child2["nodes"][crossover_point1] = child2["nodes"][crossover_point1], \
#                                                                                child1["nodes"][crossover_point1]
#     for i in range(0,crossover_point1-1):
#         for j in range(crossover_point1,length):
#             if is_overlap(child1["nodes"][i], child1["nodes"][j]):
#                 child1["nodes"][i] = move_away(child1["nodes"][i], child1["nodes"][j])
#             if is_overlap(child2["nodes"][i], child2["nodes"][j]):
#                 child2["nodes"][i] = move_away(child2["nodes"][i], child2["nodes"][j])
#     return child1,child2


def is_overlap(rect1, rect2):
    if set(rect1["place"]) & set(rect2["place"]):
        return True
    return False


def move_away(rect1, rect2):
    rect1["place"] = [x for x in rect1["place"] if x not in rect2["place"]]
    return rect1


def correct_solution(individual):
    nodes = individual['nodes']
    add = []
    all = []
    for i in range(len(nodes)):
        all.extend(nodes[i]["place"])
    for i in range(len(containers)):
        if i not in all:
            add.append(i)
    for i in add:
        node = random.choice(nodes)
        container = containers[i]
        node["place"].append(container["id"])
        i = 0
        while True:
            i += 1
            rank_containers = []
            for container_id in node['place']:
                if container['rank'] == containers[container_id]['rank']:
                    rank_containers.append(container_id)
            cpu = 0
            mem = 0
            net = 0
            gpu = 0
            for container_id in rank_containers:
                cpu = cpu + containers[container_id]['cpu']
                mem = mem + containers[container_id]['mem']
                net = net + containers[container_id]['net']
                gpu = gpu + containers[container_id]['gpu']
            #     注释，避免死循环
            if container['cpu'] <= node['cpu'] - cpu and container['mem'] <= node['mem'] - mem and container['net'] <= \
                    node['net'] - net and container['gpu'] <= node['gpu'] - gpu or i > len(individual['nodes']) * 2:
                node["place"].append(container["id"])
                break
            else:
                node = random.choice(nodes)


def remove_duplicate(population):
    new_populations = []
    for item in population:
        if item not in new_populations:
            new_populations.append(item)
    return new_populations


def dominate2(p1, p2):
    if function1(p1) < function1(p2) and function2(p1) < function2(p2):
        return True
    if function1(p1) < function1(p2) and function2(p1) <= function2(p2):
        return True
    if function1(p1) <= function1(p2) and function2(p1) < function2(p2):
        return True
    return False


def dominate3(p1, p2):
    a1 = function1(p1)
    b1 = function2(p1)
    c1 = fun3(p1)

    a2 = function1(p2)
    b2 = function2(p2)
    c2 = fun3(p2)
    if a1 < a2 and b1 < b2 and c1 < c2:
        return True
    # if a1 < a2 and b1 <= b2:
    #     return True
    # if a1 <= a2 and b1 < b2:
    #     return True
    # if a1 < a2 and c1 <= c2:
    #     return True
    # if a1 <= a2 and c1 < c2:
    #     return True
    # if b1 < b2 and c1 <= c2:
    #     return True
    # if b1 <= b2 and c1 < c2:
    #     return True
    return False


def dominate(p1, p2):
    a = [0, 0, 0]
    b = [0, 0, 0]
    a[0] = function1(p1)
    a[1] = function2(p1)
    a[2] = fun3(p1)

    b[0] = function1(p2)
    b[1] = function2(p2)
    b[2] = fun3(p2)
    isBetter = False
    for i in range(3):
        if a[i] < b[i]:
            isBetter = True
        elif a[i] > b[i]:
            return False
    return isBetter


# 快速非支配排序
def fast_nondominated_sort(population):
    fronts = [[]]
    for i in range(len(population)):
        individual = population[i]
        individual['domination_count'] = 0
        individual['dominated_solutions'] = []
        for j in range(len(population)):
            other_individual = population[j]
            if i != j:
                if dominate(individual, other_individual):
                    individual['dominated_solutions'].append(j)
                elif dominate(other_individual, individual):
                    individual['domination_count'] += 1
        # 如果被支配数为0，则为第一层
        if individual['domination_count'] == 0:
            individual['rank'] = 0
            fronts[0].append(individual)
    i = 0
    while len(fronts[i]) > 0:
        temp = []
        for individual in fronts[i]:
            for j in individual['dominated_solutions']:
                other_individual = population[j]
                other_individual['domination_count'] -= 1
                if other_individual['domination_count'] == 0:
                    other_individual['rank'] = i + 1
                    temp.append(other_individual)
        i = i + 1
        fronts.append(temp)
    return fronts


# 计算拥挤度
def calculate_crowding_distance(populations_dominated):
    # print(populations_dominated)
    # print("000000000000000000000000000")
    for front in populations_dominated:
        # print(len(front))
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual['distance'] = 0
            front.sort(key=lambda individual: function1(individual))
            # 边界解的拥挤度设为无穷大
            front[0]['distance'] = 10 ** 9
            front[solutions_num - 1]['distance'] = 10 ** 9
            # 计算拥挤度
            function1_values = [function1(individual) for individual in front]
            function2_values = [function2(individual) for individual in front]
            function3_values = [fun3(individual) for individual in front]
            scale1 = max(function1_values) - min(function1_values)
            scale2 = max(function2_values) - min(function2_values)
            scale3 = max(function3_values) - min(function3_values)
            for i in range(1, solutions_num - 1):
                front[i]['distance'] = abs(function1(front[i + 1]) - function1(front[i - 1])) / (
                        scale1 + 0.00000000000000001) + abs(
                    function2(front[i - 1]) - function2(front[i + 1])) / (
                                               scale2 + 0.00000000000000001) + abs(
                    fun3(front[i - 1]) - fun3(front[i + 1])) / (
                                               scale3 + 0.00000000000000001
                                       )
            front.sort(key=lambda individual: individual['distance'], reverse=True)


def population_init():
    population = []
    for i in range(pop_size):
        nodes = copy.deepcopy(vms)
        individual = {'nodes': vms, 'rank': 0, 'distance': 0, 'dominated_solutions': [], 'domination_count': 0}
        for container in containers:
            node = random.choice(nodes)
            i = 0
            while True:
                i += 1
                rank_containers = []
                for container_id in node['place']:
                    if container['rank'] == containers[container_id]['rank']:
                        rank_containers.append(container_id)
                cpu = 0
                mem = 0
                net = 0
                gpu = 0
                for container_id in rank_containers:
                    cpu = cpu + containers[container_id]['cpu']
                    mem = mem + containers[container_id]['mem']
                    net = net + containers[container_id]['net']
                    gpu = gpu + containers[container_id]['gpu']
                #     注释，避免死循环
                if container['cpu'] <= node['cpu'] - cpu and container['mem'] <= node['mem'] - mem and container[
                    'net'] <= \
                        node['net'] - net and container['gpu'] <= node['gpu'] - gpu or i > len(individual['nodes']) * 2:
                    node["place"].append(container["id"])
                    break
                else:
                    node = random.choice(nodes)
        individual["nodes"] = nodes
        population.append(individual)
    return population


# 选择新种群
def crowd(populations_dominated):
    new_population = []
    for front in populations_dominated:
        func = []
        for sol in front:
            if len(new_population) < pop_size:
                if [function1(sol), function2(sol)] not in func:
                    func.append([function1(sol), function2(sol)])
                    new_population.append(sol)
    return new_population


# 在种群中选择一个个体
def choose():
    # 选择
    random_two = random.sample(population, 2)
    # random_two = np.random.choice(population, 2)
    individual1 = random_two[0]
    individual2 = random_two[1]
    if dominate(individual1, individual2):
        return individual1
    elif dominate(individual2, individual1):
        return individual2
    else:
        if individual1['distance'] > individual2['distance']:
            return individual1
        else:
            return individual2


def judge(individual):
    nodes = individual["nodes"]
    for node in nodes:
        rank_containers = {}
        for container_id in node['place']:
            if containers[container_id]['rank'] not in rank_containers.keys():
                rank_containers[containers[container_id]['rank']] = [container_id]
            else:
                rank_containers[containers[container_id]['rank']].append(container_id)
        # print(rank_containers)
        for rank_container in rank_containers.values():
            cpu = 0
            mem = 0
            net = 0
            gpu = 0
            for container_id in rank_container:
                cpu = cpu + containers[container_id]['cpu']
                mem = mem + containers[container_id]['mem']
                net = net + containers[container_id]['net']
                gpu = gpu + containers[container_id]['gpu']
            if cpu > node['cpu'] or mem > node['mem'] or net > node['net'] or gpu > node['gpu']:
                return False
    return True


def show(ax, x, y, z):
    # 使用scatter方法绘制包含三个变量的散点图
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax = Axes3D(fig)
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_title('3D figure')
    ax.set_xlabel('function1')
    ax.set_ylabel('function2')
    ax.set_zlabel('function3')
    # plt.show()


def preprocess_data(x, y, z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    # 从小到大排序三个数组
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    z_sorted = z[sorted_idx]

    # 找到能够被完全平方根的大小
    size = np.ceil(np.sqrt(len(x))).astype(int)
    new_size = size ** 2

    # 在前面填充0，保证x是有序的
    x_padded = np.pad(x_sorted, (new_size - len(x_sorted), 0), mode='constant', constant_values=x_sorted[0])
    y_padded = np.pad(y_sorted, (new_size - len(y_sorted), 0), mode='constant', constant_values=y_sorted[0])
    z_padded = np.pad(z_sorted, (new_size - len(z_sorted), 0), mode='constant', constant_values=z_sorted[0])

    # 重塑成二维数组
    x_reshaped = np.reshape(x_padded, (size, size))
    y_reshaped = np.reshape(y_padded, (size, size))
    z_reshaped = np.reshape(z_padded, (size, size))

    return x_reshaped, y_reshaped, z_reshaped, x_padded, y_padded, z_padded


def plot_3d(ax, x, y, z):
    x_2d, y_2d, z_2d, x, y, z = preprocess_data(x, y, z)
    # 重塑 x 数组为一个可以用于绘制曲面的二维数组
    # x_2d = reshape_to_perfect_square(x)
    # y_2d = reshape_to_perfect_square(y)
    # z_2d = reshape_to_perfect_square(z)

    # 创建一个3D图像
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # 使用scatter方法绘制包含三个变量的散点图
    # ax.scatter(x, y, z, c='r', marker='o')

    # 绘制曲面
    ax.plot_surface(x_2d, y_2d, z_2d)

    ax.set_title('3D figure')
    ax.set_xlabel('function1')
    ax.set_ylabel('function2')
    ax.set_zlabel('function3')
    # plt.show()
    return x, y, z


def plt_3d_with_flam(ax, x, y, z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    # 设置插值点密度
    xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # 插值
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    # 创建一个3D图形对象
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')

    # 绘制3D曲面图
    ax.plot_surface(xi, yi, zi, cmap='Blues')

    # 添加标签和标题
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Surface Plot')

    # 显示图形
    # plt.show()
    return x, y, z

print("pop_size", pop_size)
print("MAX_ITER", MAX_ITER)
# 打印文件名
print("file_name", initUtils.filename)
# 种群初始化
population = population_init()
# 种群非支配排序
populations_dominated = fast_nondominated_sort(population)
# 计算拥挤度
calculate_crowding_distance(populations_dominated)
# 拥挤度排序
population = crowd(populations_dominated)

for i in range(MAX_ITER):
    print("iterator:" + str(i), datetime.datetime.now())
    # print(i)
    for j in range(pop_size):
        individual1 = choose()
        individual2 = choose()

        # print(individual1)
        # print("***********************")
        child1, child2 = cross(individual1, individual2)
        correct_solution(child1)
        correct_solution(child2)
        population.append(child1)
        population.append(child2)
    k = random.random()
    if k < 0.1:
        individual = random.choice(population)
        child = mute(individual)
        population.append(child)
    # population = remove_duplicate(population)
    populations_dominated = fast_nondominated_sort(population)
    calculate_crowding_distance(populations_dominated)
    population = crowd(populations_dominated)
    # x = []
    # y = []
    # for p in population:
    #     x.append(function1(p))
    #     y.append(function2(p))
    # plt.scatter(x, y)
    # plt.show()
# 计算拥挤度
populations_dominated = fast_nondominated_sort(population)

x = []
y = []
z = []

for p in populations_dominated[0]:
    x.append(function1(p))
    y.append(function2(p))
    z.append(fun3(p))
    print(p)
print("solution num:", len(x))

for i in range(len(x)):
    print("function1:", x[i], "function2:", y[i], "function3:", z[i], flush=True)
result = Counter(x)
print(result)

# 绘制第一个3D图
fig = plt.figure()
ax1 = fig.add_subplot(131, projection='3d')
show(ax1, x, y, z)

# 绘制第二个3D图
ax2 = fig.add_subplot(132, projection='3d')
plt_3d_with_flam(ax2, x, y, z)

# 绘制第二个3D图
ax3 = fig.add_subplot(133, projection='3d')
plot_3d(ax3, x, y, z)

# 显示图像
plt.show()
print("*************************************************************************************", flush=True)
