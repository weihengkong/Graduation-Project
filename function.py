import copy

import numpy as np

# from init import containers
# from init import trans_delays, exc_delays, ranks
# from myDAXReader import tasks as containers
# from myDAXReader import trans_delays, exc_delays, ranks

from utils.initUtils import tasks as containers
from utils.initUtils import trans_delays, exc_delays, ranks
from init import nodes

nodes_ = copy.deepcopy(nodes)


def func_rank(rank_id, individual):
    nodes = individual["nodes"]
    exc_delay = []
    trans_delay = []
    for container_id in ranks["rank" + str(rank_id)]:
        for node in nodes:
            if container_id in node["place"]:
                node_id = node["id"]
                exc_delay.append(exc_delays[container_id][node_id])
        trans_delays_np = np.array(trans_delays)
        for i in range(len(trans_delays_np[:, container_id])):
            if trans_delays_np[:, container_id][i] != 0:
                other_container_id = i
                for node in nodes:
                    if container_id in node["place"] and other_container_id not in node["place"]:
                        trans_delay.append(trans_delays_np[:, container_id][i] / (
                                (containers[container_id]['net'] + containers[other_container_id]['net']) / 2))
                    else:
                        trans_delay.append(0)
            else:
                trans_delay.append(0)
    if rank_id == 0:
        return max(exc_delay) + max(trans_delay)
    else:
        return max(exc_delay) + max(trans_delay) + func_rank(rank_id - 1, individual)


# 时延
def function1(individual):
    return func_rank(len(ranks) - 1, individual)


# # 负载
# def function2(individual):
#     gpu_util_all = np.array(
#         [sum([containers[container_id]["gpu"] for container_id in node["place"]]) / nodes_[i]["gpu"] for i, node in
#          enumerate(individual["nodes"])])
#     gpu_balance = np.std(gpu_util_all)
#
#     cpu_util_all = np.array(
#         [sum([containers[container_id]["cpu"] for container_id in node["place"]]) / nodes_[i]["cpu"] for i, node in
#          enumerate(individual["nodes"])])
#     cpu_balance = np.std(cpu_util_all)
#
#     mem_util_all = np.array(
#         [sum([containers[container_id]["mem"] for container_id in node["place"]]) / nodes_[i]["mem"] for i, node in
#          enumerate(individual["nodes"])])
#     mem_balance = np.std(mem_util_all)
#
#     net_util_all = np.array(
#         [sum([containers[container_id]["net"] for container_id in node["place"]]) / nodes_[i]["net"] for i, node in
#          enumerate(individual["nodes"])])
#     net_balance = np.std(net_util_all)
#
#     balance = (gpu_balance + cpu_balance + mem_balance + net_balance) / 4
#     return balance

# 价格
def function2(individual):
    price = 0
    nodes = individual["nodes"]
    for node in nodes:
        t = 0
        node_id = node['id']
        t_ranks = {}
        for container_id in node['place']:
            container = containers[container_id]
            container_rank = container['rank']
            t_exc = exc_delays[container_id][node_id]
            t_container = t_exc
            if container_rank not in t_ranks.keys() or t_ranks[container_rank] < t_container:
                t_ranks[container_rank] = t_container
        # print(node['place'])
        # print(t_ranks)
        # print("--------------------------------")
        for value in t_ranks.values():
            t = t + value
        price = price + node['price'] * t
    return price / 3600

# 能耗
def fun3(individual):
    ans_power = 0
    nodes = individual["nodes"]
    for node in nodes:
        node_id = node['id']
        for container_id in node['place']:
            container = containers[container_id]
            t_exc = exc_delays[container_id][node_id]
            total = node['cpu']
            container_cpu = container['cpu']
            p_total = node['power']
            p_null = p_total * 0.7
            p_use = container_cpu / total * p_total + p_null
            e = p_use * t_exc
            ans_power += e
    return ans_power
