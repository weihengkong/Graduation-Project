"""
获取初始资源
"""

from utils import fileUtils


def getInitResource(filename):
    tasks = fileUtils.load_list_from_file(filename + ".tasks.json")
    ranks = fileUtils.load_dict_from_file(filename + ".ranks.json")
    exc_delays = fileUtils.load_from_file(filename + ".exc_delays.txt")
    trans_delays = fileUtils.load_from_file(filename + ".trans_delays.txt")
    return tasks, ranks, exc_delays, trans_delays


def getInitNodes(filename):
    nodes = fileUtils.load_list_from_file(filename + ".initNodes.json")
    return nodes

# main调用目录
# filename="workflow/LIGO.n.50.0.dax"
# filename = "workflow/CYBERSHAKE.n.50.0.dax"
filename = "workflow/GENOME.n.50.0.dax"
# filename = "workflow/MONTAGE.n.50.0.dax"

# filename = "workflow/MONTAGE.n.100.0.dax"
# filename = "workflow/GENOME.n.100.0.dax"
# filename = "workflow/LIGO.n.100.0.dax"
# filename = "workflow/CYBERSHAKE.n.100.0.dax"
# 200
# filename = "workflow/GENOME.n.200.0.dax"
# filename = "workflow/LIGO.n.200.0.dax"
# filename = "workflow/CYBERSHAKE.n.200.0.dax"
# filename = "workflow/MONTAGE.n.200.0.dax"
# 500
# filename = "workflow/GENOME.n.500.0.dax"
# filename = "workflow/LIGO.n.500.0.dax"
# filename = "workflow/CYBERSHAKE.n.500.0.dax"
# filename = "workflow/MONTAGE.n.500.0.dax"
tasks, ranks, exc_delays, trans_delays = getInitResource(filename)
nodes = getInitNodes(filename)

# 本文件调用目录
# filename="../workflow/LIGO.n.50.0.dax"
# tasks, ranks, exc_delays, trans_delays = getInitResource(filename)
# nodes = getInitNodes(filename)
# print(nodes)