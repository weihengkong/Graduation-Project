import xml.sax

import numpy as np

from utils import fileUtils


class Task:
    def __init__(self, task_id, runtime):
        self.task_id = task_id
        self.runtime = runtime
        self.in_edges = []
        self.out_edges = []

    def insertInEdge(self, e):
        self.in_edges.append(e)

    def insertOutEdge(self, e):
        self.out_edges.append(e)


class Edge:
    def __init__(self, parent, child):
        self.parent = parent
        self.child = child


class TransferData:
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.source = None
        self.destinations = []

    def getSize(self):
        return self.size

    def getSource(self):
        return self.source

    def setSource(self, source):
        self.source = source

    def addDestination(self, t):
        self.destinations.append(t)

    def getDestinations(self):
        return self.destinations


class MyDAXReader(xml.sax.ContentHandler):
    def __init__(self):
        self.tags = []
        self.childId = None
        self.lastTask = None
        self.nameTaskMapping = {}
        self.transferData = {}

    def startElement(self, name, attrs):
        if name == "job":
            task_id = attrs.getValue("id")
            if task_id in self.nameTaskMapping:
                raise RuntimeError("id conflicts")
            runtime = float(attrs.getValue("runtime"))
            task = Task(task_id, runtime)
            self.nameTaskMapping[task_id] = task
            self.lastTask = task
        elif name == "uses" and self.tags[-1] == "job":
            filename = attrs.getValue("file")
            size = int(attrs.getValue("size"))
            td = self.transferData.get(filename)
            if td is None:
                td = TransferData(filename, size)
            if attrs.getValue("link") == "input":
                td.addDestination(self.lastTask)
            else:
                td.setSource(self.lastTask)
            self.transferData[filename] = td
        elif name == "child":
            self.childId = attrs.getValue("ref")
        elif name == "parent":
            child = self.nameTaskMapping[self.childId]
            parent = self.nameTaskMapping[attrs.getValue("ref")]
            e = Edge(parent, child)
            parent.insertOutEdge(e)
            child.insertInEdge(e)
        self.tags.append(name)

    def endElement(self, name):
        self.tags.pop()


class Data:
    def __init__(self):
        self.nodes = None
        self.ranks = None
        self.trans_delays = None
        self.exc_delays = None
        self.tasks = None
        self.handler = None
        self.nodeLen = 6

    def _initArray(self):

        # 生成一个len(tasks)*len(tasks)的矩阵,每个元素的意义是两个task之间是否有边，1表示有边，0表示没有边
        edges = np.zeros((len(self.tasks), len(self.tasks)))
        for task in self.tasks:
            #     如果task的后继不为空，则将task的id和后继的id对应的矩阵元素置为随机数
            if len(task['after']) != 0:
                for after in task['after']:
                    edges[task['id']][after] = np.random.randint(10, 20)
        trans_delays = edges
        exc_delays = np.zeros((len(self.tasks), len(self.nodes)))
        # 根据每个node的price来确定每个task在每个node上的执行时间
        for i in range(len(self.tasks)):
            for j in range(len(self.nodes)):
                # 根据task的需求与node的资源比例来确定task在node上的执行时间,TODO:确定一个更合理的公式
                # exc_delays[i][j] = (self.tasks[i]['cpu'] / self.nodes[j]['cpu']) * (
                #             self.tasks[i]['gpu'] / self.nodes[j]['gpu']) * (self.tasks[i]['mem'] / self.nodes[j]['mem']) * (
                #                                self.tasks[i]['net'] / self.nodes[j]['net']) * 1000000
                exc_delays[i][j] = ((self.tasks[i]['cpu'] / self.nodes[j]['cpu']) + (
                            self.tasks[i]['gpu'] / self.nodes[j]['gpu']) + (self.tasks[i]['mem'] / self.nodes[j]['mem']) + (
                                               self.tasks[i]['net'] / self.nodes[j]['net'])) * self.nodes[j]['price']

        return trans_delays, exc_delays

    def _loadFile(self, filename):
        exc_delays = fileUtils.load_from_file("./" + filename + ".exc_delays.txt")
        trans_delays = fileUtils.load_from_file("./" + filename + ".trans_delays.txt")
        return exc_delays, trans_delays

    def readDAX(self, filename):
        parser = xml.sax.make_parser()
        self.handler = MyDAXReader()
        parser.setContentHandler(self.handler)
        parser.parse(filename)
        self.tasks = self._saveTasks()
        # 将tasks保存为数组形式
        self.tasks = list(self.tasks.values())
        self.ranks = {}
        for container in self.tasks:
            r = self._rank(container)
            container['rank'] = r
            if 'rank' + str(r) in self.ranks.keys():
                self.ranks['rank' + str(r)].append(container['id'])
            else:
                self.ranks['rank' + str(r)] = [container['id']]

        # 如果文件不存在，则保存
        # if not fileUtils.isExist("./" + filename + ".tasks.txt"):
        fileUtils.save_list_to_file(self.tasks, "./" + filename + ".tasks.json")
        # if not fileUtils.isExist("./" + filename + ".ranks.txt"):
        fileUtils.save_dict_to_file(self.ranks, "./" + filename + ".ranks.json")
        # 保存随机生成的node信息
        self.nodes = self._saveInitNodes(filename)
        # 根据price来确定资源的执行时长，TODO：这里有点本末倒置，以后修改
        # 保存初始化的array信息,如果文件不存在，则生成一个新的array
        # if not fileUtils.isExist("./" + filename + ".exc_delays.txt"):
        self._saveArray(filename)
        self.exc_delays, self.trans_delays = self._loadFile(filename)

    # 定义一个函数，计算最大的rank的cpu总需求量
    def _getResourceByRank(self):
        # TODO :目前设置的是每种资源的需求量最大，后续可能考虑统一计算
        cpuMax = 0
        gpuMax = 0
        memMax = 0
        netMax = 0

        #     遍历所有的rank
        for rank in self.ranks.keys():
            cpu = 0
            gpu = 0
            mem = 0
            net = 0
            #     遍历每个rank中的所有task
            for task_id in self.ranks[rank]:
                cpu += self.tasks[task_id]['cpu']
                gpu += self.tasks[task_id]['gpu']
                mem += self.tasks[task_id]['mem']
                net += self.tasks[task_id]['net']
            #     获取task的cpu较大值
            cpuMax = max(cpu, cpuMax)
            gpuMax = max(gpu, gpuMax)
            memMax = max(mem, memMax)
            netMax = max(net, netMax)
        return cpuMax, gpuMax, memMax, netMax

    def _saveInitNodes(self, filename):
        ans = []
        cpuMax, gpuMax, memMax, netMax = self._getResourceByRank()
        # 避免在初始化的时候，资源不够，将资源最大值增加50%
        cpuMax = cpuMax * 1.5
        gpuMax = gpuMax * 1.5
        memMax = memMax * 1.5
        netMax = netMax * 1.5

        # 根据这几个值，生成初始的nodeLen个node，将资源随机分配到node上，要求资源的和等于资源最大值，保证资源量在最大值的1/8到3/8之间
        for i in range(self.nodeLen):
            len = self.nodeLen / 2
            node = {}
            node['id'] = i
            node['cpu'] = np.random.randint(cpuMax / len, cpuMax * 3 / len)
            node['gpu'] = np.random.randint(gpuMax / len, gpuMax * 3 / len)
            node['mem'] = np.random.randint(memMax / len, memMax * 3 / len)
            node['net'] = np.random.randint(netMax / len, netMax * 3 / len)

            ans.append(node)

        # 每个node添加price和power属性
        for node in ans:
            # price等于cpu+gpu+mem+net的价格和，价格分别为10,20,5,3
            node['price'] = float(node['cpu'] * 10 + node['gpu'] * 20 + node['mem'] * 5 + node['net'] * 3) / 100
            # power等于cpu+gpu+mem+net的功耗和，功耗分别为20,40,1,1
            node['power'] = float(node['cpu'] * 20 + node['gpu'] * 40 + node['mem'] * 1 + node['net'] * 1)/100
            node['place'] = []
        fileUtils.save_list_to_file(ans, "./" + filename + ".initNodes.json")

        return ans

    # 实现一个函数，将handler.nameTaskMapping 保存到init.py的tasks中，
    # 形式为container1 = {'id': 0, 'cpu': 1, 'mem': 4, 'gpu': 2, 'net': 10, 'pre': [], 'front': [], 'after': [4, 5]}
    def _saveTasks(self):
        # 生成tasks
        tasks = {}
        #     解析handler.nameTaskMapping
        for task in self.handler.nameTaskMapping.values():
            task_id = task.task_id
            runtime = task.runtime
            pre = []
            front = []
            after = []
            # 存放所有的前缀节点
            edgesTmp = []
            # 将edge.in_edges中的所有edge加入到edgesTmp中
            for edge in task.in_edges:
                edgesTmp.append(edge)
            # 将edgesTmp中的所有edge的前缀节点加入到pre中
            while len(edgesTmp) != 0:
                edge = edgesTmp.pop()
                # 如果edge的前缀节点不在pre中，则将其加入到pre中
                if edge.parent.task_id not in pre:
                    pre.append(edge.parent.task_id)
                for edge2 in edge.parent.in_edges:
                    edgesTmp.append(edge2)

            # 此处的pre存放的是task的所有前驱节点的id,将前缀节点的前缀节点加入到pre中，如此循环，直到没有前缀节点
            for edge in task.in_edges:
                front.append(edge.parent.task_id)

            # 此处的after存放的是task的直接后继的id
            for edge in task.out_edges:
                after.append(edge.child.task_id)
            # 将id信息使用int类型存储，前面的ID字段截取掉
            task_id = int(task_id[2:])
            # 将pre、front、after中的id信息使用int类型存储，前面的ID字段截取掉
            for i in range(len(pre)):
                pre[i] = int(pre[i][2:])
            for i in range(len(front)):
                front[i] = int(front[i][2:])
            for i in range(len(after)):
                after[i] = int(after[i][2:])
            # 此时与runtime没有关系了
            # CPU范围在1-10，GPU范围在0-8，MEM范围在10-100，NET范围在10-100
            cpu = np.random.randint(1, 10)
            gpu = np.random.randint(0, 8)
            mem = np.random.randint(10, 100)
            net = np.random.randint(10, 100)

            # 将task的信息存储到tasks中
            tasks[task_id] = {'id': task_id, 'cpu': cpu, 'mem': mem, 'gpu': gpu, 'net': net, 'pre': pre, 'front': front,
                              'after': after}

        return tasks

    def _rank(self, container):
        if len(container['pre']) == 0:
            return 0
        else:
            pre_rank = []
            for pre_container_id in container['pre']:
                pre_rank.append(self._rank(self.tasks[pre_container_id]))
            return max(pre_rank) + 1

    def _saveArray(self, filename):
        self.trans_delays, self.exc_delays = self._initArray()
        fileUtils.save_to_file(self.trans_delays, "./" + filename + ".trans_delays.txt")
        fileUtils.save_to_file(self.exc_delays, "./" + filename + ".exc_delays.txt")

    def readDAXUtils(self, filename):
        data = Data()
        data.readDAX(filename)
        return data.tasks, data.ranks, data.exc_delays, data.trans_delays

# 50
filename = "workflow/GENOME.n.50.0.dax"
# filename = "workflow/LIGO.n.50.0.dax"
# filename = "workflow/CYBERSHAKE.n.50.0.dax"
# filename = "workflow/MONTAGE.n.50.0.dax"
# 100
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
data = Data()
data.readDAX(filename)
tasks, ranks, exc_delays, trans_delays = data.tasks, data.ranks, data.exc_delays, data.trans_delays
print(tasks)
