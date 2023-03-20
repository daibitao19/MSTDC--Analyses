import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import random
from collections import Counter, deque
from copy import deepcopy
from math import ceil, sqrt, floor
import math


def get_descending_temp(dic_dic_dic):
    temp = pd.DataFrame(dic_dic_dic, index=['centrality'])
    temp = temp.T
    temp.sort_values(by=['centrality'], axis=0, ascending=False, inplace=True)
    temp
    return temp

#information_entropy
def get_information_entropy():
    value_dic = {}
    for node in max_com_subgraph_yuan.nodes():
        information_entropy = 0
        denominator = 0
        for nbr in max_com_subgraph_yuan[node]:
            denominator += max_com_subgraph_yuan.degree(nbr)
        for nbr in max_com_subgraph_yuan[node]:
            numerator = max_com_subgraph_yuan.degree(nbr)
            information_entropy += -(numerator / denominator) * math.log(numerator / denominator)
        value_dic[node] = information_entropy
    return (value_dic)

#Cycle_ratio
def Cycle_Ratio():
    import time
    Mygraph=max_com_subgraph_yuan.copy()
    # print(networkName)
    # print('Number of nodes = ', NodeNum)
    # print("Number of deges:", Mygraph.number_of_edges())

    DEF_IMPOSSLEN = NodeNum + 1 #Impossible simple cycle length

    SmallestCycles = set()
    NodeGirth = dict()
    NumSmallCycles = 0
    CycLenDict = dict()
    CycleRatio = {}

    SmallestCyclesOfNodes = {} #

    Coreness = nx.core_number(Mygraph)

    removeNodes =set()
    for i in Mygraph.nodes():  #
        SmallestCyclesOfNodes[i] = set()
        CycleRatio[i] = 0
        if Mygraph.degree(i) <= 1 or Coreness[i] <= 1:
            NodeGirth[i] = 0
            removeNodes.add(i)
        else:
            NodeGirth[i] = DEF_IMPOSSLEN



    Mygraph.remove_nodes_from(removeNodes)  #
    NumNode = Mygraph.number_of_nodes()  #update

    for i in range(3,Mygraph.number_of_nodes()+2):
        CycLenDict[i] = 0

    def my_all_shortest_paths(G, source, target):
        pred = nx.predecessor(G, source)
        if target not in pred:
            raise nx.NetworkXNoPath(
                f"Target {target} cannot be reached" f"from given sources"
            )
        sources = {source}
        seen = {target}
        stack = [[target, 0]]
        top = 0
        while top >= 0:
            node, i = stack[top]
            if node in sources:
                yield [p for p, n in reversed(stack[: top + 1])]
            if len(pred[node]) > i:
                stack[top][1] = i + 1
                next = pred[node][i]
                if next in seen:
                    continue
                else:
                    seen.add(next)
                top += 1
                if top == len(stack):
                    stack.append([next, 0])
                else:
                    stack[top][:] = [next, 0]
            else:
                seen.discard(node)
                top -= 1


    def getandJudgeSimpleCircle(objectList):#
        numEdge = 0
        for eleArr in list(itertools.combinations(objectList, 2)):
            if Mygraph.has_edge(eleArr[0], eleArr[1]):
                numEdge += 1
        if numEdge != len(objectList):
            return False
        else:
            return True


    def getSmallestCycles():
        NodeList = list(Mygraph.nodes())
        NodeList.sort()
        #setp 1
        curCyc = list()
        for ix in NodeList[:-2]:  #v1
            if NodeGirth[ix] == 0:
                continue
            curCyc.append(ix)
            for jx in NodeList[NodeList.index(ix) + 1 : -1]:  #v2
                if NodeGirth[jx] == 0:
                    continue
                curCyc.append(jx)
                if Mygraph.has_edge(ix,jx):
                    for kx in NodeList[NodeList.index(jx) + 1:]:      #v3
                        if NodeGirth[kx] == 0:
                            continue
                        if Mygraph.has_edge(kx,ix):
                            curCyc.append(kx)
                            if Mygraph.has_edge(kx,jx):
                                SmallestCycles.add(tuple(curCyc))
                                for i in curCyc:
                                    NodeGirth[i] = 3
                            curCyc.pop()
                curCyc.pop()
            curCyc.pop()
        # setp 2
        ResiNodeList = []  # Residual Node List
        for nod in NodeList:
            if NodeGirth[nod] == DEF_IMPOSSLEN:
                ResiNodeList.append(nod)
        if len(ResiNodeList) == 0:
            return
        else:
            visitedNodes = dict.fromkeys(ResiNodeList,set())
            for nod in ResiNodeList:
                if Coreness[nod] == 2 and NodeGirth[nod] < DEF_IMPOSSLEN:
                    continue
                for nei in list(Mygraph.neighbors(nod)):
                    if Coreness[nei] == 2 and NodeGirth[nei] < DEF_IMPOSSLEN:
                        continue
                    if not nei in visitedNodes.keys() or not nod in visitedNodes[nei]:
                        visitedNodes[nod].add(nei)
                        if nei not in visitedNodes.keys():
                            visitedNodes[nei] = set([nod])
                        else:
                            visitedNodes[nei].add(nod)
                        if Coreness[nei] == 2 and NodeGirth[nei] < DEF_IMPOSSLEN:
                            continue
                        Mygraph.remove_edge(nod, nei)
                        if nx.has_path(Mygraph, nod, nei):
                            for path in my_all_shortest_paths(Mygraph, nod, nei):
                                lenPath = len(path)
                                path.sort()
                                SmallestCycles.add(tuple(path))
                                for i in path:
                                    if NodeGirth[i] > lenPath:
                                        NodeGirth[i] = lenPath
                        Mygraph.add_edge(nod, nei)


    def StatisticsAndCalculateIndicators(): #
        global NumSmallCycles
        NumSmallCycles = len(SmallestCycles)
        for cyc in SmallestCycles:
            lenCyc = len(cyc)
            CycLenDict[lenCyc] += 1
            for nod in cyc:
                SmallestCyclesOfNodes[nod].add(cyc)
        for objNode,SmaCycs in SmallestCyclesOfNodes.items():
            if len(SmaCycs) == 0:
                continue
            cycleNeighbors = set()
            NeiOccurTimes = {}
            for cyc in SmaCycs:
                for n in cyc:
                    if n in NeiOccurTimes.keys():
                        NeiOccurTimes[n] += 1
                    else:
                        NeiOccurTimes[n] = 1
                cycleNeighbors = cycleNeighbors.union(cyc)
            cycleNeighbors.remove(objNode)
            del NeiOccurTimes[objNode]
            sum = 0
            for nei in cycleNeighbors:
                sum += float(NeiOccurTimes[nei]) / len(SmallestCyclesOfNodes[nei])
            CycleRatio[objNode] = sum + 1

    def printAndOutput_ResultAndDistribution(objectList,nameString,filename):
        addrespath =  filename+nameString + '.txt'
        Distribution = {}#

        for value in objectList.values():
            if value in Distribution.keys():
                Distribution[value] += 1
            else:
                Distribution[value] = 1

        for (myk, myv) in Distribution.items():
            Distribution[myk] = myv / float(NodeNum)

        rankedDict_ObjectList = sorted(objectList.items(), key=lambda d: d[1], reverse=True)
        fileout3 = open(addrespath, 'w')
        for d in range(len(rankedDict_ObjectList)):
            fileout3.writelines("%6d,%12.6f\n" % (float(rankedDict_ObjectList[d][0]),float(rankedDict_ObjectList[d][1])))
        fileout3.close()
        addrespath2 =  filename+'Distribution_' + nameString + '.txt'
        fileout2 = open(addrespath2, 'w')
        for (myk, myv) in Distribution.items():
            fileout2.writelines("%12.6f %12.6f  \n" % (myk, myv))
        fileout2.close()


    def printAndOutput_BasicCirclesDistribution(myCycLenDict,nameString,Outpath): #Copy_AllSimpleCircle
        Distribution = myCycLenDict
        global NumSmallCycles
        print('\nDistribution of SmallestBasicCycles:')
        float_allBasicCircles = float(NumSmallCycles)
        addrespath2 = Outpath + 'Distribution_' + nameString + '.txt'
        fileout2 = open(addrespath2, 'w')
        for (myk, myv) in Distribution.items():
            if float(myv) > 0:
                fileout2.writelines("%10d %15d  %12.6f  \n" % (float(myk), float(myv),float(myv/float_allBasicCircles)))
                print('len:%10d,count:%10d,ratio:%12.6f' % (myk, myv,myv/float_allBasicCircles))
        fileout2.close()

        List= list(SmallestCycles)
        rankedSBC_Set = sorted(List, key=lambda d: len(d), reverse=True)
        addrespath3 = Outpath + 'allSmallestBasicCycles.txt'
        fileout3 = open(addrespath3, 'w')
        for cy in rankedSBC_Set:
            fileout3.writelines("%s\n" %list(cy))
        fileout3.close()

    # main fun
    StartTime = time.time()
    getSmallestCycles()
    EndTime1 = time.time()

    StatisticsAndCalculateIndicators()


    #output
    printAndOutput_ResultAndDistribution(CycleRatio, 'CycleRatio',filename)
    # printAndOutput_BasicCirclesDistribution(CycLenDict, 'SmallestBasicCircles',OutputDistributionFile)


def get_cycle_ratio_nodes_sort():
    temp = pd.read_csv(filename+'CycleRatio.txt',header=None)
    temp.columns=['node','Cycle_Ratio']
    temp.sort_values(by=['Cycle_Ratio'], axis=0, ascending=False,inplace=True)
    return temp.index.to_list()

def get_cycle_ratio():
    temp = pd.read_csv(filename+'CycleRatio.txt',header=None)
    temp.columns=['node','Cycle_Ratio']
    return temp.set_index(['node'])['Cycle_Ratio'].to_dict()



#degree
def degree_centrality():
    value_dic = {}
    for node in max_com_subgraph_yuan.nodes():
        value_dic[node] = max_com_subgraph_yuan.degree[node]
    return (value_dic)

#betweenness
def betweenness_centrality():
    value_dic = {}
    betweenness_dic=nx.betweenness_centrality(max_com_subgraph_yuan)
    for node in max_com_subgraph_yuan.nodes():
        value_dic[node] = betweenness_dic[node]
    return (value_dic)

#closeness_centrality
def closeness_centrality():
    value_dic = {}
    closeness_dic=nx.closeness_centrality(max_com_subgraph_yuan)
    for node in max_com_subgraph_yuan.nodes():
        value_dic[node] = closeness_dic[node]
    return (value_dic)

#eigenvector_centrality
def eigenvector_centrality():
    value_dic = {}
    eigenvector_dic=nx.eigenvector_centrality(max_com_subgraph_yuan)
    for node in max_com_subgraph_yuan.nodes():
        value_dic[node] = eigenvector_dic[node]
    return (value_dic)

def k_core():
    value_dic = {}
    k_core_dic=nx.core_number(max_com_subgraph_yuan)
    for node in max_com_subgraph_yuan.nodes():
        value_dic[node] = k_core_dic[node]
    return (value_dic)

def pagerank():
    value_dic = {}
    pagerank_dic=nx.pagerank_numpy(max_com_subgraph_yuan)
    for node in max_com_subgraph_yuan.nodes():
        value_dic[node] = pagerank_dic[node]
    return (value_dic)


def BFS(seed, graph):  # graph图  s指的是开始结点
    edgelist = []
    # 需要一个队列
    # network_size=len(graph.keys())
    queue = deque([seed])
    seen = set([seed])  # 看是否访问过该结点
    while queue:
        vertex = queue.popleft()  # 保存第一结点，并弹出，方便把他下面的子节点接入
        nbr_list = list(graph[vertex])
        random.shuffle(nbr_list)  # 等概率，不然每次都是一样的顺序
        for nbr in nbr_list:  # 其实可以改成G[i]
            if nbr not in seen:  # 判断是否访问过，使用一个数组
                queue.append(nbr)
                seen.add(nbr)
                edgelist.append((vertex, nbr))
    return edgelist

# n = 30
def get_MSTDC(max_com_subgraph_node_list,max_com_subgraph_yuan, n):
    value_dic = {}
    for i in (max_com_subgraph_node_list):
        value_dic[i] = 0
    for time in range(1):#这里要思考一下，算传播是时候是取的100，瓦解的时候是1，但是做了100次
        node_list = random.sample(max_com_subgraph_node_list, n)
        for i in range(0, n):
            node = node_list[i]
            edgelist = BFS(node, max_com_subgraph_yuan)
            degree_dic = {}
            for i in (edgelist):
                degree_dic.setdefault(int(i[0]), []).append(int(i[1]))
                degree_dic.setdefault(int(i[1]), []).append(int(i[0]))
            for i in degree_dic:
                degree_dic[i] = len(set(degree_dic[i]))
            for i in degree_dic:
                value_dic[i] = value_dic[i] + degree_dic[i]
#     for i in degree_dic:
#         value_dic[i] = value_dic[i] / 100
    return (value_dic)

