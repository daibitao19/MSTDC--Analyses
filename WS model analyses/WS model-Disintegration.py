import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import random
from collections import Counter, deque
from copy import deepcopy
from math import ceil, sqrt, floor
import math
from copy import deepcopy
from multiprocessing import Process, Queue
import igraph as ig

def deleted_edge(node_list):
    edgelist = []
    for node in node_list:
        for nbr in max_com_subgraph_yuan[node]:
            edgelist.append((node, nbr))
    return edgelist


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
    NodeNum = Mygraph.number_of_nodes()  #
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


#clustering
def clustering_centrality():
    value_dic={}
    clustering_dic = nx.clustering(max_com_subgraph_yuan)
    for node in max_com_subgraph_yuan.nodes():
        value_dic[node] = clustering_dic[node]
    return (value_dic)
#degree
def degree_centrality():
    value_dic = {}
    for node in max_com_subgraph_yuan.nodes():
        value_dic[node] = max_com_subgraph_yuan.degree[node]
    return (value_dic)

# #igraph versiom
def betweenness_centrality(max_subgraph_igraph):
    value_dic = {}
    betweenness_centrality = max_subgraph_igraph.betweenness(directed=False)
    number = max_subgraph_igraph.vcount()
    number = (number - 1) * (number - 2) / 2
    for i, j in enumerate(betweenness_centrality):
        value_dic[max_subgraph_igraph.vs[i]['name']] = j / number
    return (value_dic)

def closeness_centrality(max_subgraph_igraph):
    value_dic = {}
    closeness_centrality = max_subgraph_igraph.closeness()
    for i, j in enumerate(closeness_centrality):
        value_dic[max_subgraph_igraph.vs[i]['name']] = j
    return (value_dic)

def eigenvector_centrality(max_subgraph_igraph):
    value_dic = {}
    eigenvector_centrality = max_subgraph_igraph.eigenvector_centrality(directed=False)
    for i,j in enumerate(eigenvector_centrality):
        value_dic[max_subgraph_igraph.vs[i]['name']]=j
    return (value_dic)

def k_core(max_subgraph_igraph):
    value_dic = {}
    kcore = max_subgraph_igraph.coreness(mode='all')
    for i,j in enumerate(kcore):
        value_dic[max_subgraph_igraph.vs[i]['name']]=j
    return (value_dic)

def pagerank(max_subgraph_igraph):
    value_dic = {}
    pagerank_dic = max_subgraph_igraph.pagerank(directed=False)
    for i, j in enumerate(pagerank_dic):
        value_dic[max_subgraph_igraph.vs[i]['name']] = j
    return (value_dic)

def get_ranking_excel(filename,centrality,value_dic):
    # value_dic=eval(method)
    temp = pd.DataFrame(value_dic,index=[centrality])
    temp = temp.T
    temp.sort_values(by=[centrality], axis=0, ascending=False,inplace=True)
    temp.to_excel(filename+'_'+centrality+'.xlsx')


def get_cycle_ratio_excel(filename):
    temp = pd.read_csv(filename+'CycleRatio.txt',header=None)
    temp.columns=['node','Cycle_Ratio']
    temp.sort_values(by=['Cycle_Ratio'], axis=0, ascending=False,inplace=True)
    temp.index=temp['node']
    del temp['node']
    temp.to_excel(filename+'_Cycle_Ratio'+'.xlsx')


def GraphLst(pair_lst, node_id,node_dct,edge_lst):
    print("----------------GraphLst START!-----------------")
    for tup in pair_lst:
        if tup[0] not in node_dct:
            node_dct[tup[0]] = node_id
            node_id += 1
        if tup[1] not in node_dct:
            node_dct[tup[1]] = node_id
            node_id += 1
        edge_lst.append((node_dct[tup[0]], node_dct[tup[1]]))
    return node_id


def GenerateGraph(node_dct, edge_lst):  # 利用iGraph构造网络
    print("----------------GenerateGraph START!-----------------")
    g = ig.Graph(directed=False)
    print("Adding vertices...")
    vertex_num = len(node_dct)
    g.add_vertices(vertex_num)  # 注意：从0开始！
    for key in node_dct:
        g.vs[node_dct[key]]['name'] = key

    print("Adding edges...")
    g.add_edges(edge_lst)
    #     g.es["weight"] = weight_lst
    print("Dumping GraphML file...")
    g.write_graphml('network.GraphML')
    return g

def BFS(seed, graph):
    edgelist = []
    queue = deque([seed])
    seen = set([seed])
    while queue:
        vertex = queue.popleft()
        nbr_list = list(graph[vertex])
        random.shuffle(nbr_list)
        for nbr in nbr_list:
            if nbr not in seen:
                queue.append(nbr)
                seen.add(nbr)
                edgelist.append((vertex, nbr))
    return edgelist



def get_MSTDC(max_com_subgraph_node_list,max_com_subgraph_yuan, n):
    value_dic = {}
    for i in (max_com_subgraph_node_list):
        value_dic[i] = 0
    for time in range(1):
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

def GCC_NC_connected_components_attack(G, nodes_sort):
    from copy import deepcopy
    from math import ceil, sqrt
    G_re = deepcopy(G)
    Network_size = len(G_re)
    N = ceil(Network_size  )
    nq = ceil(N * 0.01)
    largest_all = np.zeros(ceil(N / nq) + 1)
    number_of_components = np.zeros(ceil(N / nq) + 1)
    for i in range(ceil( N / nq) ):
        largest_cc = len(max(nx.connected_components(G_re), key=len))
        number_cc = len([len(c) for c in sorted(nx.connected_components(G_re), key=len)])
        remove_nodes = nodes_sort[i * nq:(i + 1) * nq]
        G_re.remove_nodes_from(remove_nodes)
        largest_all[i] = largest_cc / Network_size
        number_of_components[i] = number_cc
    return largest_all, number_of_components, np.mean(largest_all)  # R是gcc曲线的面积，因为gcc除以N，所以相对百分比求平均即可

def GCC_NC_connected_components_attack_for_mstdc(G, nodes_sort,gcc_size_all_networks_mstdc_dic, component_all_networks_mstdc_dic,time,q):
    from copy import deepcopy
    from math import ceil, sqrt
    G_re = deepcopy(G)
    Network_size = len(G_re)
    N = ceil(Network_size )
    nq = ceil(N * 0.01)
    largest_all = np.zeros(ceil(N / nq) + 1)
    number_of_components = np.zeros(ceil(N / nq) + 1)
    # largest_all[0] = len(max(nx.connected_components(G_re), key=len))/Network_size
    for i in range(ceil(N / nq) ):
        largest_cc = len(max(nx.connected_components(G_re), key=len))
        number_cc = len([len(c) for c in sorted(nx.connected_components(G_re), key=len)])
        remove_nodes = nodes_sort[i * nq:(i + 1) * nq]
        G_re.remove_nodes_from(remove_nodes)  # 这里存在延迟，第一个区间的值放到第二个区间
        largest_all[i] = largest_cc / Network_size
        number_of_components[i] = number_cc
    gcc_size_all_networks_mstdc_dic[time] = largest_all
    component_all_networks_mstdc_dic[time] = number_of_components
    q.put([gcc_size_all_networks_mstdc_dic, component_all_networks_mstdc_dic, np.mean(largest_all)])


def Centrality_qc_attack(max_com_subgraph_yuan, centrality_function,gcc_community_dic_one,component_community_dic_one,r_community_dic_one):
    i, j = centrality_function[0], centrality_function[1]
    value_dic = eval(j)  # 运行函数，将字符串变成函数
    temp = get_descending_temp(value_dic)
    nodes_sort = temp.index.tolist()
    gcc_size, components_number, r = GCC_NC_connected_components_attack(max_com_subgraph_yuan, nodes_sort)
    gcc_community_dic_one[i] = gcc_size
    component_community_dic_one[i] = components_number
    r_community_dic_one[i] = r


def GCC_NC_connected_components_attack_explore_for_n(G, nodes_sort):
    from copy import deepcopy
    from math import ceil, sqrt
    G_re = deepcopy(G)
    Network_size = len(G_re)
    N = ceil(Network_size*0.1)
    remove_nodes = nodes_sort[0:N]
    G_re.remove_nodes_from(remove_nodes)
    largest_cc = len(max(nx.connected_components(G_re), key=len))/ Network_size
    number_cc = len([len(c) for c in sorted(nx.connected_components(G_re), key=len)])
    return largest_cc,number_cc


def mstdc_attack_one_time_explore_n(max_com_subgraph_node_list,gcc_size_all_networks_mstdc_dic,component_all_networks_mstdc_dic,max_com_subgraph_yuan,n,q):
    components_list=[]
    gcc_size_list=[]
    for time in trange(100):
        value_dic = get_MSTDC(max_com_subgraph_node_list,max_com_subgraph_yuan,n)
        temp = get_descending_temp(value_dic)
        nodes_sort = temp.index.tolist()
        gcc_size, components_number= GCC_NC_connected_components_attack_explore_for_n(max_com_subgraph_yuan, nodes_sort)
        gcc_size_list.append(gcc_size)
        components_list.append(components_number)

    gcc_size_all_networks_mstdc_dic[n] = gcc_size_list
    component_all_networks_mstdc_dic[n] = components_list
    q.put([gcc_size_all_networks_mstdc_dic,component_all_networks_mstdc_dic])

def benchmark_attack_one_time(n,m,p,q,time):
    gcc_size_all_networks_dic = {}
    r_all_networks_dic = {}
    component_all_networks_dic = {}
    global max_com_subgraph_yuan, max_com_subgraph_node_list, filename
    G = nx.connected_watts_strogatz_graph(n, m,p)
    filename = 'WS_' + str(n) + '_' + str(m) + '_'+str(p)+'_'+str(time)+'_Network'
    print(filename)
    nx.write_gml(G, filename + '.gml')
    largest_cc = max(nx.connected_components(G), key=len)
    max_com_subgraph_yuan = G.subgraph(largest_cc).copy()
    network_size = max_com_subgraph_yuan.number_of_nodes()
    max_com_subgraph_node_list = list(max_com_subgraph_yuan.nodes())
    max_com_subgraph = max_com_subgraph_yuan.copy()


    source = [int(edge[0]) for edge in max_com_subgraph_yuan.edges()]
    target = [int(edge[1]) for edge in max_com_subgraph_yuan.edges()]

    pair_lst = []
    for i in range(len(source)):
        pair_lst.append((source[i], target[i]))
    node_dct = {}
    edge_lst = []
    node_id = 0
    node_id = GraphLst(pair_lst, node_id,node_dct,edge_lst)
    IG = GenerateGraph(node_dct, edge_lst)  # 无向加权网，网络要转成无向网
    largest_cc = max(IG.components(), key=len)
    max_subgraph_igraph = IG.subgraph(largest_cc).copy()  # 导出最大连通子图

    centrality_functions = {
                            'Information_Entropy': 'get_information_entropy()',
                            'Cycle_Ratio': 'get_cycle_ratio()'
        , 'Degree': 'degree_centrality()', 'Betweenness': 'betweenness_centrality(max_subgraph_igraph)',
                            'Eigenvector': 'eigenvector_centrality(max_subgraph_igraph)',
                            'Closeness': 'closeness_centrality(max_subgraph_igraph)',
                         'K-core': 'k_core(max_subgraph_igraph)',
                            'Clustering': 'clustering_centrality()',
                            'Pagerank': 'pagerank(max_subgraph_igraph)'}

    for time_ in range(1):
        Cycle_Ratio()
        # get_cycle_ratio_excel(filename)
        for i, j in centrality_functions.items():
            value_dic = eval(j)  # ���ַ�����ɺ���
            # get_ranking_excel(filename, i, value_dic)
            temp = get_descending_temp(value_dic)
            nodes_sort = temp.index.tolist()
            gcc_size, components_number, r = GCC_NC_connected_components_attack(max_com_subgraph_yuan,
                                                                                nodes_sort)
            gcc_size_all_networks_dic[i] = gcc_size
            component_all_networks_dic[i] = components_number
            r_all_networks_dic[i] = r

    # mstdc
    # print(file_save_name)
    gcc_size_all_networks_mstdc_dic = {}
    component_all_networks_mstdc_dic = {}
    q_ = Queue()
    jobs_ = []
    for time__ in trange(10):
        # print(time)
        value_dic = get_MSTDC(max_com_subgraph_node_list, max_com_subgraph_yuan, 30)
        temp = get_descending_temp(value_dic)
        nodes_sort = temp.index.tolist()

        process__ = Process(target=GCC_NC_connected_components_attack_for_mstdc,
                           args=(max_com_subgraph_yuan, nodes_sort,
                                 gcc_size_all_networks_mstdc_dic, component_all_networks_mstdc_dic, time__,
                                 q_))
        jobs_.append(process__)
        process__.start()
    results_ = [q_.get() for j in jobs_]  # [[qc,r,gcc],[qc,r,gcc]]
    for process__ in jobs_:
        process__.join()

    gcc_size_all_networks_mstdc_dic_list = [i[0] for i in results_]
    component_all_networks_mstdc_dic_list = [i[1] for i in results_]
    r_all_networks_mstdc_dic_list = [i[2] for i in results_]
    print('R_mstdc:mean:{},var:{},std:{}'.format(np.mean(r_all_networks_mstdc_dic_list),
                                                 np.var(r_all_networks_mstdc_dic_list),
                                                 np.std(r_all_networks_mstdc_dic_list)))

    gcc_size_mstdc_dic = gcc_size_all_networks_mstdc_dic_list[0]
    # print(qc_all_networks_dic)
    for i in gcc_size_all_networks_mstdc_dic_list:
        gcc_size_mstdc_dic.update(i)
    gcc_size_mstdc_df = pd.DataFrame(gcc_size_mstdc_dic)

    component_mstdc_dic = component_all_networks_mstdc_dic_list[0]
    # print(qc_all_networks_dic)
    for i in component_all_networks_mstdc_dic_list:
        component_mstdc_dic.update(i)
    component_mstdc_df = pd.DataFrame(component_mstdc_dic)



    gcc_size_mstdc_df['Col_mean'] = gcc_size_mstdc_df.apply(lambda x: x.mean(), axis=1)
    gcc_size_mstdc_df['MIN'] = gcc_size_mstdc_df.apply(lambda x: x.min(), axis=1)
    component_mstdc_df['Col_mean'] = component_mstdc_df.apply(lambda x: x.mean(), axis=1)
    component_mstdc_df['MAX'] = component_mstdc_df.apply(lambda x: x.max(), axis=1)



    gcc_size_all_networks_dic['MSTDC'] = list(gcc_size_mstdc_df['Col_mean'])
    gcc_size_all_networks_dic['MIN'] = list(gcc_size_mstdc_df['MIN'])
    component_all_networks_dic['MSTDC'] = list(component_mstdc_df['Col_mean'])
    component_all_networks_dic['MAX'] = list(component_mstdc_df['MAX'])
    r_all_networks_dic['MSTDC'] = np.mean(r_all_networks_mstdc_dic_list)
    r_all_networks_dic['MIN'] = np.min(r_all_networks_mstdc_dic_list)
    q.put([gcc_size_all_networks_dic, component_all_networks_dic,r_all_networks_dic])


if __name__ == '__main__':

    N = [500,1000]
    M = [4, 6, 8, 10]
    P = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    times=100
    for n in N:
        for m in M:
            for p in P:
                file_save_name = 'WS_' + str(n) + '_' + str(m) + '_' + str(p) + '_Network_' + str(times)
                q = Queue()
                jobs = []
                for time in trange(times):
                    # print(time)
                    process_ = Process(target=benchmark_attack_one_time,
                                       args=(n, m, p, q, time))
                    jobs.append(process_)
                    process_.start()
                results = [q.get() for j in jobs]  # [[qc,r,gcc],[qc,r,gcc]]
                for process_ in jobs:
                    process_.join()

                print('start')
                gcc_size_all_networks_dic_list = [i[0] for i in results]
                gcc_size_all_networks_dic = gcc_size_all_networks_dic_list[0]
                gcc_size_all_networks_dic_df = pd.DataFrame(gcc_size_all_networks_dic)
                # print(qc_all_networks_dic)
                for i in gcc_size_all_networks_dic_list[1:]:
                    gcc_df=pd.DataFrame(i)
                    gcc_size_all_networks_dic_df += + gcc_df
                gcc_size_all_networks_dic_df = gcc_size_all_networks_dic_df / times
                print('start')
                component_all_networks_dic_list = [i[1] for i in results]
                component_all_networks_dic = component_all_networks_dic_list[0]
                component_all_networks_dic_df = pd.DataFrame(component_all_networks_dic)
                for i in component_all_networks_dic_list[1:]:
                    component_df = pd.DataFrame(i)
                    component_all_networks_dic_df += + component_df
                component_all_networks_dic_df = component_all_networks_dic_df / times

                r_all_networks_dic_list = [i[2] for i in results]
                r_all_networks_dic={}
                for i in range(len(r_all_networks_dic_list)):
                    r_all_networks_dic[i] = r_all_networks_dic_list[i]
                    
                r_all_networks_dic_df=pd.DataFrame(r_all_networks_dic)
                r_all_networks_dic_df['Mean'] = r_all_networks_dic_df.apply(lambda x: x.mean(), axis=1)
                r_all_networks_dic_df['Min'] = r_all_networks_dic_df.apply(lambda x: x.min(), axis=1)
                r_all_networks_dic_df=r_all_networks_dic_df.T
                r_all_networks_dic_df.to_excel(file_save_name + '_R.xlsx')

                gcc_size_all_networks_dic_df.to_excel(file_save_name + '_all_gcc_size.xlsx')
                component_all_networks_dic_df.to_excel(file_save_name + '_all_component.xlsx')




