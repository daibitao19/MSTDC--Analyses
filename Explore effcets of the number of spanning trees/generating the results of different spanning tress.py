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
    return largest_all, number_of_components, np.mean(largest_all)

def GCC_NC_connected_components_attack_for_mstdc(G,nodes_sort,gcc_size_all_networks_mstdc_dic, component_all_networks_mstdc_dic,r_all_networks_mstdc_dic,time,q):
    from copy import deepcopy
    from math import ceil, sqrt

    G_re = deepcopy(G)
    Network_size = len(G_re)
    N = ceil(Network_size )
    nq = ceil(N * 0.01)
    largest_all = np.zeros(ceil(N / nq) + 1)
    number_of_components = np.zeros(ceil(N / nq) + 1)
    for i in range(ceil(N / nq) ):
        largest_cc = len(max(nx.connected_components(G_re), key=len))
        number_cc = len([len(c) for c in sorted(nx.connected_components(G_re), key=len)])
        remove_nodes = nodes_sort[i * nq:(i + 1) * nq]
        G_re.remove_nodes_from(remove_nodes)
        largest_all[i] = largest_cc / Network_size
        number_of_components[i] = number_cc
    gcc_size_all_networks_mstdc_dic[time] = largest_all
    component_all_networks_mstdc_dic[time] = number_of_components
    q.put([gcc_size_all_networks_mstdc_dic, component_all_networks_mstdc_dic, np.mean(largest_all)])


def Centrality_qc_attack(max_com_subgraph_yuan, centrality_function,gcc_community_dic_one,component_community_dic_one,r_community_dic_one):
    i, j = centrality_function[0], centrality_function[1]
    value_dic = eval(j)
    temp = get_descending_temp(value_dic)
    nodes_sort = temp.index.tolist()
    gcc_size, components_number, r = GCC_NC_connected_components_attack(max_com_subgraph_yuan, nodes_sort)
    gcc_community_dic_one[i] = gcc_size
    component_community_dic_one[i] = components_number
    r_community_dic_one[i] = r



def mstdc_attack_one_time_explore_n(filename,max_com_subgraph_node_list,max_com_subgraph_yuan,gcc_size_all_networks_dic,
                                    component_all_networks_dic,r_all_networks_dic,q, n):

    gcc_size_all_networks_mstdc_dic ={}
    component_all_networks_mstdc_dic ={}
    r_all_networks_mstdc_dic ={}

    gcc_size_all_networks_dic_one = {}
    component_all_networks_dic_one= {}
    r_all_networks_dic_one = {}

    q_ = Queue()
    jobs_ = []
    for time_ in trange(50):
        value_dic = get_MSTDC(max_com_subgraph_node_list, max_com_subgraph_yuan, n)
        temp = get_descending_temp(value_dic)
        nodes_sort = temp.index.tolist()
        process_ = Process(target=GCC_NC_connected_components_attack_for_mstdc,
                           args=(max_com_subgraph_yuan,nodes_sort,gcc_size_all_networks_mstdc_dic, r_all_networks_mstdc_dic,
                                 component_all_networks_mstdc_dic,time_,q_))
        jobs_.append(process_)
        process_.start()
    results = [q_.get() for j in jobs_]  # [[qc,r,gcc],[qc,r,gcc]]
    for process_ in jobs_:
        process_.join()

    gcc_size_all_networks_mstdc_dic_list = [i[0] for i in results]
    component_all_networks_mstdc_dic_list = [i[1] for i in results]
    r_all_networks_mstdc_dic_list = [i[2] for i in results]

    gcc_size_mstdc_dic = gcc_size_all_networks_mstdc_dic_list[0]
    # print(qc_all_networks_dic)
    for i in gcc_size_all_networks_mstdc_dic_list:
        gcc_size_mstdc_dic.update(i)
    gcc_size_mstdc_df = pd.DataFrame(gcc_size_mstdc_dic)

    component_mstdc_dic = component_all_networks_mstdc_dic_list[0]
    for i in component_all_networks_mstdc_dic_list:
        component_mstdc_dic.update(i)
    component_mstdc_df = pd.DataFrame(component_mstdc_dic)

    gcc_size_mstdc_df['Col_mean'] = gcc_size_mstdc_df.apply(lambda x: x.mean(), axis=1)
    gcc_size_mstdc_df['Min'] = gcc_size_mstdc_df.apply(lambda x: x.min(), axis=1)
    gcc_size_mstdc_df.loc['R'] = gcc_size_mstdc_df.apply(lambda x: x.mean(), axis=0)

    component_mstdc_df['Col_mean'] = component_mstdc_df.apply(lambda x: x.mean(), axis=1)
    component_mstdc_df['Min'] = component_mstdc_df.apply(lambda x: x.min(), axis=1)
    component_mstdc_df.loc['R'] = gcc_size_mstdc_df.apply(lambda x: x.mean(), axis=0)

    gcc_size_mstdc_df.to_excel( filename+'_'+str(n)+'_mstdc_gcc_size.xlsx')
    component_mstdc_df.to_excel( filename+'_'+str(n)+'_mstdc_component.xlsx')
    gcc_size_all_networks_dic[n] = list(gcc_size_mstdc_df['Col_mean'])
    component_all_networks_dic[n] = list(component_mstdc_df['Col_mean'])
    r_all_networks_dic[n] = np.mean(r_all_networks_mstdc_dic_list)


    gcc_size_all_networks_dic_one[n] = list(gcc_size_mstdc_df['Col_mean'])
    component_all_networks_dic_one[n] = list(component_mstdc_df['Col_mean'])
    r_all_networks_dic_one[n] = np.mean(r_all_networks_mstdc_dic_list)
    q.put([gcc_size_all_networks_dic_one,component_all_networks_dic_one,r_all_networks_dic_one])



if __name__ == '__main__':

    G = nx.Graph()
    filename = 'dolphins.txt'
    source = []
    target = []
    with open(filename)as file_object:

        for line in (file_object.readlines()):
            node1, node2 = line.strip('\n').split()
            G.add_edge(int(node1), int(node2))
    G.remove_edges_from(nx.selfloop_edges(G))
    filename = 'Dolphin_network'

    print(filename)
    # nx.write_gml(G, filename + '.gml')
    largest_cc = max(nx.connected_components(G), key=len)
    max_com_subgraph_yuan = G.subgraph(largest_cc).copy()
    network_size = max_com_subgraph_yuan.number_of_nodes()  # 一开始就固定住，不用重复算
    max_com_subgraph_node_list = list(max_com_subgraph_yuan.nodes())  # 一开始就固定住，不用重复
    max_com_subgraph = max_com_subgraph_yuan.copy()

    gcc_size_all_networks_dic = {}
    r_all_networks_dic = {}
    component_all_networks_dic = {}


    q = Queue()
    jobs = []

    for n in range(1,51):
        process = Process(target=mstdc_attack_one_time_explore_n,
                    args=(filename,max_com_subgraph_node_list,max_com_subgraph_yuan,
                          gcc_size_all_networks_dic, component_all_networks_dic,r_all_networks_dic,q, n))  # 这里区分union
        jobs.append(process)
        process.start()

    results = [q.get() for j in jobs]  # [[qc,r,gcc],[qc,r,gcc]]
    for process in jobs:
        process.join()
