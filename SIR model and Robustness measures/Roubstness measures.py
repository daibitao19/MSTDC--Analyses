import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import random
from collections import Counter, deque
from copy import deepcopy
from math import ceil, sqrt, floor
import math


def deleted_edge(node_list):  # graph图  s指的是开始结点
    edgelist = []
    for node in node_list:
        for nbr in max_com_subgraph_yuan[node]:  # 最大联通的边,不在最大连通片的边不考虑
            edgelist.append((node, nbr))
    return edgelist


def GCC_NC_connected_components_attack_for_mstdc(G, nodes_sort,gcc_size_all_networks_mstdc_dic, component_all_networks_mstdc_dic,time,q):
    G_re = deepcopy(G)
    Network_size = len(G_re)

    largest_all = np.zeros(Network_size + 1)
    number_of_components = np.zeros(Network_size + 1)
    for i in range(Network_size):
        largest_cc = len(max(nx.connected_components(G_re), key=len))
        number_cc = len([len(c) for c in sorted(nx.connected_components(G_re), key=len)])
        remove_nodes = nodes_sort[i]
        G_re.remove_node(remove_nodes)#
        largest_all[i] = largest_cc / Network_size
        number_of_components[i] = number_cc
    gcc_size_all_networks_mstdc_dic[time] = largest_all
    component_all_networks_mstdc_dic[time] = number_of_components
    q.put([gcc_size_all_networks_mstdc_dic, component_all_networks_mstdc_dic,np.mean(largest_all)])

def Centrality_qc_attack(max_com_subgraph_yuan, centrality_function,gcc_community_dic_one,component_community_dic_one,r_community_dic_one):
    i, j = centrality_function[0], centrality_function[1]
    value_dic = eval(j)
    temp = get_descending_temp(value_dic)
    nodes_sort = temp.index.tolist()
    gcc_size, components_number, r = GCC_NC_connected_components_attack(max_com_subgraph_yuan, nodes_sort)
    gcc_community_dic_one[i] = gcc_size
    component_community_dic_one[i] = components_number
    r_community_dic_one[i] = r