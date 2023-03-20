import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from tqdm import tqdm,trange
from collections import Counter, deque

if __name__ == '__main__':
    G = nx.Graph()
    filename = 'dolphins.txt'
    source = []
    target = []
    with open(filename)as file_object:
        # lines=file_object.readlines()
        for line in (file_object.readlines()):
            node1, node2 = line.strip('\n').split()
            G.add_edge(int(node1), int(node2))
    G.remove_edges_from(nx.selfloop_edges(G))
    filename = 'Dolphin_network'
    print(filename)
    largest_cc = max(nx.connected_components(G), key=len)
    max_com_subgraph_yuan = G.subgraph(largest_cc).copy()
    network_size = max_com_subgraph_yuan.number_of_nodes()
    max_com_subgraph_node_list = list(max_com_subgraph_yuan.nodes())
    max_com_subgraph = max_com_subgraph_yuan.copy()

    print('网络的节点数为:', max_com_subgraph_yuan.number_of_nodes())
    print('网络的边数为:', max_com_subgraph_yuan.number_of_edges())
    print('网络的直径为:', nx.diameter(max_com_subgraph_yuan))
    print('网络的平均度为:', max_com_subgraph_yuan.number_of_edges() * 2 / (max_com_subgraph_yuan.number_of_nodes()))
    print('网络的平均最短路径长度为:', nx.average_shortest_path_length(max_com_subgraph_yuan))
    print('网络的集聚系数为:', nx.average_clustering(max_com_subgraph_yuan))


    degree_dic = {}
    for node in max_com_subgraph_node_list:
        degree_dic[node] = max_com_subgraph_yuan.degree(node)

    second_order_degree_dic = {}
    for node in max_com_subgraph_node_list:
        second_order_degree_dic[node] = 0
    for node in max_com_subgraph_node_list:
        for nbr in max_com_subgraph_yuan[node]:
            second_order_degree_dic[node] += degree_dic[nbr]

    second_order_degree = list(second_order_degree_dic.values())
    b2 = sum(second_order_degree) / len(second_order_degree)
    print('平均二阶度:',b2)  # 平均二阶度
    one_order_degree = list(degree_dic.values())
    a1 = sum(one_order_degree) / len(one_order_degree)
    print('平均度:',a1)  # 平均度
    print('异质平均场（分母为相减）:',a1 / (b2 - a1))  # 异质平均场
    print('异质平均场:',a1 / b2)  # 异质平均场