import pandas as pd
import numpy as np
import os
import networkx as nx

if __name__ == '__main__':
    N = [500, 1000]
    M = [4, 6, 8, 10, ]
    P = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    topology = {}
    col_index = []
    for n in N:
        for m in M:
            for p in P:
                #             file_name='WS_'+str(n)+'_'+str(m)+'_'+str(p)+'_Network.gml'
                #             print(file_name)
                #             print('P=',p)
                G = nx.watts_strogatz_graph(n, m, p)
                col_index.append('WS_' + str(n) + '_' + str(m) + '_' + str(p))
                largest_cc = max(nx.connected_components(G), key=len)
                max_com_subgraph_yuan = G.subgraph(largest_cc).copy()
                network_size = max_com_subgraph_yuan.number_of_nodes()  # 一开始就固定住，不用重复算
                topology.setdefault('网络的节点数为', []).append(max_com_subgraph_yuan.number_of_nodes())
                topology.setdefault('网络的边数为', []).append(max_com_subgraph_yuan.number_of_edges())
                topology.setdefault('网络的密度为', []).append(nx.density(max_com_subgraph_yuan))
                topology.setdefault('网络的平均度为', []).append(
                    max_com_subgraph_yuan.number_of_edges() * 2 / (max_com_subgraph_yuan.number_of_nodes()))
                topology.setdefault('网络的集聚系数为', []).append(nx.average_clustering(max_com_subgraph_yuan))
                topology.setdefault('网络的平均最短路径长度为', []).append(nx.average_shortest_path_length(max_com_subgraph_yuan))
                topology.setdefault('网络的直径为', []).append(nx.diameter(max_com_subgraph_yuan))
    # df=pd.DataFrame(topology)
    # col_index.reverse()
    # df.index=col_reverse
    # df.to_excel('500_1000_4_10_topology_merge_inverse.xlsx')




