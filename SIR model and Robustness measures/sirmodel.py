# -*- coding: utf-8 -*-
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


def sir_model(max_com_subgraph_yuan, max_index, seed_list, walk_time, psi, pir, q, cycles):
    result_dic = {}
    number_of_nodes = len(seed_list)
    i_r = np.array([number_of_nodes])
    spread_status = np.array([0] * max_index)
    #     depth_status = np.array([0]* max_index)
    #     time_status = np.array([0]* max_index)
    for seed in seed_list:
        spread_status[seed] = 1
    infecet_list = np.where(np.array(spread_status) == 1)[0]  # .tolist()
    for time_ in range(walk_time):
        for node in infecet_list:
            for nbr in max_com_subgraph_yuan[node]:  # !!!!这里不会报错，因为如果他和该节点是邻居，那么一定在最大联通片上
                if spread_status[nbr] == 0:
                    infected_probability = np.random.uniform(0, 1)
                    if infected_probability <= psi:
                        spread_status[nbr] = 1
        for node in infecet_list:
            Recover = np.random.uniform(0, 1)
            if Recover <= pir:
                spread_status[node] = 3
        infecet_list = np.where(spread_status == 1)[0]
        number = Counter(spread_status)[1] + Counter(spread_status)[3]
        i_r = np.append(i_r, number)
    result_dic[cycles] = i_r
    q.put([result_dic])


if __name__ == '__main__':
    G = nx.Graph()
    filename = 'dolphins.txt'
    source = []
    target = []
    with open(filename)as file_object:

        for line in (file_object.readlines()):
            node1, node2 = line.strip('\n').split()  # Strip('\n')先把换行符删掉，再分隔
            G.add_edge(int(node1), int(node2))
    G.remove_edges_from(nx.selfloop_edges(G))
    filename = 'Dolphin_network'

    print(filename)
    # nx.write_gml(G, filename + '.gml')
    largest_cc = max(nx.connected_components(G), key=len)
    max_com_subgraph_yuan = G.subgraph(largest_cc).copy()
    network_size = max_com_subgraph_yuan.number_of_nodes()
    max_com_subgraph_node_list = list(max_com_subgraph_yuan.nodes())
    max_com_subgraph = max_com_subgraph_yuan.copy()

    max_index = max(max_com_subgraph_node_list) + 1
    betweenness_temp = pd.read_excel(filename + '_Betweenness.xlsx', index_col=0)
    degree_temp = pd.read_excel(filename + '_Degree.xlsx', index_col=0)
    information_entropy_temp = pd.read_excel(filename + '_Information_Entropy.xlsx', index_col=0)
    cycle_ratio_temp = pd.read_excel(filename + '_Cycle_Ratio.xlsx', index_col=0)
    pagerank_temp = pd.read_excel(filename + '_Pagerank.xlsx', index_col=0)
    eigenvector_temp = pd.read_excel(filename + '_Eigenvector.xlsx', index_col=0)
    k_core_temp = pd.read_excel(filename + '_K-core.xlsx', index_col=0)
    closeness_temp = pd.read_excel(filename + '_Closeness.xlsx', index_col=0)
    clustering_temp = pd.read_excel(filename + '_Clustering.xlsx', index_col=0)
    mstdc_temp = pd.read_excel(filename + '_MSTDC.xlsx', index_col=0)

    # paramete
    pir = 1.0
    walk_time = 50
    import math


    number_nodes = network_size
    initial_nodes_number = math.ceil(number_nodes / 10)

    for psi in [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]:
        print(psi)
        # betweenness
        print('betweenness')
        seed_list = [betweenness_temp.index.tolist()[i] for i in range(initial_nodes_number)]
        ts = []
        ganran_dic = {}
        max_depth_dic = {}
        max_time_dic = {}
        average_depth_dic = {}
        average_time_dic = {}

        q = Queue()
        jobs = []
        for cycles in trange(100):
            process_ = Process(target=sir_model,
                               args=(max_com_subgraph_yuan, max_index, seed_list, walk_time, psi, pir, q,
                                     cycles))  # 这里区分union
            jobs.append(process_)
            process_.start()
        results = [q.get() for j in jobs]  # [[qc,r,gcc],[qc,r,gcc]]
        for process_ in jobs:
            process_.join()

        betweenness_dic_list = [i[0] for i in results]
        betweenness_dic = betweenness_dic_list[0]

        for i in betweenness_dic_list:
            betweenness_dic.update(i)
        betweenness_sirdf = pd.DataFrame(betweenness_dic)
        betweenness_sirdf = betweenness_sirdf.T
        betweenness_sirdf.plot.box()
        betweenness_sirdf.mean().plot()
        # plt.show()

        # degree
        print('degree')
        seed_list = [degree_temp.index.tolist()[i] for i in range(initial_nodes_number)]
        ts = []
        ganran_dic = {}
        max_depth_dic = {}
        max_time_dic = {}
        average_depth_dic = {}
        average_time_dic = {}
        q = Queue()
        jobs = []
        for cycles in trange(100):
            process_ = Process(target=sir_model,
                               args=(max_com_subgraph_yuan, max_index, seed_list, walk_time, psi, pir, q,
                                     cycles))
            jobs.append(process_)
            process_.start()
        results = [q.get() for j in jobs]  # [[qc,r,gcc],[qc,r,gcc]]
        for process_ in jobs:
            process_.join()

        degree_dic_list = [i[0] for i in results]
        degree_dic = degree_dic_list[0]
        # print(qc_all_networks_dic)
        for i in degree_dic_list:
            degree_dic.update(i)
        degree_sirdf = pd.DataFrame(degree_dic)
        degree_sirdf = degree_sirdf.T
        degree_sirdf.plot.box()
        degree_sirdf.mean().plot()
        # plt.show()

        # cycle_ratio
        print('cycle_ratio')
        seed_list = [cycle_ratio_temp.index.tolist()[i] for i in range(initial_nodes_number)]
        ts = []
        ganran_dic = {}
        max_depth_dic = {}
        max_time_dic = {}
        average_depth_dic = {}
        average_time_dic = {}
        q = Queue()
        jobs = []
        for cycles in trange(100):
            process_ = Process(target=sir_model,
                               args=(max_com_subgraph_yuan, max_index, seed_list, walk_time, psi, pir, q,
                                     cycles))
            jobs.append(process_)
            process_.start()
        results = [q.get() for j in jobs]  # [[qc,r,gcc],[qc,r,gcc]]
        for process_ in jobs:
            process_.join()

        cycle_ratio_dic_list = [i[0] for i in results]
        cycle_ratio_dic = cycle_ratio_dic_list[0]
        # print(qc_all_networks_dic)
        for i in cycle_ratio_dic_list:
            cycle_ratio_dic.update(i)
        cycle_ratio_sirdf = pd.DataFrame(cycle_ratio_dic)
        cycle_ratio_sirdf = cycle_ratio_sirdf.T
        cycle_ratio_sirdf.plot.box()
        cycle_ratio_sirdf.mean().plot()
        # plt.show()

        # information_entropy
        print('informatio_entropy')
        seed_list = [information_entropy_temp.index.tolist()[i] for i in range(initial_nodes_number)]
        ts = []
        ganran_dic = {}
        max_depth_dic = {}
        max_time_dic = {}
        average_depth_dic = {}
        average_time_dic = {}

        q = Queue()
        jobs = []
        for cycles in trange(100):
            process_ = Process(target=sir_model,
                               args=(max_com_subgraph_yuan, max_index, seed_list, walk_time, psi, pir, q,
                                     cycles))
            jobs.append(process_)
            process_.start()
        results = [q.get() for j in jobs]  # [[qc,r,gcc],[qc,r,gcc]]
        for process_ in jobs:
            process_.join()

        information_entropy_dic_list = [i[0] for i in results]
        information_entropy_dic = information_entropy_dic_list[0]
        # print(qc_all_networks_dic)
        for i in information_entropy_dic_list:
            information_entropy_dic.update(i)
        information_entropy_sirdf = pd.DataFrame(information_entropy_dic)
        information_entropy_sirdf = information_entropy_sirdf.T
        information_entropy_sirdf.plot.box()
        information_entropy_sirdf.mean().plot()
        # plt.show()

        # pagerank
        print('page_rank')
        seed_list = [pagerank_temp.index.tolist()[i] for i in range(initial_nodes_number)]
        ts = []
        ganran_dic = {}
        max_depth_dic = {}
        max_time_dic = {}
        average_depth_dic = {}
        average_time_dic = {}

        q = Queue()
        jobs = []
        for cycles in trange(100):
            process_ = Process(target=sir_model,
                               args=(max_com_subgraph_yuan, max_index, seed_list, walk_time, psi, pir, q,
                                     cycles))
            jobs.append(process_)
            process_.start()
        results = [q.get() for j in jobs]  # [[qc,r,gcc],[qc,r,gcc]]
        for process_ in jobs:
            process_.join()

        pagerank_dic_list = [i[0] for i in results]
        pagerank_dic = pagerank_dic_list[0]
        # print(qc_all_networks_dic)
        for i in pagerank_dic_list:
            pagerank_dic.update(i)
        pagerank_sirdf = pd.DataFrame(pagerank_dic)
        pagerank_sirdf = pagerank_sirdf.T
        pagerank_sirdf.plot.box()
        pagerank_sirdf.mean().plot()
        # plt.show()

        # eigenvector
        print('eigenvector')
        seed_list = [eigenvector_temp.index.tolist()[i] for i in range(initial_nodes_number)]
        ts = []
        ganran_dic = {}
        max_depth_dic = {}
        max_time_dic = {}
        average_depth_dic = {}
        average_time_dic = {}

        q = Queue()
        jobs = []
        for cycles in trange(100):
            process_ = Process(target=sir_model,
                               args=(max_com_subgraph_yuan, max_index, seed_list, walk_time, psi, pir, q,
                                     cycles))
            jobs.append(process_)
            process_.start()
        results = [q.get() for j in jobs]  # [[qc,r,gcc],[qc,r,gcc]]
        for process_ in jobs:
            process_.join()

        eigenvector_dic_list = [i[0] for i in results]
        eigenvector_dic = eigenvector_dic_list[0]
        # print(qc_all_networks_dic)
        for i in eigenvector_dic_list:
            eigenvector_dic.update(i)
        eigenvector_sirdf = pd.DataFrame(eigenvector_dic)
        eigenvector_sirdf = eigenvector_sirdf.T
        eigenvector_sirdf.plot.box()
        eigenvector_sirdf.mean().plot()
        # plt.show()

        # k_core
        print('k_core')
        seed_list = [k_core_temp.index.tolist()[i] for i in range(initial_nodes_number)]
        ts = []
        ganran_dic = {}
        max_depth_dic = {}
        max_time_dic = {}
        average_depth_dic = {}
        average_time_dic = {}

        q = Queue()
        jobs = []
        for cycles in trange(100):
            process_ = Process(target=sir_model,
                               args=(max_com_subgraph_yuan, max_index, seed_list, walk_time, psi, pir, q, cycles))
            jobs.append(process_)
            process_.start()
        results = [q.get() for j in jobs]  # [[qc,r,gcc],[qc,r,gcc]]
        for process_ in jobs:
            process_.join()

        k_core_dic_list = [i[0] for i in results]
        k_core_dic = k_core_dic_list[0]
        # print(qc_all_networks_dic)
        for i in k_core_dic_list:
            k_core_dic.update(i)
        k_core_sirdf = pd.DataFrame(k_core_dic)
        k_core_sirdf = k_core_sirdf.T
        k_core_sirdf.plot.box()
        k_core_sirdf.mean().plot()
        # plt.show()

        # closeness
        print('closeness')
        seed_list = [closeness_temp.index.tolist()[i] for i in range(initial_nodes_number)]
        ts = []
        ganran_dic = {}
        max_depth_dic = {}
        max_time_dic = {}
        average_depth_dic = {}
        average_time_dic = {}

        q = Queue()
        jobs = []
        for cycles in trange(100):
            process_ = Process(target=sir_model,
                               args=(max_com_subgraph_yuan, max_index, seed_list, walk_time, psi, pir, q, cycles))  # 这里区分union
            jobs.append(process_)
            process_.start()
        results = [q.get() for j in jobs]  # [[qc,r,gcc],[qc,r,gcc]]
        for process_ in jobs:
            process_.join()

        closeness_dic_list = [i[0] for i in results]
        closeness_dic = closeness_dic_list[0]
        # print(qc_all_networks_dic)
        for i in closeness_dic_list:
            closeness_dic.update(i)
        closeness_sirdf = pd.DataFrame(closeness_dic)
        closeness_sirdf = closeness_sirdf.T
        closeness_sirdf.plot.box()
        closeness_sirdf.mean().plot()
        # plt.show()

        # clustering
        print('clustering')
        seed_list = [clustering_temp.index.tolist()[i] for i in range(initial_nodes_number)]
        ts = []
        ganran_dic = {}
        max_depth_dic = {}
        max_time_dic = {}
        average_depth_dic = {}
        average_time_dic = {}

        q = Queue()
        jobs = []
        for cycles in trange(100):
            process_ = Process(target=sir_model,
                               args=(max_com_subgraph_yuan, max_index, seed_list, walk_time, psi, pir, q, cycles))  # 这里区分union
            jobs.append(process_)
            process_.start()
        results = [q.get() for j in jobs]  # [[qc,r,gcc],[qc,r,gcc]]
        for process_ in jobs:
            process_.join()

        clustering_dic_list = [i[0] for i in results]
        clustering_dic = clustering_dic_list[0]
        # print(qc_all_networks_dic)
        for i in clustering_dic_list:
            clustering_dic.update(i)
        clustering_sirdf = pd.DataFrame(clustering_dic)
        clustering_sirdf = clustering_sirdf.T
        clustering_sirdf.plot.box()
        clustering_sirdf.mean().plot()
        # plt.show()

        # mstdc
        print('mstdc')
        seed_list = [mstdc_temp.index.tolist()[i] for i in range(initial_nodes_number)]
        ts = []
        ganran_dic = {}
        max_depth_dic = {}
        max_time_dic = {}
        average_depth_dic = {}
        average_time_dic = {}

        q = Queue()
        jobs = []
        for cycles in trange(100):
            process_ = Process(target=sir_model,
                               args=(max_com_subgraph_yuan, max_index, seed_list, walk_time, psi, pir, q, cycles))  # 这里区分union
            jobs.append(process_)
            process_.start()
        results = [q.get() for j in jobs]  # [[qc,r,gcc],[qc,r,gcc]]
        for process_ in jobs:
            process_.join()

        mstdc_dic_list = [i[0] for i in results]
        mstdc_dic = mstdc_dic_list[0]
        # print(qc_all_networks_dic)
        for i in mstdc_dic_list:
            mstdc_dic.update(i)
        mstdc_sirdf = pd.DataFrame(mstdc_dic)
        mstdc_sirdf = mstdc_sirdf.T
        mstdc_sirdf.plot.box()
        mstdc_sirdf.mean().plot()
        # plt.show()


        merge_scatter_df = pd.DataFrame(
            [betweenness_sirdf.mean(), degree_sirdf.mean(), cycle_ratio_sirdf.mean(), information_entropy_sirdf.mean(),
             pagerank_sirdf.mean(),
             eigenvector_sirdf.mean(), k_core_sirdf.mean(), closeness_sirdf.mean(), clustering_sirdf.mean(),
             mstdc_sirdf.mean()],
            index=['betweenness', 'degree_Centrality', 'Cycle_Ratio', 'Information_Entropy', 'Pagerank', 'Eigenvector',
                   'K_core', 'Closeness', 'Clustering', 'Mstdc'])
        merge_scatter_df
        merge_scatter_df.T.plot()
        merge_scatter_df = merge_scatter_df.T
        merge_scatter_df.to_excel(
            filename + '-chuanbo_mergedf_fanxiu' + str(psi) + '_initial_nodes_number_' + str(initial_nodes_number) + '.xlsx')

        merge_scatter_df = merge_scatter_df / number_nodes
        merge_scatter_df.plot()
        plt.show()
        merge_scatter_df.to_excel(filename + '-chuanbo_mergedf_fanxiu' + str(psi) + '_initial_nodes_number_' + str(
            initial_nodes_number) + '显示百分比.xlsx')