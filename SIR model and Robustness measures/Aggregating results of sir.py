import pandas as pd
import numpy as np

if __name__ == '__main__':
    Network = 'Dolphin_network'
    psi_list = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    combination_df = pd.DataFrame()
    for psi in psi_list:
        filename= Network + '-chuanbo_mergedf_fanxiu' + str(psi) + '_initial_nodes_number_' + str(
            7) + '显示百分比.xlsx'
        df = pd.read_excel(filename, index_col=0)
        combination_df[psi] = list(df.loc[50])
    combination_df.index= df.columns
    combination_df = combination_df.T
    combination_df = combination_df[['Mstdc', 'degree_Centrality', 'K_core', 'Closeness', 'betweenness', 'Eigenvector',
                                     'Cycle_Ratio', 'Information_Entropy', 'Pagerank', 'Clustering']]
    combination_df.to_excel(Network+'_sir_results_aggregation.xlsx')