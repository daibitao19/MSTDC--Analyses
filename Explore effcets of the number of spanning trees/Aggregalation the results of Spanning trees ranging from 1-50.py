import pandas as pd
import numpy as np

if __name__ == '__main__':
    N = [i for i in range(1,50)]

    combination_df = pd.DataFrame()
    for n in N:
        filename= 'Dolphin_network_' +str(n) +'_mstdc_gcc_size.xlsx'
        df = pd.read_excel(filename,index_col=0)
        combination_df[n] = list(df['Col_mean'])
    combination_df.index = df.index
    combination_df.to_excel('Dolphin_Network_n=1_50_R_mean.xlsx')

    combination_df = pd.DataFrame()
    for n in N:
        filename = 'Dolphin_network_' + str(n) + '_mstdc_gcc_size.xlsx'
        df = pd.read_excel(filename, index_col=0)
        combination_df[n] = list(df['Min'])
    combination_df.index = df.index
    combination_df.to_excel('Dolphin_Network_n=1_50_R_min.xlsx')