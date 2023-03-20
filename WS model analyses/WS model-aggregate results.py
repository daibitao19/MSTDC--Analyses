import pandas as pd
import numpy as np

if __name__ == '__main__':
    N = [500, 1000]
    M = [ 4,  6,  8,  10]
    P = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    # Parameters = [i for i in range(100)]
    #wr = pd.ExcelWriter("WS_500_2000_m=2_11_p_100R_from_gcc_size_promotion_rate.xlsx")
    wr = pd.ExcelWriter("WS_500_1000_m=2_10_p_10%gcc_only_promotion_rate_based_on_networksize.xlsx")
    df=pd.read_excel('WS_500_1000_2000_m=2_11_p_10%gcc_only.xlsx')
    df['Min_degree_bet']=df[['Degree','Betweenness']].min(axis=1)
    df['Our_Min-Min']=df['MIN']- df['Min_degree_bet']
    df['Min-promotion']=df['Our_Min-Min']/df['Min_degree_bet']

    df['Our_Mean-Min'] = df['MSTDC'] -  df['Min_degree_bet']
    df['Mean-promotion'] = df['Our_Mean-Min'] / df['Min_degree_bet']
    df.to_excel(wr,sheet_name="原始")



    combination_df=pd.DataFrame()
    col_index=[i for i in range(2,12)]      #0-9, 10-19, 20-29   分
    in_index = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    for i in range(10):

        combination_df[i]= list(df['Min-promotion'])[i*7:i*7+7]

    combination_df.columns=col_index
    combination_df.index = in_index
    print('500-Min')
    print(combination_df)
    combination_df.to_excel(wr,sheet_name="500-Min")


    combination_df=pd.DataFrame()

    for i in range(10,20):

        combination_df[i]= list(df['Min-promotion'])[i*7:i*7+7]
    #
    combination_df.columns=col_index
    combination_df.index = in_index
    print('1000')
    print(combination_df)
    combination_df.to_excel(wr,sheet_name="1000-Min")

    combination_df=pd.DataFrame()

    for i in range(20,30):

        combination_df[i]= list(df['Min-promotion'])[i*7:i*7+7]

    combination_df.columns=col_index
    combination_df.index = in_index
    print('2000')
    print(combination_df)
    combination_df.to_excel(wr,sheet_name="2000-Min")

    combination_df=pd.DataFrame()
    col_index=[i for i in range(2,12)]      #0-9, 10-19, 20-29   分三个70=10x7
    in_index = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    for i in range(10):

        combination_df[i]= list(df['Mean-promotion'])[i*7:i*7+7]

    combination_df.columns=col_index
    combination_df.index = in_index
    print('500')
    print(combination_df)
    combination_df.to_excel(wr,sheet_name="500-Mean")


    combination_df=pd.DataFrame()

    for i in range(10,20):
        combination_df[i]= list(df['Mean-promotion'])[i*7:i*7+7]
    combination_df.columns=col_index
    combination_df.index = in_index
    print('1000')
    print(combination_df)
    combination_df.to_excel(wr,sheet_name="1000-Mean")
    #
    #
    combination_df=pd.DataFrame()


    wr.save()