# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:09:15 2022

@author: user
"""
# =============================================================================
# 
# # #실습 1번
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# 
# # Data = pd.read_csv('lin_regression_data_03.csv',names = ['Age','Height'])
# 
# # x = np.asarray(Data['Age'])             #x는 나이
# # y = np.asarray(Data['Height'])          #y는 키
# 
# 
# # plt.plot(x, y, 'co')
# # plt.xlabel("Age")
# # plt.ylabel("Height")
# # plt.legend(['Age & Height'], )
# # plt.grid(True)
# # plt.title(" #1")
# # plt.show()
# =============================================================================


# =============================================================================
# # #실습 2번
# #data 순서 변경하지 말것
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# 
# T_x = []
# S_x = []
# T_y = []
# S_y = []
# 
# Data = pd.read_csv('lin_regression_data_03.csv',names = ['Age','Height'])
# 
# x = np.asarray(Data['Age'])             #x는 나이
# y = np.asarray(Data['Height'])          #y는 키
# 
# T_x = x[:20]
# S_x = x[20:25]
# 
# T_y = y[:20]
# S_y = y[20:25]
# 
# plt.plot(T_x, T_y ,'y>', S_x, S_y , 'co')
# plt.xlabel("Age")
# plt.ylabel("Height")
# plt.legend(['T (training set)', 'S (test set)'], )
# plt.grid(True)
# plt.title(" #2")
# plt.show()
# =============================================================================





# # w0 = np.sum(((1/N)*(y)*(x-(1/N)*np.sum(x)))/(((1/N)*np.sum(x**2))-(np.sum(((1/N)*x)))**2))
# # w1 = (1/N)*np.sum((y - w0*x))
# # 
# for k in all_K:
    
#     K_list = list(range(0,k))
#     for r in K_list:
#         avg.append(np.min(x)+(((np.max(x)-np.min(x))*r)/(k-1)))     #평

#     o=((((np.max(x)-np.min(x))*r)/(k-1)))       #분산
    
# for c in list(range(0,x.size*k)):
    
#     if count ==k:
#         count = 0
#         count_x = count_x + 1
#         count_y = count_y + 1
        
#     phi.append(math.exp((-1/2)*(((x[count_x]-avg[count_x])/o)**2)))
#     test_get_MSE = (1/N)*(np.sum((((phi)-y[count_y])**2)))
#     MSE.append(test_get_MSE)


# =============================================================================


# =============================================================================
# # #실습 3번
# # #K 가 6개면 매개변수도 6개 출력
# 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# 
# Data = pd.read_csv('lin_regression_data_03.csv',names = ['Age','Height'])
# 
# T_x = []
# T_y = []
# 
# S_x = []
# S_y = []
# 
# 
# x = np.asarray(Data['Age'])             #x는 나이
# y = np.asarray(Data['Height'])          #y는 키
# N = len(Data['Height'])
# 
# x=x[0:25]
# y=y[0:25]
# 
# 
# T_x = x[:20]
# S_x = x[20:25]
# 
# T_y = y[:20]
# S_y = y[20:25]
# 
# # xsort=np.array(Data['x'].values.tolist())
# # xsort=xsort[0:25]
# # xsort.sort()
# 
# total_mse=[]
# total_k=[6, 7, 8, 9 , 10 ,11, 12, 13]
# 
# for k in total_k:
#     mu=[]
# 
#     #k=8
# 
#     listk = list(range(0,k))
# 
#     #print(listk)
# 
#     for i in listk:
#         mu.append(min(T_x)+((max(T_x)-min(T_x))/(k-1))*i)
#         
#         
#     sig = (max(T_x)-min(T_x))/(k-1)
#     print("k=",k)
#     print('mu=',mu,'\nsig=',sig)
# 
# 
#     pi=[]
# 
#     counter=xcounter=0;
# 
#     for j in list(range(0,T_x.size*k)):
#         
#         
#         if counter==k:
#             counter=0
#             xcounter=xcounter+1
#         
#         #print(j,muc,x[xcounter])
#         pi.append(math.exp((-1/2)*((T_x[xcounter]-mu[counter])/sig)**2))
#         counter=counter+1
#         
#         
# 
#     pi=np.array(pi)
# 
#     pi2=pi.reshape(T_x.size,k)
# 
# 
#     add_1=np.full((T_x.size,1),1)
# 
#     pi2=np.append(pi2, add_1, axis = 1)
# 
# 
#     w=np.linalg.inv(np.transpose(pi2)@pi2)@(np.transpose(pi2)@T_y)
# 
#     print('\nw=',w)
#     
# 
#     y_hat=0
#     
# 
#     for l in list(range(0,k)):
#         y_hat=y_hat+(w[l]*(np.e**((-1/2)*((T_x-mu[l])/sig)**2)))
#         
# 
#     y_hat=y_hat+w[k]
# 
# 
# 
#     mse=(1/T_x.size)*sum((y_hat-T_y)**2)
#     total_mse.append(mse)
# 
#     print("mse=",mse)
#     
#     #graph
#     y_hat_gp=0
#     for l in list(range(0,k)):
#         y_hat_gp = y_hat_gp + (w[l]*(np.e**((-1/2)*((T_x-mu[l])/sig)**2)))
#         
# 
#     y_hat_gp = y_hat_gp + w[k]
#     
#     
# plt.figure()
# plt.title('#3')
# plt.plot(total_k,total_mse,'c--o')
# plt.xlabel("k")
# plt.ylabel("MSE")
# plt.grid(True)
# plt.show()
# =============================================================================
 
#AI

#평균제곱오차 ( 훈련데이터, 일반화 성분 => 총 2가지 출력) // MSE data to set traning data and checking data
#과적합  과소적합 나오는 그래프가 출력되었으면 함 // test

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

Data = pd.read_csv('lin_regression_data_03.csv',names = ['Age','Height'])

T_x = []        #training set
T_y = []

S_x = []        #checking set
S_y = []

x = np.asarray(Data['Age'])             #x == Age
y = np.asarray(Data['Height'])          #y == Height
N = len(Data['Height'])


T_x = x[:20]                            #divide 0~20 data for training
S_x = x[20:25]                          #divide 21~25 data for setting

T_y = y[:20]
S_y = y[20:25]


total_mse_training = []
total_mse_check = []
total_k=[6, 7, 8, 9 , 10 ,11, 12, 13]

for k in total_k:
    mu=[]

    listk = list(range(0,k))

    #print(listk)

    for i in listk:
        mu.append(min(T_x)+((max(T_x)-min(T_x))/(k-1))*i)  
        
    sig= (max(T_x)-min(T_x))/(k-1)


    pi=[]     
    pi_S=[]

    counter=xcounter=ycounter =0

    for j in list(range(0,T_x.size*k)):
        
        
        if counter==k:
            counter=0
            xcounter=xcounter+1
        
        #print(j,muc,x[xcounter])
        pi.append(math.exp((-1/2)*((T_x[xcounter]-mu[counter])/sig)**2))
        counter=counter+1
          

    pi=  np.array(pi)

    pi2=pi.reshape(T_x.size,k)

    add_1=np.full((T_x.size,1),1)

    pi2=np.append(pi2, add_1, axis = 1)


    w_T=np.linalg.inv(np.transpose(pi2)@pi2)@(np.transpose(pi2)@T_y)

    y_hat=0
    y_hat_S = 0
    

    for l in list(range(0,k)):
        y_hat=   y_hat+(w_T[l]*(np.e**((-1/2)*((T_x-mu[l])/sig)**2)))
        y_hat_S= y_hat_S+(w_T[l]*(np.e**((-1/2)*((S_x-mu[l])/sig)**2)))
        

    y_hat=y_hat+w_T[k]
    y_hat_S=y_hat_S+w_T[k]

    
    mse_T=(1/T_x.size)*sum((y_hat-T_y)**2)
    total_mse_training.append(mse_T)
        
    mse_S=(1/S_x.size)*sum((y_hat_S-S_y)**2)
    total_mse_check.append(mse_S)


    
    #graph
    y_hat_gp_T=0
    y_hat_gp_S=0
    l =0
    
    for l in list(range(0,k)):
        y_hat_gp_T = y_hat_gp_T + (w_T[l]*(np.e**((-1/2)*((T_x-mu[l])/sig)**2)))
        y_hat_gp_S = y_hat_gp_S + (w_T[l]*(np.e**((-1/2)*((S_x-mu[l])/sig)**2)))

    y_hat_gp_T = y_hat_gp_T + w_T[k]
    y_hat_gp_S = y_hat_gp_S + w_T[k]
    
    
plt.figure()
plt.title('#4')
plt.plot(total_k,total_mse_training,'c--o',total_k, total_mse_check, 'r--^')
plt.xlabel("k")
plt.ylabel("MSE")
plt.grid(True)
plt.show()



# =============================================================================
# 
# #실습 5번
# #5겹 cross 로 분할 ( 순서 변경 x)
# #하나의subset 을 그래프를 그려라 ( 마커이용 )
# #찢어진 5개의 데이터를 각각 훈련시켜 small - > k
# 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# 
# Data = pd.read_csv('lin_regression_data_03.csv',names = ['Age','Height'])
# 
# x = np.asarray(Data['Age'])             #x는 나이
# y = np.asarray(Data['Height'])          #y는 키
# N = len(Data['Height'])
# 
# # a = x.slice(0, 25[, 5])
# # y.split(',', 5)
# 
# print(x)
# print(y)
# 
# x1 = np.split(x, 5)[0]
# x2 = np.split(x, 5)[1]
# x3 = np.split(x, 5)[2]
# x4 = np.split(x, 5)[3]
# x5 = np.split(x, 5)[4]
# 
# y1 = np.split(y, 5)[0]
# y2 = np.split(y, 5)[1]
# y3 = np.split(y, 5)[2]
# y4 = np.split(y, 5)[3]
# y5 = np.split(y, 5)[4]
# 
# Q1 = (x1, y1)
# 
# Q2 = (x2, y2)
# 
# Q3 = (x3, y3)
# 
# Q4 = (x4, y4)
# 
# Q5 = (x5, y5)
# 
# print(Q1)
# print(Q2)
# print(Q3)
# print(Q4)
# print(Q5)
# 
# plt.figure()
# plt.title('#5')
# plt.plot(x1, y1,'co')
# plt.plot(x2, y2,'ro')
# plt.plot(x3, y3,'bo')
# plt.plot(x4, y4,'mo')
# plt.plot(x5, y5,'go')
# plt.xlabel("x Age")
# plt.ylabel("y Height")
# plt.legend(['1~5', '6~10', '11~15', '16~20', '21~25'])
# plt.grid(True)
# plt.show()
# 
# 
# =============================================================================

# =============================================================================
# 
#     
# 
# #실습 6번  // total_k=[9]를 이용
# # 5겹 홀드아웃을 만들어 - > 선형 기저함수 모델을 K=9으로 고정
# #5개의 홀드아웃에 대한 최적해를 계산하는 것을 만들고
# #5개의 일반화 오차 구하기
# 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# 
# Data = pd.read_csv('lin_regression_data_03.csv',names = ['Age','Height'])
# 
# x = np.asarray(Data['Age'])             #x는 나이
# y = np.asarray(Data['Height'])          #y는 키
# N = len(Data['Height'])
# 
# 
# x1 = np.split(x, 5)[0]
# x2 = np.split(x, 5)[1]
# x3 = np.split(x, 5)[2]
# x4 = np.split(x, 5)[3]
# x5 = np.split(x, 5)[4]
# 
# y1 = np.split(y, 5)[0]
# y2 = np.split(y, 5)[1]
# y3 = np.split(y, 5)[2]
# y4 = np.split(y, 5)[3]
# y5 = np.split(y, 5)[4]
# 
# 
# X = np.vstack((x1, x2, x3, x4, x5))
# Y = np.vstack((y1, y2, y3, y4, y5))
# 
# # print(X)
# 
# # T 훈련 x1[1:4]
# # s 검증 x1[0]
# # T 훈련 x1[0, 2:4]or get1 = []
# 
# get1_x = [] #훈련 데이터 리스트
# get2_x = []
# get3_x = []
# get4_x = []
# get5_x = []
# 
# 
# get1_y = [] #훈련 데이터 리스트
# get2_y = []
# get3_y = []
# get4_y = []
# get5_y = []
# 
# 
# total_mse_training = []
# total_mse_check = []
# total_k=[9]
# 
# T_x = []
# T_y = []
# 
# S_x = []
# S_y = []
# 
# print(X)
# print()
# 
# 
# #검증할때 검증 데이터
# for a in range(5):
#     S_x = X[a]   #S_x setting 검증 set
#     S_y = Y[a]   #S_y
#     
# # X = np.vstack((x1, x2, x3, x4, x5))
# # Y = np.vstack((y1, y2, y3, y4, y5))
# 
# h = 0
# for h in range(5):
#     # REMOVE = np.delete(X, )
#     after =np.delete(X, h, axis = 0)  #훈련 데이터 set
#     print(after)
#     print()
# 
#     T_x = np.hstack(after)
#     print(T_x)
#     print()
# 
#     
#     for k in total_k:
#         mu=[]
#     
#         listk = list(range(0,k))
#     
#     
#         for i in listk:
#             mu.append(np.min(T_x)+((np.max(T_x)-np.min(T_x))/(k-1))*i)  
#             
#         sig= (np.max(T_x)-np.min(T_x))/(k-1)
#     
#     
#         pi=[]     
#         pi_S=[]
#     
#         counter=xcounter=ycounter =0
#         
#         for j in list(range(0,T_x.size*k)):
#             
#             
#             if counter == k:
#                 counter=0
#                 xcounter=xcounter+1
#                 
#             # for c in range(len(X)):
#             #     #print(j,muc,x[xcounter])
#                 pi.append(math.exp((-1/2)*((T_x[xcounter]-mu[counter])/sig)**2))
#                 print(pi)
#                 counter=counter+1
#     
#         pi=  np.array(pi)
#     
#         pi2= pi.reshape(T_x.size,k)
#     
#         add_1=np.full((T_x.size,1),1)
#     
#         pi2=np.append(pi2, add_1, axis = 1)
#     
#     
#         w_T=np.linalg.inv(np.transpose(pi2)@pi2)@(np.transpose(pi2)@T_y)
#     
#         y_hat=0
#         y_hat_S = 0
#         
#     
#         for l in list(range(0,k)):
#             y_hat=   y_hat+(w_T[l]*(np.e**((-1/2)*((T_x-mu[l])/sig)**2)))
#             y_hat_S= y_hat_S+(w_T[l]*(np.e**((-1/2)*((S_x-mu[l])/sig)**2)))
#             
#     
#         y_hat=y_hat+w_T[k]
#         y_hat_S=y_hat_S+w_T[k]
#     
#         
#         mse_T=(1/T_x.size)*sum((y_hat-T_y)**2)
#         total_mse_training.append(mse_T)
#             
#         mse_S=(1/S_x.size)*sum((y_hat_S-S_y)**2)
#         total_mse_check.append(mse_S)
#     
#     
#         
#         #graph
#         y_hat_gp_T=0
#         y_hat_gp_S=0
#         l =0
#         
#         for l in list(range(0,k)):
#             y_hat_gp_T = y_hat_gp_T + (w_T[l]*(np.e**((-1/2)*((T_x-mu[l])/sig)**2)))
#             y_hat_gp_S = y_hat_gp_S + (w_T[l]*(np.e**((-1/2)*((S_x-mu[l])/sig)**2)))
#     
#         y_hat_gp_T = y_hat_gp_T + w_T[k]
#         y_hat_gp_S = y_hat_gp_S + w_T[k]
#         
#      
#     print("매개변수 ="  ,total_mse_training)
#     print("MSE = ", total_mse_check)
# 
# # plt.figure()
# # plt.title('#6')
# # plt.plot(total_k,total_mse_training,'c--o',total_k, total_mse_check, 'r--^')
# # plt.xlabel("k")
# # plt.ylabel("MSE")
# # plt.grid(True)
# # plt.show()
# 
# =============================================================================


#실습 7번
#5개 그래프    실습 6번에서 한것을 그래프로 출력하라 

#실습 8번
#실습 7번에서 구한 그래프들의 평균을 하나의 그래프에 출력하라