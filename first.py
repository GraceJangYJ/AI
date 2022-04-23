# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:25:52 2022

@author: user
"""

#실습 1번

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import csv

data = pd.read_csv('lin_regression_data_01.csv',names = ['Weight','Height'])
# from numpy import loadtxt

x = np.asarray(data['Weight'])    #x는 몸무게
y = np.asarray(data['Height'])        #y는 키


plt.plot(x, y, 'b>')
plt.xlabel("Weight")
plt.ylabel("Height")
plt.legend(['Weight', 'Height'], )
plt.grid(True)
plt.title(" TEST #1")
plt.show()

# =============================================================================
# #실습 2번
# 
#   
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # import csv
# 
# data = pd.read_csv('lin_regression_data_01.csv',names = ['Weight','Height'])
# # from numpy import loadtxt
# 
# x = np.asarray(data['Weight'])    #x는 몸무게
# y = np.asarray(data['Height'])        #y는 키
# N = len(data['Height'])
# a = np.dot(x,y) #행렬 합 후 더하기
# 
# w0 = np.sum(((1/N)*(y)*(x-(1/N)*np.sum(x)))/(((1/N)*np.sum(x**2))-(np.sum(((1/N)*x)))**2))
# w1 = (1/N)*np.sum((y - w0*x))   
#     
# print()
# print("w0 = ", w0)
# print()
# print("w1 = ", w1)
# print()
# 
# 
# #wo = 1.xxxx
# #w1 = 7.xxxxx
# 
# 
# #np.dot(a,b) 두 행렬을 이용해서 사용해보기  
#         
# =============================================================================
# =============================================================================
# plt.plot(x, y, 'b>')
# plt.xlabel("Weight")
# plt.ylabel("Height")
# plt.legend(['Weight', 'Height'], )
# plt.grid(True)
# plt.title(" TEST #1")
# plt.show()
# =============================================================================



# =============================================================================
# #실습 3번
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # import csv
# 
# data = pd.read_csv('lin_regression_data_01.csv',names = ['Weight','Height'])
# # from numpy import loadtxt
# 
# x = np.asarray(data['Weight'])    #x는 몸무게
# y = np.asarray(data['Height'])        #y는 키
# N = len(data['Height'])
# 
# w0 = np.sum(((1/N)*(y)*(x-(1/N)*np.sum(x)))/(((1/N)*np.sum(x**2))-(np.sum(((1/N)*x)))**2))
# w1 = (1/N)*np.sum((y - w0*x))   
# 
#     
# yhat = w0*(x)+w1
# 
# plt.plot(x, yhat, 'bo', x, y, 'ro')
# plt.xlabel("x")
# plt.ylabel("y hat & y")
# plt.legend(['yhat', 'y'])
# plt.grid(True)
# plt.title('')
# plt.show()
# 
# =============================================================================
# =============================================================================
# #실습 4번
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # import csv
# 
# data = pd.read_csv('lin_regression_data_01.csv',names = ['Weight','Height'])
# # from numpy import loadtxt
# 
# x = np.asarray(data['Weight'])    #x는 몸무게
# y = np.asarray(data['Height'])        #y는 키
# N = len(data['Height'])
# 
# w0 = np.sum(((1/N)*(y)*(x-(1/N)*np.sum(x)))/(((1/N)*np.sum(x**2))-(np.sum(((1/N)*x)))**2))
# w1 = (1/N)*np.sum((y - w0*x))   
# 
#     
# yhat = w0*(x)+w1
# 
# 
# MSE = np.sum((1/N)*((yhat-y)**2))
# 
# print("Mean squared error(MSE) = ",MSE)
# =============================================================================
# =============================================================================
# 
# #실습 5번은 for문을 이용해서 3000번 한정시켜랏
# #외부 함수 머신러닝을 하는 함수를 활용하지 말것
# #실습 5번 Gradient descent 경사하강법
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# 
# data = pd.read_csv('lin_regression_data_01.csv',names = ['Weight','Height'])
# # from numpy import loadtxt
# 
# x = np.asarray(data['Weight'])    #x는 몸무게
# y = np.asarray(data['Height'])        #y는 키
# N = len(data['Height'])
# a = 0.000015                         #학습률 learning rate
# 
# 
# # w0 = np.sum(((1/N)*(y)*(x-(1/N)*np.sum(x)))/(((1/N)*np.sum(x**2))-(np.sum(((1/N)*x)))**2))
# # w1 = (1/N)*np.sum((y - w0*x))  
# G0 = []
# G1 = []
# w0 = [ 1, ]
# w1 = [ 4, ]
# # w1 = [1:4000]
# 
# print(w0)
# 
# for i in range(10):
#     G0= w0[i]-np.sum(((a*2)/N)*x*((w0[i]*x)+w1[i]-y))  #w0[i+1]
#     w0.append(G0)
#     # print("w0 = ",w0)
#     G1 = w1[i]-np.sum(((a*2)/N)*(x*w0[i]+w1[i]-y))      #w1[i+1]
#     w1.append(G1)
#     # print("w1 = ",w1)
# print("w0 = ", w0)
# print("w1 = ", w1)
# 
# 
# =============================================================================

# =============================================================================
# 
# #실습 6번 Gradient descent 경사하강법
# # 테스트 K 3000번에서 10000번으로 변경함 오류 나타나면 다시 체크할 것
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# 
# data = pd.read_csv('lin_regression_data_01.csv',names = ['Weight','Height'])
# # from numpy import loadtxt
# 
# x = np.asarray(data['Weight'])    #x는 몸무게
# y = np.asarray(data['Height'])        #y는 키
# N = len(data['Height'])
# a = 0.000015                         #학습률 learning rate
# K = 10000 
# 
# # w0 = np.sum(((1/N)*(y)*(x-(1/N)*np.sum(x)))/(((1/N)*np.sum(x**2))-(np.sum(((1/N)*x)))**2))
# # w1 = (1/N)*np.sum((y - w0*x))  
# G0 = []
# G1 = []
# w0 = [ 1, ]
# w1 = [ 4, ]
# 
# 
# for i in range(K):
#     G0= w0[i]-np.sum(((a*2)/N)*x*((w0[i]*x)+w1[i]-y))  #w0[i+1]
#     w0.append(G0)
#     # print("w0 = ",w0)
#     G1 = w1[i]-np.sum(((a*2)/N)*(x*w0[i]+w1[i]-y))      #w1[i+1]
#     w1.append(G1)
#     # print("w1 = ",w1)
# # print("w0 = ", w0)
# # print("w1 = ", w1)
# 
#       
# yhat = w0[3000]*(x)+w1[3000]
# 
# 
# MSE = np.sum((1/N)*((yhat-y)**2))
# print("학습률 = a(알파) = ",a)
# print()
# print("초기값 w0 =",w0[0], ' ,'," 초기값 w1=",w1[0])
# print()
# print("반복회수 : ", K)
# print()
# print("최종 평균제곱 오차 : ",MSE)
# print()
# print("최적매개변수 w0 = ",w0[K], ' ,'," 최적매개변수 w1= ",w1[K])
# print()
# 
# =============================================================================

# =============================================================================
# #실습 7번 수렴하는지 확인할 것
# 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# 
# data = pd.read_csv('lin_regression_data_01.csv',names = ['Weight','Height'])
# # from numpy import loadtxt
# 
# x = np.asarray(data['Weight'])    #x는 몸무게
# y = np.asarray(data['Height'])        #y는 키
# N = len(data['Height'])
# a = 0.000015                         #학습률 learning rate
# 
# 
# # w0 = np.sum(((1/N)*(y)*(x-(1/N)*np.sum(x)))/(((1/N)*np.sum(x**2))-(np.sum(((1/N)*x)))**2))
# # w1 = (1/N)*np.sum((y - w0*x))  
# G0 = []
# G1 = []
# w0 = [ 1, ]
# w1 = [ 4, ]
# yhat = []
# MSE = []
# K = 3000
# 
# for i in range(K):
#     G0= w0[i]-np.sum(((a*2)/N)*x*((w0[i]*x)+w1[i]-y))  #w0[i+1]
#     w0.append(G0)
#     # print("w0 = ",w0)
#     G1 = w1[i]-np.sum(((a*2)/N)*(x*w0[i]+w1[i]-y))      #w1[i+1]
#     w1.append(G1)
#       
#     yhat = w0[i]*(x)+w1[i]
# 
#     r= np.sum((1/N)*((yhat-y)**2))
#     MSE.append(r)
# 
# 
# plt.plot( w0, 'b:o', w1, 'r--')
# plt.xlabel("Step")
# plt.ylabel("w0, & w1")
# plt.legend(['w0', 'w1'])
# plt.grid(True)
# plt.title('w0, w1')
# plt.show()
# 
# plt.plot(MSE, 'y--^')
# plt.xlabel("Step")
# plt.ylabel("MSE")
# plt.legend(['MSE'])
# plt.grid(True)
# plt.title('MSE')
# plt.show()
# 
# =============================================================================
# =============================================================================
# 
# # 실습 8번 해석해와 경사하강법  
# 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# 
# data = pd.read_csv('lin_regression_data_01.csv',names = ['Weight','Height'])
# yhat = []
# get = []
# MSE1 = []
# 
# x = np.asarray(data['Weight'])    #x는 몸무게
# y = np.asarray(data['Height'])        #y는 키
# N = len(data['Height'])
# a = 0.000015                         #학습률 learning rate
# 
# w0 = np.sum(((1/N)*(y)*(x-(1/N)*np.sum(x)))/(((1/N)*np.sum(x**2))-(np.sum(((1/N)*x)))**2))
# w1 = (1/N)*np.sum((y - w0*x))   
#     
# 
# for k in range(N):
#     yhat = w0*(x)+w1
#     get = np.sum((1/N)*((yhat-y)**2)) #최적해 MSE
#     MSE1.append(get)
# 
# G0 = []
# G1 = []
# w0 = [ 1, ]
# w1 = [ 4, ]
# yhat = []
# MSE = []
# K = 50
# 
# #최적해 구한것
# 
# 
# 
# #경사하강법 MSE
# for i in range(K):
#     G0= w0[i]-np.sum(((a*2)/N)*x*((w0[i]*x)+w1[i]-y))  #w0[i+1]
#     w0.append(G0)
#     # print("w0 = ",w0)
#     G1 = w1[i]-np.sum(((a*2)/N)*(x*w0[i]+w1[i]-y))      #w1[i+1]
#     w1.append(G1)
#       
#     yhat = w0[i]*(x)+w1[i]
# 
#     r= np.sum((1/N)*((yhat-y)**2))
#     MSE.append(r)
# 
# 
# plt.plot(MSE1, 'y--s', MSE, 'c:o')
# plt.xlabel("Step")
# plt.ylabel("MSE")
# plt.legend(['MSE(Cost)', 'MSE(GDM)'])
# plt.grid(True)
# plt.title('MSE')
# plt.show()
# 
# =============================================================================
#실습 9번 기저함수 함수를 만들어서 K값을 임의로 지정(random) 해석해 식을 참고할 것
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# data = pd.read_csv('lin_regression_data_01.csv',names = ['Weight','Height'])

# x = np.asarray(data['Weight'])    #x는 몸무게
# y = np.asarray(data['Height'])        #y는 키
# N = len(data['Height'])
# a = 0.000015 
# =============================================================================
# 
# #실습 10번
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# 
# data = pd.read_csv('lin_regression_data_01.csv',names = ['Weight','Height'])
# 
# x = np.asarray(data['Weight'])    #x는 몸무게
# y = np.asarray(data['Height'])        #y는 키
# N = len(data['Height'])
# 
# all_K = [3, 5, 8, 10] #K의 경우를 확인 case 별로 대입하기
# 
# yhat = 0
# avg = []
# o = []
# phi = []
# get = []
# test_get = []
# MSE = []
# MSE1 = []
# w0 = []
# w1 = []
# count = 0
# count_x = 0
# 
# count_y = 0
# 
# 
# 
# x = np.asarray(data['Weight'])    #x는 몸무게
# y = np.asarray(data['Height'])        #y는 키
# N = len(data['Height'])
# 
# 
# w0 = np.sum(((1/N)*(y)*(x-(1/N)*np.sum(x)))/(((1/N)*np.sum(x**2))-(np.sum(((1/N)*x)))**2))
# w1 = (1/N)*np.sum((y - w0*x))
# 
# for k in all_K:
#     
#     K_list = list(range(0,k))
#     for r in K_list:
#         avg.append(np.min(x)+(((np.max(x)-np.min(x))*r)/(k-1)))     #평
# 
#     o=((((np.max(x)-np.min(x))*r)/(k-1)))       #분산
#     
# for c in list(range(0,x.size*k)):
#     
#     if count ==k:
#         count = 0
#         count_x = count_x + 1
#         count_y = count_y + 1
#         
#     phi.append(math.exp((-1/2)*(((x[count_x]-avg[count_x])/o)**2)))
#     test_get_MSE = (1/N)*(np.sum((((phi)-y[count_y])**2)))
#     MSE.append(test_get_MSE)
#     
# 
# 
#     add_one = np.full((x.size,1),1) #맨 마지막 부분에 1을 더해준다
# 
# phi = np.array(phi) 
# phi_real = phi.reshape(x.size,k)
# 
# phi_real = np.append(phi_real, add_one, axis = 1)
# 
# Q = np.linalg.inv(np.transpose(phi_real)@phi_real)@(np.transpose(phi_real)@y)
# 
# print(Q)
# 
# # for t in list(range(0,k)):
#     
# #     yhat = yhat+(Q[t]*(math.exp((-1/2)*(((x-avg[t])/o)**2))))
#     
# #     yhat = yhat+ Q[k]
# 
# yhat = []
# for k in range(N):
#     yhat = w0*(x)+w1
#     get = np.sum((1/N)*((yhat-y)**2)) #최적해 MSE
#     MSE1.append(get)
# 
# #훈련 데이터 출력
# plt.plot(x, y, 'b>')
# plt.xlabel("Weight")
# plt.ylabel("Height")
# plt.legend(['Weight', 'Height'], )
# plt.grid(True)
# plt.title(" Training data Set")
# plt.show()
# 
# # K = 3
# plt.plot( MSE1, 'y:o')
# plt.xlabel("K")
# plt.ylabel("MSE")
# plt.legend(['K'])
# plt.grid(True)
# plt.title('K')
# plt.show()
# 
# =============================================================================
# =============================================================================
#실습 11번 
#평균제곱 오차 출력
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import math

# data = pd.read_csv('lin_regression_data_01.csv',names = ['Weight','Height'])

# x = np.asarray(data['Weight'])    #x는 몸무게
# y = np.asarray(data['Height'])        #y는 키
# N = len(data['Height'])
# a = 0.000015                         #학습률 learning rate


# #K = 3
# plt.plot(MSE1, 'y--s', MSE, 'c:o')
# plt.xlabel("K = 3")
# plt.ylabel("MSE")
# plt.legend([''])
# plt.grid(True)
# plt.title('MSE')
# plt.show()

# #K =5
# plt.plot(MSE1, 'y--s', MSE, 'c:o')
# plt.xlabel("K = 5")
# plt.ylabel("MSE")
# plt.legend([''])
# plt.grid(True)
# plt.title('MSE')
# plt.show()