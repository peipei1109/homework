#-*- encoding: utf-8 -*-
'''
Created on 2015-10-26

@author: Administrator

'''


#前后向算法加上scale参数

import numpy as np
import calculateB_SmallBlock_v1
import scipy.stats
import math
from log import *
class HMM_two_red:
    def __init__(self, Ann, Bnm,Cnm,pi1n):
        self.A = np.array(Ann)
        self.B = np.array(Bnm)
        self.C = np.array(Cnm)
        self.pi = np.array(pi1n)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]        
    
    def BFrequency(self, i,j):
        #用该点的概率密度来极端
        res =scipy.stats.norm(self.B[i,0],self.B[i,1]).pdf(j)
#         #用该点一个区间的积分密度来计算
#         res =scipy.stats.norm(self.B[i,0],self.B[i,1]).cdf(j+1)-scipy.stats.norm(self.B[i,0],self.B[i,1]).cdf(j)
        return res


    def Forward(self,T,O,alpha,pprob):
        for i in range(self.N):
            for j in range(self.N):
                results1=scipy.stats.norm(self.B[i,0],self.B[i,1]).pdf(O[0]) #pdf 函数可以出传入一个list，返回的结果也是一个list
                results2=self.BFrequency(j,O[1])
                alpha[0,i,j] = self.pi[i]*results1*self.C[i,j]*results2
                
        for t in range(T-2):
            for i in range(self.N):
                for j in range(self.N):
                    sum = 0.0
                    for k in range(self.N):                        
                        sum += alpha[t,k,i]*self.A[k,i,j]
                    result3=self.BFrequency(j,O[t+2])
                    alpha[t+1,i,j] =sum*result3
                    
        sum = 0.0
        for i in range(self.N):
            for j in range(self.N):                
                sum += alpha[T-2,i,j]
        pprob[0] = sum
        
    def forward_with_scale(self, T,O,alpha,scale,pprob):  
        for i in range(self.N):
            for j in range(self.N):
                results1=scipy.stats.norm(self.B[i,0],self.B[i,1]).pdf(O[0]) #pdf 函数可以出传入一个list，返回的结果也是一个list
                results2=self.BFrequency(j,O[1])
                alpha[0,i,j] = self.pi[i]*results1*self.C[i,j]*results2
        scale[0] = np.sum(alpha[0,:,:])
        alpha[0,:,:] /= scale[0]
        for t in range(T-2):
            for i in range(self.N):
                for j in range(self.N):
                    sum = 0.0
                    for k in range(self.N):                        
                        sum += alpha[t,k,i]*self.A[k,i,j]
                    result3=self.BFrequency(j,O[t+2])
                    alpha[t+1,i,j] =sum*result3
            scale[t+1] = np.sum(alpha[t+1,:,:])
            alpha[t+1,:,:] /= scale[t+1]
                    
#         sum = 0.0
#         for i in range(self.N):
#             for j in range(self.N):                
#                 sum += alpha[T-2,i,j]
#         pprob[0] = sum 
        pprob[0] = np.sum(np.log(scale[:]))

    def Backward(self,T,O,beta):
        for i in range(self.N):
            for j in range(self.N):
                beta[T-2,i,j] = 1.0

        for t in range(T-3,-1,-1):
            for i in range(self.N):
                for j in range(self.N):
                    sum=0.0
                    for k in range(self.N):
                        sum += self.A[i,j,k]*self.BFrequency(k,O[t+2])*beta[t+1,j,k]
                    beta[t,i,j] = sum
                    
    def backward_with_scale(self, T,O,beta, scale):
        for i in range(self.N):
            for j in range(self.N):
                beta[T-2,i,j] = 1.0/scale[T-2]

        for t in range(T-3,-1,-1):
            for i in range(self.N):
                for j in range(self.N):
                    sum=0.0
                    for k in range(self.N):
                        sum += self.A[i,j,k]*self.BFrequency(k,O[t+2])*beta[t+1,j,k]
                    beta[t,i,j] = sum/scale[t]
                

#         pprob[0] = 0.0
#         for i in range(self.N):
#             for j in  range(self.N):
#                 pprob[0] += self.pi[i]*self.BFrequency(i,O[0])*self.C[i,j]*self.BFrequency(j,O[1])*beta[0,i,j]
                
    
    # Viterbi
    def viterbi(self,O):
        T = len(O)       
        delta = np.zeros((T-1,self.N,self.N),np.float)  
        phi = np.zeros((T-1,self.N,self.N),np.float)  
        I = np.zeros(T)
        for i in range(self.N):
            for j in range(self.N): 
                delta[0,i,j] = self.pi[i]*self.BFrequency(i,O[0])*self.C[i,j]*self.BFrequency(j,O[1]) 
                phi[0,i,j] = 0
       
        for t in range(1,T-1):  
            for i in range(self.N): 
                for j in range(self.N):                                 
                    delta[t,i,j] = self.BFrequency(j,O[t+1])*np.array([delta[t-1,k,i]*self.A[k,i,j]  for k in range(self.N)]).max()
                    phi[t,i,j] = np.array([delta[t-1,k,i]*self.A[k,i,j]  for k in range(self.N)]).argmax()
        prob = delta[T-2,:,:].max()                    
        index=delta[T-2,:,:].argmax() #得到的是第几个元素，比如返回的是3 ，表明最大的那个元素是delta【T:1：1】,
        I[T-1]=index/self.N
        I[T-2]=index%self.N
         
        for t in range(T-3,-1,-1): 
            I[t] = phi[t+1,I[t+1],I[t+2]]
        return I,prob
    
    #给定参数r和观察序列O的情况下，t=i,t+1=j的概率
    def ComputeGamma(self, T, alpha, beta, gamma):
        for t in range(T-1):
            denominator = 0.0  #P(o|r)
            for  i in range(self.N):
                for j in range(self.N):
                    gamma[t,i,j] = alpha[t,i,j]*beta[t,i,j]
#                     print alpha[t,i,j],beta[t,i,j],gamma[t,i,j]
                    denominator += gamma[t,i,j]
            for  i in range(self.N):
                for j in range(self.N):
#                     print "denominator:",denominator
                    gamma[t,i,j]=gamma[t,i,j]/denominator
    
    #给定参数r和观测序列0的情况下，t=i,t+1=j，t+2=k的概率
    def ComputeXi(self,T,O,alpha,beta,xi):
        for t in range(T-2):
            sum = 0.0 #P(o|r)
            for i in range(self.N):
                for j in range(self.N):
                    for k in range(self.N):
                            xi[t,i,j,k] = alpha[t,i,j]*beta[t+1,j,k]*self.A[i,j,k]*self.BFrequency(k,O[t+2])
                            sum += xi[t,i,j,k]
            for i in range(self.N):
                for j in range(self.N):
                    for k in range(self.N):
                        xi[t,i,j,k] /= sum
                    
    # Baum-Welch

    def BaumWelch(self,L,T,O,alpha,beta,gamma):
        DELTA = 0.01 ; round = 0 ;  probf = [0.0]
        delta = 0.0 ; deltaprev = 10e-80 ; probprev = 0.0 ; ratio = 1.0 ;
        
        xi = np.zeros((T-2,self.N,self.N,self.N))
        pi = np.zeros((self.N),np.float)
        C=np.zeros((self.N,self.N),np.float)
        denominatorA = np.zeros((self.N,self.N),np.float)
        denominatorB = np.zeros((self.N,self.N),np.float)
        numeratorA = np.zeros((self.N,self.N,self.N),np.float)
        scale=np.zeros((T-1),np.float)
        
        
        while True :
            probf[0] = 0
            # E - step
            for l in range(L):
                self.forward_with_scale(T,O[l],alpha,scale,probf)
                self.backward_with_scale(T,O[l],beta,scale)
                self.ComputeGamma(T,alpha,beta,gamma)
                self.ComputeXi(T,O[l],alpha,beta,xi)
                for i in range(self.N):
                    pi[i]=0
                    for j in range(self.N):
                        pi[i] += gamma[0,i,j]
                        for t in range(T-2): 
                            denominatorA[i,j] += gamma[t,i,j] #在观测O下，状态i，j 的转移的期望                            
                            denominatorB[i,j] += gamma[t,i,j]
                        denominatorB[i,j] += gamma[T-2,i,j]   #在观测O下，状态i，j 出现的期望
                        for k in range(self.N):
                            for t in range(T-2):
                                numeratorA[i,j,k] += xi[t,i,j,k]     #在观测0下，状态i,j,k 出现的期望    
                            
            # M - step

            for i in range(self.N):
                self.pi[i] = pi[i]
                for j in range(self.N):
#                     self.C[i,j]=C[i,j]
                    for k in range(self.N):
                        self.A[i,j,k] = numeratorA[i,j,k]/denominatorA[i,j]
                        numeratorA[i,j,k] = 0.0
                
                    denominatorA[i,j]=denominatorB[i,j]=0.0;            
            if round == 0:
                probprev = probf[0]
                round += 1
                continue           
            delta = probf[0] - probprev
            ratio = delta / deltaprev
#             ratio = delta / probprev
            probprev = probf[0]
            deltaprev = delta

            
            round += 1
            
            if  ratio<=0.01 or round>=10:
                print "self.A:" ,self.A
                print "self.pi",self.pi
                print "num iteration ",round
                break
            return self.A
    def calculat_forecast(self,i,j):
        forecast=0
        for k in range(self.N):
            forecast+=self.A[i,j,k]*self.B[k,0]
        return int(round(forecast))

class HMM_two_green:
    def __init__(self, Ann, Bnm,Cnm,pi1n):
        self.A = np.array(Ann)
        self.B = np.array(Bnm)
        self.C = np.array(Cnm)
        self.pi = np.array(pi1n)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]        
    
    def BFrequency(self, i,j):
        #用该点的概率密度来极端
        res =scipy.stats.norm(self.B[i,0],self.B[i,1]).pdf(j)
#         #用该点一个区间的积分密度来计算
#         res =scipy.stats.norm(self.B[i,0],self.B[i,1]).cdf(j+1)-scipy.stats.norm(self.B[i,0],self.B[i,1]).cdf(j)
        return res


    def Forward(self,T,O,alpha,pprob):
        for i in range(self.N):
            for j in range(self.N):
                results1=scipy.stats.norm(self.B[i,0],self.B[i,1]).pdf(O[0]) #pdf 函数可以出传入一个list，返回的结果也是一个list
                results2=self.BFrequency(j,O[1])
                alpha[0,i,j] = self.pi[i]*results1*self.C[i,j]*results2
                
        for t in range(T-2):
            for i in range(self.N):
                for j in range(self.N):
                    sum = 0.0
                    for k in range(self.N):                        
                        sum += alpha[t,k,i]*self.A[k,i,j]
                    result3=self.BFrequency(j,O[t+2])
                    alpha[t+1,i,j] =sum*result3
                    
        sum = 0.0
        for i in range(self.N):
            for j in range(self.N):                
                sum += alpha[T-2,i,j]
        pprob[0] = sum
        
    def forward_with_scale(self, T,O,alpha,scale,pprob):  
        for i in range(self.N):
            for j in range(self.N):
                results1=scipy.stats.norm(self.B[i,0],self.B[i,1]).pdf(O[0]) #pdf 函数可以出传入一个list，返回的结果也是一个list
                results2=self.BFrequency(j,O[1])
                alpha[0,i,j] = self.pi[i]*results1*self.C[i,j]*results2
        scale[0] = np.sum(alpha[0,:,:])
        alpha[0,:,:] /= scale[0]
        for t in range(T-2):
            for i in range(self.N):
                for j in range(self.N):
                    sum = 0.0
                    for k in range(self.N):                        
                        sum += alpha[t,k,i]*self.A[k,i,j]
                    result3=self.BFrequency(j,O[t+2])
                    alpha[t+1,i,j] =sum*result3
            scale[t+1] = np.sum(alpha[t+1,:,:])
            alpha[t+1,:,:] /= scale[t+1]
                    
#         sum = 0.0
#         for i in range(self.N):
#             for j in range(self.N):                
#                 sum += alpha[T-2,i,j]
#         pprob[0] = sum 
        pprob[0] = np.sum(np.log(scale[:]))

    def Backward(self,T,O,beta):
        for i in range(self.N):
            for j in range(self.N):
                beta[T-2,i,j] = 1.0

        for t in range(T-3,-1,-1):
            for i in range(self.N):
                for j in range(self.N):
                    sum=0.0
                    for k in range(self.N):
                        sum += self.A[i,j,k]*self.BFrequency(k,O[t+2])*beta[t+1,j,k]
                    beta[t,i,j] = sum
                    
    def backward_with_scale(self, T,O,beta, scale):
        for i in range(self.N):
            for j in range(self.N):
                beta[T-2,i,j] = 1.0/scale[T-2]

        for t in range(T-3,-1,-1):
            for i in range(self.N):
                for j in range(self.N):
                    sum=0.0
                    for k in range(self.N):
                        sum += self.A[i,j,k]*self.BFrequency(k,O[t+2])*beta[t+1,j,k]
                    beta[t,i,j] = sum/scale[t]
                

#         pprob[0] = 0.0
#         for i in range(self.N):
#             for j in  range(self.N):
#                 pprob[0] += self.pi[i]*self.BFrequency(i,O[0])*self.C[i,j]*self.BFrequency(j,O[1])*beta[0,i,j]
                
    
    # Viterbi
    def viterbi(self,O):
        T = len(O)       
        delta = np.zeros((T-1,self.N,self.N),np.float)  
        phi = np.zeros((T-1,self.N,self.N),np.float)  
        I = np.zeros(T)
        for i in range(self.N):
            for j in range(self.N): 
                delta[0,i,j] = self.pi[i]*self.BFrequency(i,O[0])*self.C[i,j]*self.BFrequency(j,O[1]) 
                phi[0,i,j] = 0
       
        for t in range(1,T-1):  
            for i in range(self.N): 
                for j in range(self.N):                                 
                    delta[t,i,j] = self.BFrequency(j,O[t+1])*np.array([delta[t-1,k,i]*self.A[k,i,j]  for k in range(self.N)]).max()
                    phi[t,i,j] = np.array([delta[t-1,k,i]*self.A[k,i,j]  for k in range(self.N)]).argmax()
        prob = delta[T-2,:,:].max()                    
        index=delta[T-2,:,:].argmax() #得到的是第几个元素，比如返回的是3 ，表明最大的那个元素是delta【T:1：1】,
        I[T-1]=index/self.N
        I[T-2]=index%self.N
         
        for t in range(T-3,-1,-1): 
            I[t] = phi[t+1,I[t+1],I[t+2]]
        return I,prob
    
    #给定参数r和观察序列O的情况下，t=i,t+1=j的概率
    def ComputeGamma(self, T, alpha, beta, gamma):
        for t in range(T-1):
            denominator = 0.0  #P(o|r)
            for  i in range(self.N):
                for j in range(self.N):
                    gamma[t,i,j] = alpha[t,i,j]*beta[t,i,j]
#                     print alpha[t,i,j],beta[t,i,j],gamma[t,i,j]
                    denominator += gamma[t,i,j]
            for  i in range(self.N):
                for j in range(self.N):
#                     print "denominator:",denominator
                    gamma[t,i,j]=gamma[t,i,j]/denominator
    
    #给定参数r和观测序列0的情况下，t=i,t+1=j，t+2=k的概率
    def ComputeXi(self,T,O,alpha,beta,xi):
        for t in range(T-2):
            sum = 0.0 #P(o|r)
            for i in range(self.N):
                for j in range(self.N):
                    for k in range(self.N):
                            xi[t,i,j,k] = alpha[t,i,j]*beta[t+1,j,k]*self.A[i,j,k]*self.BFrequency(k,O[t+2])
                            sum += xi[t,i,j,k]
            for i in range(self.N):
                for j in range(self.N):
                    for k in range(self.N):
                        xi[t,i,j,k] /= sum
                    
    # Baum-Welch

    def BaumWelch(self,L,T,O,alpha,beta,gamma):
        DELTA = 0.01 ; round = 0 ;  probf = [0.0]
        delta = 0.0 ; deltaprev = 10e-80 ; probprev = 0.0 ; ratio = 1.0 ;
        
        xi = np.zeros((T-2,self.N,self.N,self.N))
        pi = np.zeros((self.N),np.float)
        C=np.zeros((self.N,self.N),np.float)
        denominatorA = np.zeros((self.N,self.N),np.float)
        denominatorB = np.zeros((self.N,self.N),np.float)
        numeratorA = np.zeros((self.N,self.N,self.N),np.float)
        scale=np.zeros((T-1),np.float)
        
        
        while True :
            probf[0] = 0
            # E - step
            for l in range(L):
                self.forward_with_scale(T,O[l],alpha,scale,probf)
                self.backward_with_scale(T,O[l],beta,scale)
                self.ComputeGamma(T,alpha,beta,gamma)
                self.ComputeXi(T,O[l],alpha,beta,xi)
                for i in range(self.N):
                    pi[i]=0
                    for j in range(self.N):
                        pi[i] += gamma[0,i,j]
                        for t in range(T-2): 
                            denominatorA[i,j] += gamma[t,i,j] #在观测O下，状态i，j 的转移的期望                            
                            denominatorB[i,j] += gamma[t,i,j]
                        denominatorB[i,j] += gamma[T-2,i,j]   #在观测O下，状态i，j 出现的期望
                        for k in range(self.N):
                            for t in range(T-2):
                                numeratorA[i,j,k] += xi[t,i,j,k]     #在观测0下，状态i,j,k 出现的期望    
                            
            # M - step

            for i in range(self.N):
                self.pi[i] = pi[i]
                for j in range(self.N):
#                     self.C[i,j]=C[i,j]
                    for k in range(self.N):
                        self.A[i,j,k] = numeratorA[i,j,k]/denominatorA[i,j]
                        numeratorA[i,j,k] = 0.0
                
                    denominatorA[i,j]=denominatorB[i,j]=0.0;            
            if round == 0:
                probprev = probf[0]
                round += 1
                continue           
            delta = probf[0] - probprev
            ratio = delta / deltaprev
#             ratio = delta / probprev
            probprev = probf[0]
            deltaprev = delta

            
            round += 1
            
            if  ratio<=0.01 or round>=10:
                print "self.A:" ,self.A
                print "self.pi",self.pi
                print "num iteration ",round
                break
            return self.A
    def calculat_forecast(self,i,j):
        forecast=0
        for k in range(self.N):
            forecast+=self.A[i,j,k]*self.B[k,0]
        return int(round(forecast))
            

if __name__ == "__main__": 
    #block 249  
    pi=[0.5,0.5]
    C=[[0.5,0.5],[0.5,0.5]]
    A=[[[0.5,0.5],[0.5,0.5]],[[0.5,0.5],[0.5,0.5]]]
    B=[[16.0, 4.0], [7.0, 4.0]]
    O=[[9,15,7,3],[7,8,23,17],[14,16,19,11],[17,22,16,11]]
    T=len(O[0])
    L=len(O)
    hmm1=HMM_two_red(A,B,C,pi)
    hmm2=HMM_two_red(A,B,C,pi)
    alpha=np.zeros((T-1,hmm1.N,hmm1.N),np.float)
    beta=np.zeros((T-1,hmm1.N,hmm1.N),np.float)
    gamma=np.zeros((T-1,hmm1.N,hmm1.N),np.float)
    hmm1.D=  hmm1.BaumWelch(L, T, O, alpha, beta, gamma)
    print hmm1.D
    res=hmm1.viterbi(O[1])
    real=O[1][3]
    forecast=hmm1.calculat_forecast(int(res[0][-3]), int(res[0][-2]))
    print "Forecast_forEachBlock_Rank2 is %d ,mean is 11.6,real is %d" % (forecast,real)
    error=float(abs(forecast-real))/float(real)
    error2=abs(11.6-real)/float(real)
    print "预测 偏差是： ",error
    print "与平均值的偏差：",error2

