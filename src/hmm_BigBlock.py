# -*- encoding: utf-8 -*-
'''
Created on 2015.10.14

@author: Administrator
'''
import numpy as np
import scipy.stats
from log import *
class HMM_one:
    def __init__(self, Ann, Bnm, pi1n):
        self.A = np.array(Ann)
        self.B = np.array(Bnm)
        self.pi = np.array(pi1n)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]
                
    def BFrequency(self, i, j):
        res = scipy.stats.norm(self.B[i, 0], self.B[i, 1]).pdf(j)
        return res
    
    def Forward(self, T, O, alpha, pprob):
        for i in range(self.N):
            alpha[0, i] = self.pi[i] * self.BFrequency(i, O[0])

        for t in range(T - 1):
            for j in range(self.N):
                sum = 0.0
                for i in range(self.N):
                    sum += alpha[t, i] * self.A[i, j]
                alpha[t + 1, j] = sum * self.BFrequency(j, O[t + 1])

        sum = 0.0
        for i in range(self.N):
            sum += alpha[T - 1, i]
        pprob[0] = sum
    
    def ForwardWithScale(self,T,O,alpha,scale,pprob):
        scale[0] = 0.0
        for i in range(self.N):
            alpha[0, i] = self.pi[i] * self.BFrequency(i, O[0])
            scale[0] += alpha[0,i]
        
        for i in range(self.N):
            alpha[0,i] /= scale[0]
        for t in range(T-1):
            scale[t+1] = 0.0
            for j in range(self.N):
                sum = 0.0
                for i in range(self.N):
                    sum += alpha[t,i]*self.A[i,j]                
                alpha[t + 1, j] = sum * self.BFrequency(j, O[t + 1])
                scale[t+1] += alpha[t+1,j]
            for j in range(self.N):
                alpha[t+1,j] /= scale[t+1]
        for t in range(T):
            pprob[0] += np.log(scale[t])
    
    def Backward(self, T, O, beta, pprob):
        for i in range(self.N):
            beta[T - 1, i] = 1.0

        for t in range(T - 2, -1, -1):
            for i in range(self.N):
                sum = 0.0
                for j in range(self.N):
                    sum += self.A[i, j] * self.BFrequency(j, O[t + 1]) * beta[t + 1, j]
                beta[t, i] = sum                
        pprob[0] = 0.0
        for i in range(self.N):
            pprob[0] += self.pi[i] * self.BFrequency(i, O[0]) * beta[0, i]
    
    def BackwardWithScale(self,T,O,beta,scale):
        for i in range(self.N):
            beta[T-1,i] = 1.0/scale[T-1]
    
        for t in range(T-2,-1,-1):
            for i in range(self.N):
                sum = 0.0
                for j in range(self.N):
                    sum += self.A[i, j] * self.BFrequency(j, O[t + 1]) * beta[t + 1, j]
                beta[t,i] = sum / scale[t]
    
    # Viterbi
    def viterbi(self, O):
        T = len(O)
       
        delta = np.zeros((T, self.N), np.float)  
        phi = np.zeros((T, self.N), np.float)  
        I = np.zeros(T)
        for i in range(self.N):  
            delta[0, i] = self.pi[i] * self.BFrequency(i, O[0])  
            phi[0, i] = 0
       
        for t in range(1, T):  
            for i in range(self.N):                                  
                delta[t, i] = self.BFrequency(i, O[t]) * np.array([delta[t - 1, j] * self.A[j, i]  for j in range(self.N)]).max()
                phi[t, i] = np.array([delta[t - 1, j] * self.A[j, i]  for j in range(self.N)]).argmax()
       
        prob = delta[T - 1, :].max()  
        I[T - 1] = delta[T - 1, :].argmax()
         
        for t in range(T - 2, -1, -1): 
            I[t] = phi[t + 1, I[t + 1]]
        return I, prob
    
  
    def ComputeGamma(self, T, alpha, beta, gamma):
        for t in range(T):
            denominator = 0.0
            for j in range(self.N):
                gamma[t, j] = alpha[t, j] * beta[t, j]
                denominator += gamma[t, j]
            for i in range(self.N):
                gamma[t, i] = gamma[t, i] / denominator
    
    
    def ComputeXi(self, T, O, alpha, beta, gamma, xi):
        for t in range(T - 1):
            sum = 0.0
            for i in range(self.N):
                for j in range(self.N):
                    xi[t, i, j] = alpha[t, i] * beta[t + 1, j] * self.A[i, j] * self.BFrequency(j, O[t + 1])
                    sum += xi[t, i, j]
            for i in range(self.N):
                for j in range(self.N):
                    xi[t, i, j] /= sum
                    
    # Baum-Welch

    def BaumWelch(self, L, T, O, alpha, beta, gamma):
        DELTA = 0.01 ; round = 0 ;  probf = [0.0]
        delta = 0.0 ; deltaprev = 10e-80 ; probprev = 0.0 ; ratio = 1.0 ;
      
        xi = np.zeros((T, self.N, self.N))
        pi = np.zeros((T),np.float)
        denominatorA = np.zeros((self.N), np.float)
        denominatorB = np.zeros((self.N), np.float)
        numeratorA = np.zeros((self.N, self.N), np.float)
        scale = np.zeros((T),np.float)
        while True :
            probf[0] = 0
            # E - step
            for l in range(L):
#                 self.Forward(T, O[l], alpha, probf)
#                 self.Backward(T, O[l], beta, probf)
                self.ForwardWithScale(T, O[l], alpha, scale, probf)
                self.BackwardWithScale(T, O[l], beta, scale)
                self.ComputeGamma(T, alpha, beta, gamma)
                self.ComputeXi(T, O[l], alpha, beta, gamma, xi)
                for i in range(self.N):
                    pi[i] += gamma[0,i]
                    for t in range(T - 1): 
                        denominatorA[i] += gamma[t, i]
                        denominatorB[i] += gamma[t, i]
                    denominatorB[i] += gamma[T - 1, i]
                    
                    for j in range(self.N):
                        for t in range(T - 1):
                            numeratorA[i, j] += xi[t, i, j]
                            
            # M - step

            for i in range(self.N):
                self.pi[i] = 0.001/self.N + 0.999*pi[i]/L
                for j in range(self.N):
                    #第一种迭代方法
#                     self.A[i, j] = numeratorA[i, j] / denominatorA[i]
                    #第二种迭代方法
                    self.A[i,j] = 0.001/self.N + 0.999*numeratorA[i,j]/denominatorA[i]
                    numeratorA[i, j] = 0.0
                                    
                pi[i]= denominatorA[i] = denominatorB[i] = 0.0;
            round += 1
            
            if round == 1:
                probprev = probf[0]
                round += 1
                continue           
            delta = probf[0] - probprev
            ratio = delta / deltaprev
#             ratio = delta / probprev
            probprev = probf[0]
            deltaprev = delta
#             print "probf:" ,probf[0]
            logger.info("probf: %f" , probf[0])
            
            round += 1
#             print "delta:",delta
#             print "rati0：",ratio
            if delta < 0 or ratio <= 0.01 or round>=20:
                print "self.A:" , self.A
                logger.info("self.A is %s",self.A)
                print "self.pi", self.pi
                print "num iteration ", round
                break
    def calculat_forecast(self, i):
        forecast = 0
        for k in range(self.N):
            forecast += self.A[i, k] * self.B[k, 0]
        return round(forecast)
         

if __name__ == "__main__":
    # block 249  
    pi = [0.5, 0.5]
    A = [[0.5, 0.5], [0.5, 0.5]]
    B = [[16.0, 4.0], [7.0, 4.0]]
    O = [[9, 15, 7, 3], [7, 8, 23, 17], [14, 16, 19, 11], [17, 22, 16, 11]]
    T = len(O[0])
    L = len(O)
    hmm = HMM_one(A, B, pi)
    alpha = np.zeros((T, hmm.N), np.float)
    beta = np.zeros((T, hmm.N), np.float)
    gamma = np.zeros((T, hmm.N), np.float)
    hmm.BaumWelch(L, T, O, alpha, beta, gamma)
    res = hmm.viterbi(O[1])
    print res
    real = O[1][3]
    forecast = hmm.calculat_forecast(int(res[0][-2]))
    print "Forecast_forEachBlock_Rank2 is %d ,mean is 11.6,real is %d" % (forecast, real)
    error = float(abs(forecast - real)) / float(real)
    error2 = abs(11.6 - real) / float(real)
    print "预测 偏差是： ", error
    print "与平均值的偏差：", error2

