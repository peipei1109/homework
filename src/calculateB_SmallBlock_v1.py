#-*-coding:UTF-8-*-
'''
Created on 2015-10-20
@author: Administrator
'''
import sys
import scipy.stats
import log
from config import *
import mysql.connector 


#init db

#query
def getO_forSmallBlock(blocknum): 
    conn = mysql.connector.connect(host='127.0.0.1', database='t_order', user='root', password='1234567')
    crsr = conn.cursor()   
    testsql="select  time_lag_1,time_lag_2,time_lag_3,time_lag_4  from t_date_block_interval_distribution where t_order_block_number=%d and t_order_create_date>='2015-05-01'and t_order_create_date<='2015-07-31'"  %(blocknum)
    
    crsr.execute(testsql)
    result = crsr.fetchall()
    for i in range(len(result)):
        result[i]=list(result[i])
        result[i][0]=int(result[i][0])
        result[i][1]=int(result[i][1])
        result[i][2]=int(result[i][2])
        result[i][3]=int(result[i][3]) 
    crsr.close()
    conn.close()
    return result
    

def calculateBandO_forSmallBlock(Observe_block,blocknum):

    result_list=[]
    green_list=[]
    red_list=[]
    mean_list=[]
    doublemean_list=[]
    observe_list=getO_forSmallBlock(blocknum)
    Observe_block[blocknum]=observe_list
    for i in range(len(observe_list)):       
        result_list+=observe_list[i]
    result_list.sort()
    avg_green=0
    avg_red=0
    avg_rr=0
    avg_rg=0
    avg_gr=0
    avg_gg=0
    mid=len(result_list)/2
    mid1=len(result_list)/4
    green_list=result_list[:mid]
    red_list=result_list[mid:]
    for i in range(0,mid1):
        avg_gg=avg_gg+result_list[i]       
    avg_gg=round(float(avg_gg)/(len(result_list)/4),2)
    for i in range(mid1,mid):
        avg_gr=avg_gr+result_list[i]       
    avg_gr=round(float(avg_gr)/(len(result_list)/4),2)
    for i in range(mid,mid1+mid):
        avg_rg=avg_rg+result_list[i]       
    avg_rg=round(float(avg_rg)/(len(result_list)/4),2)
    for i in range(mid1+mid,len(result_list)):
        avg_rr=avg_rr+result_list[i]       
    avg_rr=round(float(avg_rr)/(len(result_list)/4),2)
    doublemean_list.append(avg_rr)
    doublemean_list.append(avg_rg)
    doublemean_list.append(avg_gr)
    doublemean_list.append(avg_gg)
    
    for i in range(0,mid):
        avg_green=avg_green+result_list[i]
    avg_green=round(float(avg_green)/(len(result_list)/2),2)
    for j in range(mid,len(result_list)):
        avg_red=avg_red+result_list[j]
    avg_red=round(float(avg_red)/(len(result_list)-len(result_list)/2),2)  
    mean_list.append(avg_red)
    mean_list.append(avg_green)
    log.logger.info("block %d 计算出来的红绿状态的平均值为：%s",blocknum,mean_list)
    keylist=["green","red"]
    valuelist=[green_list,red_list]
    resDict=dict(zip(keylist,valuelist))
    #calculate mean ,sigma
    #得到序列期望值和标准差
    res1=list(scipy.stats.norm.fit(resDict["red"])) 
    res2=list(scipy.stats.norm.fit(resDict["green"]))
    #用拟合后的平均值
    for i in range(2):
        res1[i]=round(res1[i])
        res2[i]=round(res2[i])
    listB=[res1,res2]
    log.logger.info("block %d 拟合出来的B为:%s",blocknum,listB)

    return listB,mean_list,doublemean_list
  
def getOandB_forSmallBlock():
    list_block=[]
    Dict_block={}
    Mean_block={}
    doublemean_block={}
    Observe_block={}
    list_block=getList_Number()
    print list_block
    for i in list_block:
        res=calculateBandO_forSmallBlock(Observe_block,i)
        Dict_block[i]=res[0]
        Mean_block[i]=res[1]
        doublemean_block[i]=res[2]
         
    log.logger.info("得到的各block的拟合的B的字典为：%s",Dict_block)
    log.logger.info("得到的各block的红绿状态的字典为：%s",Mean_block)  
    log.logger.info("得到的各block的观测的字典为：%s",Observe_block)
    log.logger.info("得到的各block的四个组合状态的字典为：%s",doublemean_block)    
    return  Dict_block ,Mean_block ,Observe_block,doublemean_block

if __name__ == '__main__':
    result=getOandB_forSmallBlock()
    print result[0]
    print result[1]
    print result[2]
    print result[3]
    

    
