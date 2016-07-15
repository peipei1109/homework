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
def getO_forBigBlock(blocknum): 
    conn = mysql.connector.connect(host='127.0.0.1', database='t_order', user='root', password='1234567')
    crsr = conn.cursor()   
    testsql="select  sum(time_lag_1),sum(time_lag_2),sum(time_lag_3),sum(time_lag_4)  from (select  *  from t_date_block_interval_distribution where t_order_block_number= %d and t_order_create_date>='2015-05-01'and t_order_create_date<='2015-07-31' UNION select  *  from t_date_block_interval_distribution where t_order_block_number=%d and t_order_create_date>='2015-05-01'and t_order_create_date<='2015-07-31' UNION select  *  from t_date_block_interval_distribution where t_order_block_number=%d and t_order_create_date>='2015-05-01'and t_order_create_date<='2015-07-31' UNION select  * from t_date_block_interval_distribution where t_order_block_number=%d and t_order_create_date>='2015-05-01'and t_order_create_date<='2015-07-31' UNION select  * from t_date_block_interval_distribution where t_order_block_number=%d and t_order_create_date>='2015-05-01'and t_order_create_date<='2015-07-31' UNION select *  from t_date_block_interval_distribution where t_order_block_number=%d and t_order_create_date>='2015-05-01'and t_order_create_date<='2015-07-31' UNION select * from t_date_block_interval_distribution where t_order_block_number=%d and t_order_create_date>='2015-05-01'and t_order_create_date<='2015-07-31' UNION select  * from t_date_block_interval_distribution where t_order_block_number=%d and t_order_create_date>='2015-05-01'and t_order_create_date<='2015-07-31'UNION select  * from t_date_block_interval_distribution where t_order_block_number=%d and t_order_create_date>='2015-05-01'and t_order_create_date<='2015-07-31')sz1 group by t_order_create_date"  %(blocknum+19,blocknum+20,blocknum+21,blocknum-1,blocknum,blocknum+1,blocknum-21,blocknum-20,blocknum-19)
    
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
    

def calculateBandO_forBigBlock(Observe_block,blocknum):
    area_block=[blocknum+19,blocknum+20,blocknum+21,blocknum-1,blocknum,blocknum+1,blocknum-21,blocknum-20,blocknum-19]

    result_list=[]
    green_list=[]
    red_list=[]
    mean_list=[]
    observe_list=getO_forBigBlock(blocknum)
    Observe_block[blocknum]=observe_list
    for i in range(len(observe_list)):       
        result_list+=observe_list[i]
    result_list.sort()

    avg_green=0
    avg_red=0
    mid=len(result_list)/2
    green_list=result_list[:mid]
    red_list=result_list[mid:]
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
#     #只用计算出来的平均值，不用拟合的平均值，但保留拟合的标准差
#     res1[0]=avg_red
#     res1[1]=round(res1[1])
#     res2[0]=avg_green
#     res2[1]=round(res2[1])
    #用拟合后的平均值
    for i in range(2):
        res1[i]=round(res1[i])
        res2[i]=round(res2[i])
    listB=[res1,res2]
    log.logger.info("block %d 拟合出来的B为:%s",blocknum,listB)

    return listB,mean_list
  
def getOandB_forBigBlock():
    list_block=[]
    Dict_block={}
    Mean_block={}
    Observe_block={}
    list_block=getList_Number()
    print list_block
    for i in list_block:
        res=calculateBandO_forBigBlock(Observe_block,i)
        Dict_block[i]=res[0]
        Mean_block[i]=res[1]
         
    log.logger.info("得到的各block的拟合的B的字典为：%s",Dict_block)
    log.logger.info("得到的各block的红绿状态的字典为：%s",Mean_block)  
    log.logger.info("得到的各block的观测的字典为：%s",Observe_block)   
    return  Dict_block ,Mean_block ,Observe_block

if __name__ == '__main__':
    result=getOandB_forBigBlock()
    print result[0]
    print result[1]
    print result[2]
    print len(result[2][267])

    
