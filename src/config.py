#-*- encoding: utf-8 -*-

import mysql.connector  
import numpy as np
import matplotlib.pyplot as plt
from  log import *
import datetime

#初始化pi，C,A
#红绿状态的分布
pi=[0.5,0.5]
#大block的A
A1=[[0.5,0.5],[0.5,0.5]]
#小block的A
A=[[[0.5,0.5],[0.5,0.5]],[[0.5,0.5],[0.5,0.5]]]
C = [[0.5, 0.5], [0.5, 0.5]]

#A=[[[0.5,0.5],[0.5,0.5]],[[0.5,0.5],[0.5,0.5]]]


#list_block,最后会进行预测的块

def getList_Number():
    list_block=[]
    conn = mysql.connector.connect(host='127.0.0.1', database='t_order', user='root', password='1234567')
    crsr = conn.cursor()
    testsql="select * from block_number_forecast"
    crsr.execute(testsql)
    fname = crsr.fetchall()
    for i in range(len(fname)):
        list_block.append(fname[i][0]) 
    logger.info("list_block: %s ",list_block)
    crsr.close()  
    conn.close()
    return list_block

if __name__ == '__main__':
    block_list=getList_Number()
    print block_list
    