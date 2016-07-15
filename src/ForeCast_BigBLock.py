#-*- encoding: utf-8 -*-
'''
Created on 2015-10-26

@author: Administrator

'''
import calculateB_BigBlock
from  hmm_BigBlock import *
from config import *
import mysql.connector
from log import *

class Forecast_Big(object):  
    
    
    def getMean(self,blocknum,Observe_block):
        #获取block=num,在时间间隔4的平均值
        sum=0.0
        for i in xrange(len(Observe_block[blocknum])):
            sum+=Observe_block[blocknum][i][3]
        mean_target= sum/len(Observe_block[blocknum])   
        return mean_target 
      
    def forecast_forEachBlock(self,block,obj_list,status_dict,Observe_block,B):
        status_list=[]    
        O=Observe_block[block]
        print "BLOCK %d:" % block,len(O),O
        mean=self.getMean(block,Observe_block) 
        T=len(O[0])
        L=len(O)
        hmm=HMM_one(A1,B,pi)
        alpha=np.zeros((T,hmm.N),np.float)
        beta=np.zeros((T,hmm.N),np.float)
        gamma=np.zeros((T,hmm.N),np.float)
        hmm.BaumWelch(L, T, O, alpha, beta, gamma)
        obj_list[block]=hmm
        for i in range(L):
            res=hmm.viterbi(O[i])
            status_list.append(list(res[0]))
        status_dict[block]=status_list
        
    def forecast_allBlock(self):
        big_obj_list={}
        fore_block={}
        status_dict={}
        Observe_block={}
  
        result=calculateB_BigBlock.getOandB_forBigBlock()
        Observe_block=result[2]  
        list_block=getList_Number()
        print len(list_block),list_block 
        conn = mysql.connector.connect(host='127.0.0.1', database='t_order', user='root', password='1234567')
        crsr = conn.cursor()
           
        for i in list_block:
            B=result[0][i]
            self.forecast_forEachBlock(i,big_obj_list,status_dict,Observe_block,B)
        
            print "status_dict[%d]:"% i,status_dict[i]
            #init db            
            #更新预测的状态结果数据到数据库  
            print "len(status_dict[i]):",len(status_dict[i])
            for j in range(len(status_dict[i])):          
                testsql="insert into   hmm_veterbi_BigBlock  values('%d','%d','%d','%d','%d')" % (i,status_dict[i][j][0],status_dict[i][j][1],status_dict[i][j][2],status_dict[i][j][3])
                crsr.execute(testsql)
                conn.commit()
#             testsql="update hmm_rank1_veterbi set  t_status_1=%d,t_status_2=%d,t_status_3=%d,t_status_4=%d where t_block_number=%d " % (status_dict[i][0],status_dict[i][1],status_dict[i][2],status_dict[i][3],i)
#             crsr.execute(testsql)
#             conn.commit()
        crsr.close()
        conn.close() 
        print "预测状态是： %s" % status_dict  
        logger.info("预测的Big Block的状态字典是： %s",status_dict)
        logger.info("预测的Big Block的马尔可夫模型是是： %d,%s",len(big_obj_list),big_obj_list)
        
   
        
        return fore_block,status_dict,big_obj_list
   
if __name__ == '__main__':
    hmm=Forecast_Big()
#     observe_list={249: [[40, 45, 41, 42], [74, 77, 58, 68], [69, 77, 73, 57], [119, 106, 84, 99], [131, 135, 105, 112], [161, 147, 139, 120], [137, 154, 129, 116], [168, 163, 129, 139], [114, 127, 119, 94], [51, 63, 65, 69], [140, 125, 113, 132], [150, 145, 126, 103], [166, 164, 154, 122], [171, 162, 175, 158], [176, 165, 183, 158], [116, 91, 107, 98], [87, 91, 67, 71], [127, 120, 144, 104], [158, 116, 133, 124], [163, 144, 140, 119], [160, 150, 160, 142], [168, 150, 145, 166], [139, 139, 97, 103], [97, 79, 87, 73], [123, 147, 133, 118], [116, 159, 125, 133], [149, 167, 142, 150], [134, 124, 119, 120], [160, 151, 168, 157], [119, 104, 118, 88], [103, 70, 96, 60], [150, 133, 124, 124], [141, 136, 141, 97], [164, 128, 137, 124], [143, 117, 131, 132], [124, 107, 115, 116], [112, 74, 95, 61], [131, 141, 115, 80], [117, 77, 58, 69], [134, 133, 118, 117], [159, 154, 160, 127], [15, 11, 7, 11], [147, 111, 118, 112], [162, 153, 158, 146], [128, 137, 114, 117], [124, 145, 149, 136], [69, 56, 65, 50], [74, 60, 62, 61], [75, 49, 69, 50], [112, 118, 89, 83], [144, 124, 67, 128], [119, 106, 97, 120], [42, 45, 84, 112], [98, 96, 102, 96], [76, 73, 86, 50], [122, 91, 111, 91], [158, 138, 141, 105], [152, 144, 103, 132], [137, 140, 134, 136], [172, 141, 145, 137], [87, 120, 77, 86], [76, 65, 74, 82], [88, 98, 91, 104], [159, 141, 123, 127], [138, 176, 113, 150], [184, 176, 139, 150], [146, 136, 150, 159], [102, 103, 94, 108], [86, 89, 88, 86], [129, 119, 128, 111], [135, 130, 142, 134], [147, 128, 126, 122], [163, 125, 163, 103], [180, 167, 117, 135], [90, 99, 103, 94], [107, 88, 84, 51], [112, 141, 116, 89], [109, 103, 114, 128], [89, 84, 90, 108], [135, 133, 134, 142], [134, 172, 148, 121], [105, 114, 87, 84], [73, 87, 81, 75], [53, 45, 38, 34], [124, 124, 132, 102], [97, 118, 121, 104], [165, 148, 140, 128]]}
#     forecast=hmm.getMean(249,observe_list)
#     status_dict={}
#     obj_list={}
#     B=[[143.0, 16.0], [87.0, 24.0]]
#     hmm.forecast_forEachBlock(249,obj_list,status_dict,observe_list,B)
    hmm.forecast_allBlock()
    
