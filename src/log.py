#coding:utf-8
'''
Created on 2015-10-22

@author: Administrator

'''

  
import logging
import os,sys
  
# create a logger    
logger = logging.getLogger()  
logger.setLevel(logging.DEBUG)

#create filename
file_name=sys.argv[0][sys.argv[0].rfind(os.sep)+1:-3]+'.log' 

if  os.path.exists(file_name):
    os.remove(file_name)  

# create a handler，used to write log into the  file   
fh = logging.FileHandler(file_name)  
  
# create a handler more ，used to write log into the  console     
ch = logging.StreamHandler()  
  
#definite   handler's formatter    
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')  
fh.setFormatter(formatter)  
ch.setFormatter(formatter)     
logger.addHandler(fh)  
#logger.addHandler(ch)  
