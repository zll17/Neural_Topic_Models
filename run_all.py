import os
import subprocess

EPOCH = 200
task = {#'dadoc':{'n_topics':[5,8,10,15,20],'epoch':50,'gpu':'0'},\
        #'daline':{'n_topics':[5,8,10,15,20],'epoch':50,'gpu':'1'},\
        'sub':{'n_topics':list(range(20,60,10)),'epoch':EPOCH,'gpu':'1'}#,\
        #'subX':{'n_topics':list(range(20,30,10)),'epoch':EPOCH,'gpu':'1'},\
        #'sohu100k':{'n_topics':list(range(20,30,10)),'epoch':EPOCH,'gpu':'0'},\
        #'weibo':{'n_topics':list(range(20,30,10)),'epoch':EPOCH,'gpu':'0'}#,\
        #'sogou':{'n_topics':[10,20,30,40,50],'epoch':15},'gpu':'1'}
        }

for taskname,config in task.items():
    epoch = config['epoch']
    gpu = config['gpu']
    for n_topics in config['n_topics']:
        comp_code1 = subprocess.run('python NTM_run.py --taskname {} --n_topic {} --num_epochs {}'.format(taskname,n_topics,epoch).split(' ')) 
        print('Train done for {}, n_topics:{}, epoch:{}'.format(taskname,n_topics,epoch))
        print('='*30)
