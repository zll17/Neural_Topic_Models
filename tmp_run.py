import os

n_topics = 10
taskname = 'weibo'
gpu = '0'
epoch = 10

os.system('python main.py --mode train --dataset {} --data_path data/{} --num_topics {} --train_embeddings 0 --epochs {} --emb_path data/{}/embeddings.txt --gpu {}'.format(taskname,taskname,n_topics,epoch,taskname,gpu)) 
print('Train done for {}, n_topics:{}, epoch:{}'.format(taskname,n_topics,epoch))
