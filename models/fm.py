import tensorflow as tf
import utils
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers import core as layers_core
from models.base_model import BaseModel
import numpy as np
import time 
import os
import math
class Model(BaseModel):
    def __init__(self,hparams):
        self.hparams=hparams
        if hparams.metric in ['logloss']:
            self.best_score=100000
        else:
            self.best_score=0
        self.build_graph(hparams)   
        self.optimizer(hparams)
        params = tf.trainable_variables()
        utils.print_out("# Trainable variables")
        for param in params:
            utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),param.op.device))   
  
    def set_Session(self,sess):
        self.sess=sess
        
    def build_graph(self, hparams):
        # 随机初始化器,比如 tf.contrib.layers.xavier_initializer
        initializer = self._get_initializer(hparams)
        
        # label
        self.label = tf.placeholder(shape=(None), dtype=tf.float32) 
        # features: None*field_size
        self.features=tf.placeholder(shape=(None,hparams.field_size), dtype=tf.int32) 
        
        # emb_v1: (hash_ids,1) 
        self.emb_v1=tf.get_variable(shape=[hparams.hash_ids,1],
                                    initializer=initializer,name='emb_v1')
        # emb_v2: (hash_ids,embedding_size)
        self.emb_v2=tf.get_variable(shape=[hparams.hash_ids,hparams.embedding_size],
                                    initializer=initializer,name='emb_v2')
        
        """
            1阶部分
        """
        # emb_inp_v1: (None * field_size * 1)  
        # 可以证明，1阶的embeeding其实就是在计算每个 w_i * x_i
        emb_inp_v1=tf.nn.embedding_lookup(self.emb_v1, self.features)
        
        # first_order: (None,)
        # 这里求和其实就是在每个batch上计算  w_1*x_1 + w_2*x_2...，对于每一个batch,结果是一个数
        first_order = tf.reduce_sum(emb_inp_v1,[-1,-2])
        tf.logging.debug("first_order= {}".format(first_order.shape))
        
        """
            2阶部分
        """
        # emb_inp_v2 : (None,field_size,embedding_size)
        # emb_inp_v2 算的就是每个特征对应的隐向量,如果特征是One-hot的，对应的其实是隐向量*特征值
        emb_inp_v2=tf.nn.embedding_lookup(self.emb_v2, self.features)
        
        summed_features_emb = tf.reduce_sum(emb_inp_v2,axis=1)
        summed_features_emb_square = tf.square(summed_features_emb)
        
        square_features_emb = tf.square(emb_inp_v2)
        #[batch_size,embedding_size]
        square_features_emb_summed = tf.reduce_sum(square_features_emb,axis=1)
        
        second_order = 0.5 * tf.subtract(summed_features_emb_square,square_features_emb_summed)
        second_order = tf.reduce_sum(second_order,-1) 

        
        """针对logloss一点优化，但只能用于二元分类"""
        logit=first_order+second_order
        self.prob=tf.nn.sigmoid(logit)
        self.loss = tf.losses.log_loss(self.label, self.prob)
        # logit_1=tf.log(self.prob+1e-20)
        # logit_0=tf.log(1-self.prob+1e-20)
        # self.loss=-tf.reduce_mean(self.label*logit_1+(1-self.label)*logit_0) #loss
        #self.cost=-(self.label*logit_1+(1-self.label)*logit_0)
        # 用于模型保存，默认只保存最近的5个模型
        self.saver= tf.train.Saver(max_to_keep=5)
            
    def optimizer(self,hparams):
        """
           参数更新，使用了clipped grad
        """
        # 返回优化器：train_step = tf.train.AdadeltaOptimizer(hparams.learning_rate)
        opt=self._build_train_opt(hparams)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss,params,colocate_gradients_with_ops=True)
        clipped_grads, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)  
        self.grad_norm =gradient_norm 
        self.update = opt.apply_gradients(zip(clipped_grads, params)) 

    def train(self,train_data,dev_data=None):
        """
            train_data: (num_train_data * field*size, num_train_data *1)
            dev_data:   (num_dev_data * field*size, num_dev_data * field*size)
        """
        
        hparams=self.hparams
        sess=self.sess
        assert len(train_data[0])==len(train_data[1])
        
        for epoch in range(hparams.epoch):
            info={}
            info['loss']=[]
            info['norm']=[]
            start_time = time.time()
            for idx in range(math.ceil(len(train_data[0])/hparams.batch_size)):
                     
                """分batch,并针对每个特征值做hash"""
                batch=train_data[0][idx*hparams.batch_size:\
                                    min((idx+1)*hparams.batch_size,len(train_data[0]))]
                batch=utils.hash_batch(batch,hparams) # hash
                label=train_data[1][idx*hparams.batch_size:\
                                    min((idx+1)*hparams.batch_size,len(train_data[1]))]
                """训练"""
                loss,_,norm=sess.run([self.loss,self.update,self.grad_norm],\
                                     feed_dict={self.features:batch,self.label:label})
                """打印一些信息"""
                info['loss'].append(loss)
                info['norm'].append(norm)
                if (idx+1)%hparams.num_display_steps==0:
                    info['learning_rate']=hparams.learning_rate
                    info["train_ppl"]= np.mean(info['loss'])
                    info["avg_grad_norm"]=np.mean(info['norm'])
                    utils.print_step_info("  ", epoch,idx+1, info)
                    del info
                    info={}
                    info['loss']=[]
                    info['norm']=[]
                if (idx+1)%hparams.num_eval_steps==0 and dev_data:
                    T=(time.time()-start_time)
                    self.eval(T,dev_data,hparams,sess)
        
        """保存模型"""
        self.saver.restore(sess,'model_tmp/model')
        T=(time.time()-start_time)
        self.eval(T,dev_data,hparams,sess)
        os.system("rm -r model_tmp")
        
      
    def infer(self,dev_data):
        hparams=self.hparams
        sess=self.sess
        assert len(dev_data[0])==len(dev_data[1]), "Size of features data must be equal to label"       
        preds=[]
        total_loss=[]
        for idx in range(len(dev_data[0])//hparams.batch_size+1):
            batch=dev_data[0][idx*hparams.batch_size:\
                              min((idx+1)*hparams.batch_size,len(dev_data[0]))]
            if len(batch)==0:
                break
            batch=utils.hash_batch(batch,hparams)
            label=dev_data[1][idx*hparams.batch_size:\
                              min((idx+1)*hparams.batch_size,len(dev_data[1]))]
            pred=sess.run(self.prob,\
                          feed_dict={self.features:batch,self.label:label})  
            preds.append(pred)   
        preds=np.concatenate(preds)
        return preds
    
    def get_embedding(self,dev_data):
        """
         应该是把中间的FM二阶embeeding向量返回，可以用于别的任务?
        """
        hparams=self.hparams
        sess=self.sess
        assert len(dev_data[0])==len(dev_data[1]), "Size of features data must be equal to label"       
        embedding=[]
        total_loss=[]
        for idx in range(len(dev_data[0])//hparams.batch_size+1):
            batch=dev_data[0][idx*hparams.batch_size:\
                              min((idx+1)*hparams.batch_size,len(dev_data[0]))]
            if len(batch)==0:
                break
            batch=utils.hash_batch(batch,hparams)
            label=dev_data[1][idx*hparams.batch_size:\
                              min((idx+1)*hparams.batch_size,len(dev_data[1]))]
            temp=sess.run(self.emb_inp_v2,\
                          feed_dict={self.features:batch,self.label:label})  
            embedding.append(temp)   
        embedding=np.concatenate(embedding,0)
        return embedding
            
