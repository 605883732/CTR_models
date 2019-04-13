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
        self.initializer = self._get_initializer(hparams)
        self.label = tf.placeholder(shape=(None), dtype=tf.float32)
        # 训练时和测试时使用不同的batch_norm策略
        self.use_norm=tf.placeholder(tf.bool)
        
        # 训练和测试时使用不同的dropout
        self.second_order_dropout = tf.placeholder(tf.float32)
        self.dnn_dropout = tf.placeholder(shape=(len(hparams.dnn_dropout),),dtype=tf.float32)
        self.features=tf.placeholder(shape=(None,hparams.field_size), dtype=tf.int32)
        self.emb_v1=tf.get_variable(shape=[hparams.hash_ids,1],
                                    initializer=self.initializer,name='emb_v1')
        self.emb_v2=tf.get_variable(shape=[hparams.hash_ids,hparams.embedding_size],
                                    initializer=self.initializer,name='emb_v2')
        
        """
            1阶部分
        """
        # emb_inp_v1: (None * field_size * 1)  
        # 可以证明，1阶的embeeding其实就是在计算每个 w_i * x_i
        emb_inp_v1=tf.nn.embedding_lookup(self.emb_v1, self.features)
        
        # first_order: (None,field_size)
        first_order = tf.reduce_sum(emb_inp_v1,[-1])
        
        
        """
        DNN部分
        """
        emb_inp_v2=tf.nn.embedding_lookup(self.emb_v2, self.features)
        y_deep = tf.reshape(emb_inp_v2,shape=[-1, hparams.field_size * hparams.embedding_size])
        
        input_size =int(y_deep.shape[1])
        for idx in range(0, len(hparams.hidden_size)):
            glorot = np.sqrt(2.0 / (input_size + hparams.hidden_size[idx]))
            W = tf.Variable(np.random.normal(loc=0, scale=glorot,size=(input_size, hparams.hidden_size[idx])), 
                             dtype=tf.float32)
            b = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, hparams.hidden_size[idx])),
                            dtype=tf.float32)
            y_deep = tf.add(tf.matmul(y_deep, W), b) 
            
            """Batch Normalization"""
            if hparams.batch_norm is True:
                y_deep = self.batch_norm_layer( y_deep,self.use_norm,'norm_'+str(idx))
            
            """激励函数，一般是relu"""
            y_deep = self._activate(y_deep, hparams.activation[idx])
            
            """dropout"""
            y_deep = tf.nn.dropout(y_deep, self.dnn_dropout[idx]) 
            
            # input_size的形状重新设定
            input_size =int(y_deep.shape[1]) 
        
        """
        CIN部分
        """
        cin =self._build_extreme_FM(hparams, emb_inp_v2,direct=hparams.cin_direct, bias=hparams.cin_bias)  
        
        
        """
            logit部分
        """
        concat_input = tf.concat([first_order,y_deep,cin],axis=1)
        input_size = int(concat_input.shape[1])
        glorot = np.sqrt(2.0 / (input_size + 1))
        W = tf.Variable(np.random.normal(loc=0, scale=glorot,size=(input_size, 1)), dtype=tf.float32)
        #b = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1,)),dtype=np.float32)
        b = tf.Variable(tf.constant(0.01),dtype=tf.float32)
        
        logit=tf.matmul(concat_input,W)+b  # (None,1)
        """logit 要和 label 维度一致"""
        logit=tf.reshape(logit,[-1]) #(None,)
        
        self.prob=tf.nn.sigmoid(logit)
        self.loss = tf.losses.log_loss(self.label, self.prob)
        # logit_1=tf.log(self.prob+1e-20)
        # logit_0=tf.log(1-self.prob+1e-20)
        # self.loss=-tf.reduce_mean(self.label*logit_1+(1-self.label)*logit_0) #loss
        # self.cost=-(self.label*logit_1+(1-self.label)*logit_0)
        # 用于模型保存，默认只保存最近的5个模型
        self.saver= tf.train.Saver(max_to_keep=5)
        
    def _build_extreme_FM(self, hparams, nn_input, direct=False, bias=False):
        """
        nn_input: 即emb_inp_v2(batch_size,field_size,embedding_size)==>(batch_size,H_0,embedding_size)
        """
        
        # 存储X^0,X^1,X^2,...
        hidden_nn_layers = [] 
        # 存储 H_0,H_1,H_2...
        field_nums = []
        
        #  把H_0加入 field_nums
        field_nums.append(hparams.field_size)
        # 把X^0加入 hidden_nn_layers
        hidden_nn_layers.append(nn_input)
        
        # 最后Sum pooling拼接后形成的输出向量长度
        final_len = 0 
        
        final_result = []
        """
            将X^0分成embedding_size个 batch_size*H_0*1的小矩阵
            可以认为此时 X^0的形状为 (embedding_size,batch*size,H_0,1)
        """
        X_0 = tf.split(hidden_nn_layers[0], hparams.embedding_size * [1], 2)
    
        with tf.variable_scope("exfm_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.cin_layer_sizes):
                """
                把 X^k分成embedding_size个 batch_size*H_k*1的小矩阵,
                即形状为 (embedding_size,batch_size,H_k,1)
                """
                X_k = tf.split(hidden_nn_layers[-1], hparams.embedding_size * [1], 2)
                """
                矩阵相乘，相当于论文中的每行两两交叉：(embedding_size,batch_size,H_0,H_k)
                相当于得到论文中的那个立方体 Z^(k+1)
                """
                dot_result_origin = tf.matmul(X_0, X_k, transpose_b=True)
                
                """
                把最后两维的平面展平：(embedding_size ,batch_size ,H_0 * H_k)
                展平后，该平面和W的加权求和可以很容易用1维卷积来实现
                """
                dot_result_flat = tf.reshape(dot_result_origin, 
                                             shape=[hparams.embedding_size, -1, field_nums[0]*field_nums[-1]])
                
                """ 把batch维度交换到最前面： (batch_size,embedding_size,H_0 * H_k)"""
                dot_result = tf.transpose(dot_result_flat, [1, 0, 2]) 

                
                """ 
                   1维卷积，由于卷积核和数据一样大，其实就是做layer_size遍线性加权求和。
                   curr_out: (batch_size,embedding_size,layer_size)
                """
                filters = tf.get_variable(name="f_"+str(idx),
                                         shape=[1, field_nums[0]*field_nums[-1], layer_size],
                                         dtype=tf.float32)
                curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')
                
                
                """如果要加上bias"""
                if bias:
                    b = tf.get_variable(name="f_b" + str(idx),
                                    shape=[layer_size],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                    curr_out = tf.nn.bias_add(curr_out, b)
                
                """激活函数"""
                curr_out = self._activate(curr_out, hparams.cin_activation)
                
                # 做一下转置：(batch_size,layer_size,embedding_size)
                curr_out = tf.transpose(curr_out, perm=[0, 2, 1])
                
                
                """
                sum pooling 的两种方式
                """
                if direct:
                    """
                    direct: 直接把完整curr_out作为最后输出结果的一部分，
                            同时把完整的curr_out作为计算下一个隐藏层向量的输入
                    """
                    direct_connect = curr_out
                    next_hidden = curr_out
                    final_len += layer_size
                    field_nums.append(int(layer_size))

                else:
                    """
                    非direct方式：把curr_out按照layer_size进行均分，
                                 前一半作为计算下一个隐藏层向量的输入，
                                 后一半作为最后输出结果的一部分。
                    """
                    if idx != len(hparams.cin_layer_sizes) - 1:
                        next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                        final_len += int(layer_size / 2)
                    else:
                        direct_connect = curr_out
                        next_hidden = 0
                        final_len += layer_size
                    field_nums.append(int(layer_size / 2))

                final_result.append(direct_connect)
                hidden_nn_layers.append(next_hidden)
                
            # result (batch_size,H_0+H_1+...,embedding_size)
            result = tf.concat(final_result, axis=1)
            
            """这里真正做了sum pooling,把embedding维度加掉  (batch_size,H_0,H_1....)"""
            result = tf.reduce_sum(result, -1)
            
            return result
        
 
    def optimizer(self,hparams):
        opt=self._build_train_opt(hparams)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss,params,colocate_gradients_with_ops=True)
        clipped_grads, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)  
        self.grad_norm =gradient_norm 
        self.update = opt.apply_gradients(zip(clipped_grads, params)) 

    def train(self,train_data,dev_data):
        hparams=self.hparams # 超参数
        sess=self.sess # sess
        assert len(train_data[0])==len(train_data[1]), "Size of features data must be equal to label"
        
        """
        训练
        """
        for epoch in range(hparams.epoch):
            info={}
            info['loss']=[]
            info['norm']=[]
            start_time = time.time()
            for idx in range(math.ceil(len(train_data[0])/hparams.batch_size)):
                """
                train_data[0] 是数据， train_data[1]是label
                """
                # 截取一个batch的数据，最后一个batch的大小可能不满batch_size
                batch=train_data[0][idx*hparams.batch_size: min((idx+1)*hparams.batch_size,len(train_data[0]))]
                
                # 对每一个特征值做hash(类似于做one-hot)
                batch=utils.hash_batch(batch,hparams)
                
                # 截取一个batch的label，最后一个batch的大小可能不满batch_size
                label=train_data[1][idx*hparams.batch_size: min((idx+1)*hparams.batch_size,len(train_data[1]))]
                

                loss,_,norm=sess.run([self.loss,self.update,self.grad_norm],feed_dict=\
                                     {self.features:batch,self.label:label,self.use_norm:True,
                                      self.dnn_dropout:hparams.dnn_dropout
                                     })
                
                info['loss'].append(loss) # 损失
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
                    
                """eval时如果模型效果好会保存模型"""
                if (idx+1)%hparams.num_eval_steps==0 and dev_data:
                    T=(time.time()-start_time)
                    self.eval(T,dev_data,hparams,sess)
        

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
            """
            infer时不开dropout,BN
            """
            pred=sess.run(self.prob,feed_dict=\
                          {self.features:batch,self.label:label,self.use_norm:False,
                           self.dnn_dropout:len(hparams.dnn_dropout) * [1]
                          })  
            preds.append(pred)   
        preds=np.concatenate(preds)
        return preds
    
    def get_embedding(self,dev_data):
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
            