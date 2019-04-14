import pandas as pd
import numpy as np
import tensorflow as tf
import ctrNet
import sys
from sklearn.model_selection import train_test_split
import utils
tf.logging.set_verbosity(tf.logging.DEBUG)
#tf.enable_eager_execution()
import os
from sklearn import metrics


train_df=pd.read_csv('data/train_small.txt',header=None,sep='\t')
train_df.columns=['label']+['f'+str(i) for i in range(39)]
train_df, dev_df,_,_ = train_test_split(train_df,train_df,test_size=0.1, random_state=2019)
dev_df, test_df,_,_ = train_test_split(dev_df,dev_df,test_size=0.5, random_state=2019)
features=['f'+str(i) for i in range(39)]

# 第一列是label,后面39列是特征，有数值特征，也有categorical特征
# 未做任何one-hot,归一化
# train_df (9000,40)
# dev_df   (5000,40)
# test_df  (5000,40)


# DeepFM
DeepFM_params=tf.contrib.training.HParams(
                model='deepFM',
                batch_norm=True,
                batch_norm_decay=0.9,
                hidden_size=[128,128],
                dnn_dropout=[0.5,0.5],
                second_order_dropout=0.5,
                embedding_size=8,
                hash_ids=int(1e5),
                batch_size=64,
                optimizer="adam",
                learning_rate=0.001,
                num_display_steps=100,# 多少个step打印一次输出
                num_eval_steps=1000,# 多少个step时做一次eval（eval时会保存模型）
                epoch=2,
                metric='auc',
                activation=['relu','relu'],
                #['tnormal','uniform','normal','xavier_normal','xavier_uniform','he_normal','he_uniform']
                init_method='uniform',
                init_value=0.1,
                field_size=len(features),
                l2 = 0.01 # l2正则,不可是整数
)

# NFFM
NFFM_params=tf.contrib.training.HParams(
                model='nffm',
                batch_norm=True,
                batch_norm_decay=0.9,
                hidden_size=[128,128],
                dnn_dropout=[0.5,0.5],
                second_order_dropout=0.5,
                use_first_order=False, # 是否把first_order拼接到second_order上，作为DNN的输入
                activation=['relu','relu'],
                drop_out_list =[0.5,0.5],
                embedding_size=8,
                hash_ids=int(1e5),
                batch_size=64,
                optimizer="adam",
                learning_rate=0.001,
                num_display_steps=100, # 多少个step打印一次输出
                num_eval_steps=1000,  # 多少个step时做一次eval（eval时会保存模型）
                epoch=2,
                metric='auc',
                init_method='uniform', 
                init_value=0.1, 
                field_size=len(features)
)

# FFM
FFM_params=tf.contrib.training.HParams(
            model='ffm', #['fm','ffm','nffm']
            embedding_size=8,
            hash_ids=int(1e5),
            batch_size=64,
            optimizer="adam", #['adadelta','adagrad','sgd','adam','ftrl','gd','padagrad','pgd','rmsprop']
            learning_rate=0.0002,
            num_display_steps=100,
            num_eval_steps=1000,
            epoch=2,
            metric='auc', #['auc','logloss']
            init_method='uniform', #['tnormal','uniform','normal','xavier_normal','xavier_uniform','he_normal','he_uniform']
            init_value=0.1,
            field_size=len(features)
)

# FM
FM_params=tf.contrib.training.HParams(
            model='fm', #['fm','ffm','nffm']
            embedding_size=16,
            hash_ids=int(1e5), #  hash桶大小(可以理解为one-hot后共有100000维左右)
            batch_size=64,
            optimizer="adam", #['adadelta','adagrad','sgd','adam','ftrl','gd','padagrad','pgd','rmsprop']
            learning_rate=0.0005,
            num_display_steps=100,
            num_eval_steps=1000,
            epoch=2,
            metric='auc', #['auc','logloss']
            init_method='uniform', #['tnormal','uniform','normal','xavier_normal','xavier_uniform','he_normal','he_uniform']
            init_value=0.1,
            field_size=len(features)  #39个field
)

# DCN
DCN_params=tf.contrib.training.HParams(
                model='DCN',
                batch_norm=True,
                batch_norm_decay=0.9,
                hidden_size=[128,128],
                dnn_dropout=[0.5,0.5],
                cross_layer_num = 3, # 交叉层层数
                embedding_size=8,
                hash_ids=int(1e5),
                batch_size=64,
                optimizer="adam",
                learning_rate=0.001,
                num_display_steps=100,# 多少个step打印一次输出
                num_eval_steps=1000,# 多少个step时做一次eval（eval时会保存模型）
                epoch=2,
                metric='auc',
                activation=['relu','relu'],
                #['tnormal','uniform','normal','xavier_normal','xavier_uniform','he_normal','he_uniform']
                init_method='uniform',
                init_value=0.1,
                field_size=len(features))

# XDeepFM
XDeepFM_params=tf.contrib.training.HParams(
                    model='xdeepfm',
                    batch_norm=True,
                    batch_norm_decay=0.9,
                    hidden_size=[128,128],
                    dnn_dropout=[0.5,0.5],
                    cin_layer_sizes=[128,128,128],
                    embedding_size=8,
                    hash_ids=int(2e5),
                    batch_size=64,
                    optimizer="adam",
                    learning_rate=0.001,
                    num_display_steps=100,
                    num_eval_steps=1000,
                    epoch=2,
                    metric='auc',
                    activation=['relu','relu','relu'],
                    cin_activation='identity',
                    init_method='uniform',
                    init_value=0.1,
                    field_size=len(features),
                    cin_bias=False,
                    cin_direct=False
)

# AFM
AFM_params=tf.contrib.training.HParams(
                model='AFM',
                embedding_size=8,
                hash_ids=int(1e5),
                batch_size=64,
                optimizer="adam",
                learning_rate=0.001,
                num_display_steps=100,# 多少个step打印一次输出
                num_eval_steps=1000,# 多少个step时做一次eval（eval时会保存模型）
                epoch=2,
                metric='auc',
                #['tnormal','uniform','normal','xavier_normal','xavier_uniform','he_normal','he_uniform']
                init_method='uniform',
                init_value=0.1,
                field_size=len(features), 
                attention_size = 10, #Attention Net 的hidden size
                l2 = 0.0 # l2正则,不可是整数
)

if __name__=='__main__':
    model_name = sys.argv[1]
    if(model_name == "NFFM"):
        hparam = NFFM_params
    elif(model_name == "DeepFM"):
        hparam = DeepFM_params
    elif(model_name == "FFM"):
        hparam = FFM_params   
    elif(model_name == "FM"):
        hparam = FM_params
    elif(model_name == "DCN"):
        hparam = DCN_params
    elif(model_name == "XDeepFM"):
        hparam = XDeepFM_params
    elif(model_name == "AFM"):
        hparam = AFM_params
        
    utils.print_hparams(hparam) # 打印模型参数
    
    model=ctrNet.build_model(hparam) # 建立模型

    print("Start "+ model_name)
    model.train(train_data=(train_df[features],train_df['label']),dev_data=(dev_df[features],dev_df['label']))
    
    preds=model.infer(dev_data=(test_df[features],test_df['label']))
    
    fpr, tpr, thresholds = metrics.roc_curve(test_df['label']+1, preds, pos_label=2)
    auc=metrics.auc(fpr, tpr)
    print("last model auc {}".format(auc))
    print(model_name + " Done")