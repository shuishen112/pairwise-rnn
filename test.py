# -*- coding: utf-8 -*-

from tensorflow import flags
import tensorflow as tf
from config import Singleton
import data_helper

import datetime
import os
import models
import numpy as np
import evaluation

from data_helper import log_time_delta,getLogger

logger=getLogger()
    


args = Singleton().get_rnn_flag()
#args = Singleton().get_8008_flag()

args._parse_flags()
opts=dict()
logger.info("\nParameters:")
for attr, value in sorted(args.__flags.items()):
    logger.info(("{}={}".format(attr.upper(), value)))
    opts[attr]=value


train,test,dev = data_helper.load(args.data,filter = args.clean)

q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))

alphabet = data_helper.get_alphabet([train,test,dev],dataset=args.data )
logger.info('the number of words :%d '%len(alphabet))

if args.data=="quora" or  args.data=="8008" :
    print("cn embedding")
    embedding = data_helper.get_embedding(alphabet,dim=200,language="cn",dataset=args.data )
    train_data_loader = data_helper.getBatch48008
else:
    embedding = data_helper.get_embedding(alphabet,dim=300,dataset=args.data )
    train_data_loader = data_helper.get_mini_batch
opts["embeddings"] =embedding
opts["vocab_size"]=len(alphabet)
opts["max_input_right"]=a_max_sent_length
opts["max_input_left"]=q_max_sent_length
opts["filter_sizes"]=list(map(int, args.filter_sizes.split(",")))

print("innitilize over")


   
 
#with tf.Graph().as_default(), tf.device("/gpu:" + str(args.gpu)):
with tf.Graph().as_default():    
    # with tf.device("/cpu:0"):
    session_conf = tf.ConfigProto()
    session_conf.allow_soft_placement = args.allow_soft_placement
    session_conf.log_device_placement = args.log_device_placement
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    model=models.setup(opts)
    model.build_graph()    
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())  # fun first than print or save
    
    
    ckpt = tf.train.get_checkpoint_state("checkpoint")    
    if ckpt and ckpt.model_checkpoint_path:    
        # Restores from checkpoint    
        saver.restore(sess, ckpt.model_checkpoint_path)
    print(sess.run(model.position_embedding)[0])
    if os.path.exists("model") :                        
        import shutil
        shutil.rmtree("model")
    builder = tf.saved_model.builder.SavedModelBuilder("./model")
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
    builder.save(True)
    variable_averages = tf.train.ExponentialMovingAverage(  model)    
    variables_to_restore = variable_averages.variables_to_restore()    
    saver = tf.train.Saver(variables_to_restore)  
    for name in variables_to_restore:    
        print(name) 
   
    @log_time_delta
    def predict(model,sess,batch,test):
        scores = []
        for data in batch:            
            score = model.predict(sess,data)
            scores.extend(score)  
        return np.array(scores[:len(test)])
    
    
    text = "怎么 提取 公积金 ？"
  
    splited_text=data_helper.encode_to_split(text,alphabet)

    mb_q,mb_q_mask = data_helper.prepare_data([splited_text])
    mb_a,mb_a_mask = data_helper.prepare_data([splited_text])
    
    data = (mb_q,mb_a,mb_q_mask,mb_a_mask)
    score = model.predict(sess,data)
    print(score)
    feed_dict = {
                model.question:data[0],
                model.answer:data[1],
                model.q_mask:data[2],
                model.a_mask:data[3],
                model.dropout_keep_prob_holder:1.0
            }   
    sess.run(model.position_embedding,feed_dict=feed_dict)[0]

    
   