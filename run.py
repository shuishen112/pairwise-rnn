from tensorflow import flags
import tensorflow as tf
from config import Singleton
import data_helper

import datetime,os

import models
import numpy as np
import evaluation

import sys
import logging

import time
now = int(time.time())
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
log_filename = "log/" +time.strftime("%Y%m%d", timeArray)

program = os.path.basename('program')
logger = logging.getLogger(program) 
if not os.path.exists(log_filename):
    os.makedirs(log_filename)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',datefmt='%a, %d %b %Y %H:%M:%S',filename=log_filename+'/qa.log',filemode='w')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
    


from data_helper import log_time_delta,getLogger

logger=getLogger()
    



args = Singleton().get_qcnn_flag()

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
    
#    ckpt = tf.train.get_checkpoint_state("checkpoint")    
#    if ckpt and ckpt.model_checkpoint_path:    
#        # Restores from checkpoint    
#        saver.restore(sess, ckpt.model_checkpoint_path)
#    if os.path.exists("model") :                        
#        import shutil
#        shutil.rmtree("model")        
#    builder = tf.saved_model.builder.SavedModelBuilder("./model")
#    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
#    builder.save(True)
#    variable_averages = tf.train.ExponentialMovingAverage(  model)    
#    variables_to_restore = variable_averages.variables_to_restore()    
#    saver = tf.train.Saver(variables_to_restore)  
#    for name in variables_to_restore:    
#        print(name) 

    sess.run(tf.global_variables_initializer())
    @log_time_delta
    def predict(model,sess,batch,test):
        scores = []
        for data in batch:            
            score = model.predict(sess,data)
            scores.extend(score)  
        return np.array(scores[:len(test)])
    
    best_p1=0
    
    

    
    for i in range(args.num_epoches):  
        
        for data in train_data_loader(train,alphabet,args.batch_size,model=model,sess=sess):
#        for data in data_helper.getBatch48008(train,alphabet,args.batch_size):
            _, summary, step, loss, accuracy,score12, score13, see = model.train(sess,data)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g} ,positive {:g},negative {:g}".format(time_str, step, loss, accuracy,np.mean(score12),np.mean(score13)))
            logger.info("{}: step {}, loss {:g}, acc {:g} ,positive {:g},negative {:g}".format(time_str, step, loss, accuracy,np.mean(score12),np.mean(score13)))
#<<<<<<< HEAD
#        
# 
#        if i>0 and i % 5 ==0:
#            test_datas = data_helper.get_mini_batch_test(test,alphabet,args.batch_size)
#        
#            predicted_test = predict(model,sess,test_datas,test)
#            map_mrr_test = evaluation.evaluationBypandas(test,predicted_test)
#        
#            logger.info('map_mrr test' +str(map_mrr_test))
#            print('map_mrr test' +str(map_mrr_test))
#            
#            test_datas = data_helper.get_mini_batch_test(dev,alphabet,args.batch_size)
#            predicted_test = predict(model,sess,test_datas,dev)
#            map_mrr_test = evaluation.evaluationBypandas(dev,predicted_test)
#        
#            logger.info('map_mrr dev' +str(map_mrr_test))
#            print('map_mrr dev' +str(map_mrr_test))
#            map,mrr,p1 = map_mrr_test
#            if p1>best_p1:
#                best_p1=p1
#                filename= "checkpoint/"+args.data+"_"+str(p1)+".model"
#                save_path = saver.save(sess, filename)  
#        #            load_path = saver.restore(sess, model_path)
#                
#                import shutil
#                shutil.rmtree("model")
#                builder = tf.saved_model.builder.SavedModelBuilder("./model")
#                builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
#                builder.save(True)
#        
#        
#=======

        test_datas = data_helper.get_mini_batch_test(test,alphabet,args.batch_size)

        predicted_test = predict(model,sess,test_datas,test)
        map_mrr_test = evaluation.evaluationBypandas(test,predicted_test)

        logger.info('map_mrr test' +str(map_mrr_test))
        print('epoch '+ str(i) + 'map_mrr test' +str(map_mrr_test))

