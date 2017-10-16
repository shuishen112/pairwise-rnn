from tensorflow import flags
import tensorflow as tf
from config import Singleton
import data_helper
import time
import datetime
import os
import models
import numpy as np
import evaluation
import sys
import logging


now = int(time.time()) 
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
log_filename = "log/" +time.strftime("%Y%m%d", timeArray)

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program) 
if not os.path.exists(log_filename):
    os.path.mkdir(log_filename)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',datefmt='%a, %d %b %Y %H:%M:%S',filename=log_filename+'/qa.log',filemode='w')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
    



args = Singleton().get_rnn_flag()
args._parse_flags()
opts=dict()
logger.info("\nParameters:")
for attr, value in sorted(args.__flags.items()):
    logger.info(("{}={}".format(attr.upper(), value)))
    opts[attr]=value



logger.info('load data ...........')
train,test,dev = data_helper.load(args.data,filter = args.clean)

q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))

alphabet = data_helper.get_alphabet([train,test,dev])
logger.info('the number of words :%d '%len(alphabet))


embedding = data_helper.get_embedding(alphabet)

opts["embeddings"] =embedding
opts["vocab_size"]=len(alphabet)
opts["max_input_right"]=a_max_sent_length
opts["max_input_left"]=q_max_sent_length
opts["filter_sizes"]=list(map(int, args.filter_sizes.split(",")))


with tf.Graph().as_default(), tf.device("/gpu:" + str(args.gpu)):
    # with tf.device("/cpu:0"):
    session_conf = tf.ConfigProto()
    session_conf.allow_soft_placement = args.allow_soft_placement
    session_conf.log_device_placement = args.log_device_placement
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    model=models.setup(opts)
    model.build_graph()    
    sess.run(tf.global_variables_initializer())
 
    def predict(model,sess,batch,test):
        scores = []
        for data in batch:            
            score = model.predict(sess,data)
            scores.extend(score)  
        return np.array(scores[:len(test)])

    for i in range(args.num_epoches):  
        
        for data in data_helper.get_mini_batch(train,alphabet,args.batch_size):
            _, summary, step, loss, accuracy,score12, score13, see = model.train(sess,data)
            time_str = datetime.datetime.now().isoformat()
#            print("{}: step {}, loss {:g}, acc {:g} ,positive {:g},negative {:g}".format(time_str, step, loss, accuracy,np.mean(score12),np.mean(score13)))
            logger.info("{}: step {}, loss {:g}, acc {:g} ,positive {:g},negative {:g}".format(time_str, step, loss, accuracy,np.mean(score12),np.mean(score13)))

        test_datas = data_helper.get_mini_batch_test(test,alphabet,args.batch_size)

        predicted_test = predict(model,sess,test_datas,test)
        map_mrr_test = evaluation.evaluationBypandas(test,predicted_test)

        logger.info('map_mrr test' +str(map_mrr_test))
        print('map_mrr test' +str(map_mrr_test))
