
import data_helper
import time
import datetime
import os
import tensorflow as tf

import numpy as np
import evaluation
now = int(time.time()) 
    
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
timeDay = time.strftime("%Y%m%d", timeArray)
print (timeStamp)

def main(args):
    args._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(args.__flags.items()):
        print(("{}={}".format(attr.upper(), value)))
    log_dir = 'log/'+ timeDay
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    data_file = log_dir + '/test_' + args.data + timeStamp
    precision = data_file + 'precise'
    print('load data ...........')
    train,test,dev = data_helper.load(args.data,filter = args.clean)

    q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))

    alphabet = data_helper.get_alphabet([train,test,dev])
    print('the number of words',len(alphabet))

    print('get embedding')
    if args.data=="quora":
        embedding = data_helper.get_embedding(alphabet,language="cn")
    else:
        embedding = data_helper.get_embedding(alphabet)
    
    

    with tf.Graph().as_default(), tf.device("/gpu:" + str(args.gpu)):
        # with tf.device("/cpu:0"):
        session_conf = tf.ConfigProto()
        session_conf.allow_soft_placement = args.allow_soft_placement
        session_conf.log_device_placement = args.log_device_placement
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)

        model = QA_CNN_extend(max_input_left = q_max_sent_length,
            max_input_right = a_max_sent_length,
            batch_size = args.batch_size,
            vocab_size = len(alphabet),
            embedding_size = args.embedding_dim,
            filter_sizes = list(map(int, args.filter_sizes.split(","))),
            num_filters = args.num_filters, 
            hidden_size = args.hidden_size,
            dropout_keep_prob = args.dropout_keep_prob,
            embeddings = embedding,
            l2_reg_lambda = args.l2_reg_lambda,
            trainable = args.trainable,
            pooling = args.pooling,
            conv = args.conv)

        model.build_graph()

        sess.run(tf.global_variables_initializer())
        def train_step(model,sess,batch):
            for data in batch:
                feed_dict = {
                    model.question:data[0],
                    model.answer:data[1],
                    model.answer_negative:data[2],
                    model.q_mask:data[3],
                    model.a_mask:data[4],
                    model.a_neg_mask:data[5]

                }
                _, summary, step, loss, accuracy,score12, score13, see = sess.run(
                        [model.train_op, model.merged,model.global_step,model.loss, model.accuracy,model.score12,model.score13, model.see],
                        feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g} ,positive {:g},negative {:g}".format(time_str, step, loss, accuracy,np.mean(score12),np.mean(score13)))
        def predict(model,sess,batch,test):
            scores = []
            for data in batch:
                feed_dict = {
                    model.question:data[0],
                    model.answer:data[1],
                    model.q_mask:data[2],
                    model.a_mask:data[3]

                }
                score = sess.run(
                        model.score12,
                        feed_dict)
                scores.extend(score)
      
            return np.array(scores[:len(test)])
        
                
        

        
        for i in range(args.num_epoches):
            datas = data_helper.get_mini_batch(train,alphabet,args.batch_size)
            train_step(model,sess,datas)
            test_datas = data_helper.get_mini_batch_test(test,alphabet,args.batch_size)

            predicted_test = predict(model,sess,test_datas,test)
            print(len(predicted_test))
            print(len(test))
            map_mrr_test = evaluation.evaluationBypandas(test,predicted_test)

            print('map_mrr test',map_mrr_test)





                



