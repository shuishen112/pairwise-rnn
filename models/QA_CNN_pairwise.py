#coding:utf-8
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import models.blocks as blocks
# model_type :apn or qacnn
class QA_CNN_extend(object):
#    def __init__(self,max_input_left,max_input_right,batch_size,vocab_size,embedding_size,filter_sizes,num_filters,hidden_size,
#        dropout_keep_prob = 1,learning_rate = 0.001,embeddings = None,l2_reg_lambda = 0.0,trainable = True,pooling = 'attentive',conv = 'narrow'):
#
#        """
#            QA_RNN model for question answering
#
#            Args:
#                self.dropout_keep_prob: dropout rate
#                self.num_filters : number of filters
#                self.para : parameter list
#                self.extend_feature_dim : my extend feature dimension
#                self.max_input_left : the length of question
#                self.max_input_right : the length of answer
#                self.pooling : pooling strategy :max pooling or attentive pooling
#                
#        """
#        self.dropout_keep_prob =  tf.placeholder(tf.float32,name = 'dropout_keep_prob')
#        self.num_filters = num_filters
#        self.embeddings = embeddings
#        self.embedding_size = embedding_size
#        self.batch_size = batch_size
#        self.filter_sizes = filter_sizes
#        self.l2_reg_lambda = l2_reg_lambda
#        self.para = []
#
#        self.max_input_left = max_input_left
#        self.max_input_right = max_input_right
#        self.trainable = trainable
#        self.vocab_size = vocab_size
#        self.pooling = pooling
#        self.total_num_filter = len(self.filter_sizes) * self.num_filters
#
#        self.conv = conv
#        self.pooling = 'traditional'
#        self.learning_rate = learning_rate
#
#        self.hidden_size = hidden_size
#
#        self.attention_size = 100
    def __init__(self,opt):
        for key,value in opt.items():
            self.__setattr__(key,value)
        self.attention_size = 100
        self.pooling = 'mean'
        self.total_num_filter = len(self.filter_sizes) * self.num_filters
        self.para = []
        self.dropout_keep_prob_holder =  tf.placeholder(tf.float32,name = 'dropout_keep_prob')
    def create_placeholder(self):
        print(('Create placeholders'))
        # he length of the sentence is varied according to the batch,so the None,None
        self.question = tf.placeholder(tf.int32,[None,None],name = 'input_question')
        self.max_input_left = tf.shape(self.question)[1]
   
        self.batch_size = tf.shape(self.question)[0]
        self.answer = tf.placeholder(tf.int32,[None,None],name = 'input_answer')
        self.max_input_right = tf.shape(self.answer)[1]
        self.answer_negative = tf.placeholder(tf.int32,[None,None],name = 'input_right')
        # self.q_mask = tf.placeholder(tf.int32,[None,None],name = 'q_mask')
        # self.a_mask = tf.placeholder(tf.int32,[None,None],name = 'a_mask')
        # self.a_neg_mask = tf.placeholder(tf.int32,[None,None],name = 'a_neg_mask')

    def add_embeddings(self):
        print( 'add embeddings')
        if self.embeddings is not None:
            print( "load embedding")
            W = tf.Variable(np.array(self.embeddings),name = "W" ,dtype="float32",trainable = self.trainable)
            
        else:
            print( "random embedding")
            W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="W",trainable = self.trainable)
        self.embedding_W = W
       
        # self.overlap_W = tf.Variable(a,name="W",trainable = True)
        self.para.append(self.embedding_W)

        self.q_embedding =  tf.nn.embedding_lookup(self.embedding_W,self.question)


        self.a_embedding = tf.nn.embedding_lookup(self.embedding_W,self.answer)
        self.a_neg_embedding = tf.nn.embedding_lookup(self.embedding_W,self.answer_negative)
        #real length
        self.q_len,self.q_mask = blocks.length(self.question)
        self.a_len,self.a_mask = blocks.length(self.answer)
        self.a_neg_len,self.a_neg_mask = blocks.length(self.answer_negative)

    def convolution(self):
        print( 'convolution:wide_convolution')
        self.kernels = []
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-max-pool-%s' % filter_size):
                filter_shape = [filter_size,self.embedding_size,1,self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name="W")
                b = tf.Variable(tf.constant(0.0, shape=[self.num_filters]), name="b")
                self.kernels.append((W,b))
                self.para.append(W)
                self.para.append(b)
       
        embeddings = [self.q_embedding,self.a_embedding,self.a_neg_embedding]

        self.q_cnn,self.a_cnn,self.a_neg_cnn = [self.wide_convolution(tf.expand_dims(embedding,-1)) for embedding in embeddings]

        #convolution
    def pooling_graph(self):
        if self.pooling == 'mean':

            self.q_pos_cnn = self.mean_pooling(self.q_cnn,self.q_mask)
            self.q_neg_cnn = self.mean_pooling(self.q_cnn,self.q_mask)
            self.a_pos_cnn = self.mean_pooling(self.a_cnn,self.a_mask)
            self.a_neg_cnn = self.mean_pooling(self.a_neg_cnn,self.a_neg_mask)
        elif self.pooling == 'attentive':
            self.q_pos_cnn,self.a_pos_cnn = self.attentive_pooling(self.q_cnn,self.a_cnn,self.q_mask,self.a_mask)
            self.q_neg_cnn,self.a_neg_cnn = self.attentive_pooling(self.q_cnn,self.a_neg_cnn,self.q_mask,self.a_neg_mask)
        elif self.pooling == 'position':
            self.q_pos_cnn,self.a_pos_cnn = self.position_attention(self.q_cnn,self.a_cnn,self.q_mask,self.a_mask)
            self.q_neg_cnn,self.a_neg_cnn = self.position_attention(self.q_cnn,self.a_neg_cnn,self.q_mask,self.a_neg_mask)
        elif self.pooling == 'traditional':
            print( self.pooling)
            print(self.q_cnn)
            self.q_pos_cnn,self.a_pos_cnn = self.traditional_attention(self.q_cnn,self.a_cnn,self.q_mask,self.a_mask)
            self.q_neg_cnn,self.a_neg_cnn = self.traditional_attention(self.q_cnn,self.a_neg_cnn,self.q_mask,self.a_neg_mask)

    def para_initial(self):
        # print(("---------"))
        # self.W_qp = tf.Variable(tf.truncated_normal(shape = [self.hidden_size * 2,1],stddev = 0.01,name = 'W_qp'))
        self.U = tf.Variable(tf.truncated_normal(shape = [self.total_num_filter,self.total_num_filter],stddev = 0.01,name = 'U'))
        self.W_hm = tf.Variable(tf.truncated_normal(shape = [self.total_num_filter,self.total_num_filter],stddev = 0.01,name = 'W_hm'))
        self.W_qm = tf.Variable(tf.truncated_normal(shape = [self.total_num_filter,self.total_num_filter],stddev = 0.01,name = 'W_qm'))
        self.W_ms = tf.Variable(tf.truncated_normal(shape = [self.total_num_filter,1],stddev = 0.01,name = 'W_ms'))
        self.M_qi = tf.Variable(tf.truncated_normal(shape = [self.total_num_filter,self.embedding_size],stddev = 0.01,name = 'M_qi'))



    def mean_pooling(self,conv,mask):
   
        conv = tf.squeeze(conv,2)
        print( tf.expand_dims(tf.cast(mask,tf.float32),-1))
        # conv_mask = tf.multiply(conv,tf.expand_dims(tf.cast(mask,tf.float32),-1))
        # self.see = conv_mask
        # print( conv_mask)
        return tf.reduce_mean(conv,axis = 1);
    def attentive_pooling(self,input_left,input_right,q_mask,a_mask):

        Q = tf.squeeze(input_left,axis = 2)
        A = tf.squeeze(input_right,axis = 2)
        print( Q)
        print( A)
        # Q = tf.reshape(input_left,[-1,self.max_input_left,len(self.filter_sizes) * self.num_filters],name = 'Q')
        # A = tf.reshape(input_right,[-1,self.max_input_right,len(self.filter_sizes) * self.num_filters],name = 'A')
        # G = tf.tanh(tf.matmul(tf.matmul(Q,self.U),\
        # A,transpose_b = True),name = 'G')
        
        first = tf.matmul(tf.reshape(Q,[-1,len(self.filter_sizes) * self.num_filters]),self.U)
        second_step = tf.reshape(first,[-1,self.max_input_left,len(self.filter_sizes) * self.num_filters])
        result = tf.matmul(second_step,tf.transpose(A,perm = [0,2,1]))
        print( second_step)
        print( tf.transpose(A,perm = [0,2,1]))
        # print( 'result',result)
        G = tf.tanh(result)
        
        # G = result
        # column-wise pooling ,row-wise pooling
        row_pooling = tf.reduce_max(G,1,True,name = 'row_pooling')
        col_pooling = tf.reduce_max(G,2,True,name = 'col_pooling')
    
        self.attention_q = tf.nn.softmax(col_pooling,1,name = 'attention_q')
        self.attention_q_mask = tf.multiply(self.attention_q,tf.expand_dims(tf.cast(q_mask,tf.float32),-1))
        self.attention_a = tf.nn.softmax(row_pooling,name = 'attention_a')
        self.attention_a_mask = tf.multiply(self.attention_a,tf.expand_dims(tf.cast(a_mask,tf.float32),1))
        
        self.see = G

        R_q = tf.reshape(tf.matmul(Q,self.attention_q_mask,transpose_a = 1),[-1,self.num_filters * len(self.filter_sizes)],name = 'R_q')
        R_a = tf.reshape(tf.matmul(self.attention_a_mask,A),[-1,self.num_filters * len(self.filter_sizes)],name = 'R_a')

        return R_q,R_a

    def traditional_attention(self,input_left,input_right,q_mask,a_mask):
        input_left = tf.squeeze(input_left,axis = 2)
        input_right = tf.squeeze(input_right,axis = 2) 

        input_left_mask = tf.multiply(input_left, tf.expand_dims(tf.cast(q_mask,tf.float32),2))
        Q = tf.reduce_mean(input_left_mask,1)
        a_shape = tf.shape(input_right)
        A = tf.reshape(input_right,[-1,self.total_num_filter])
        m_t = tf.nn.tanh(tf.reshape(tf.matmul(A,self.W_hm),[-1,a_shape[1],self.total_num_filter]) + tf.expand_dims(tf.matmul(Q,self.W_qm),1))
        f_attention = tf.exp(tf.reshape(tf.matmul(tf.reshape(m_t,[-1,self.total_num_filter]),self.W_ms),[-1,a_shape[1],1]))
        self.f_attention_mask = tf.multiply(f_attention,tf.expand_dims(tf.cast(a_mask,tf.float32),2))
        self.f_attention_norm = tf.divide(self.f_attention_mask,tf.reduce_sum(self.f_attention_mask,1,keep_dims = True))
        self.see = self.f_attention_norm
        a_attention = tf.reduce_sum(tf.multiply(input_right,self.f_attention_norm),1)
        return Q,a_attention
    def position_attention(self,input_left,input_right,q_mask,a_mask):
        input_left = tf.squeeze(input_left,axis = 2)
        input_right = tf.squeeze(input_right,axis = 2)
        # Q = tf.reshape(input_left,[-1,self.max_input_left,self.hidden_size*2],name = 'Q')
        # A = tf.reshape(input_right,[-1,self.max_input_right,self.hidden_size*2],name = 'A')

        Q = tf.reduce_mean(tf.multiply(input_left,tf.expand_dims(tf.cast(self.q_mask,tf.float32),2)),1)

        QU = tf.matmul(Q,self.U)
        QUA = tf.multiply(tf.expand_dims(QU,1),input_right)
        self.attention_a = tf.cast(tf.argmax(QUA,2)
            ,tf.float32)
        # q_shape = tf.shape(input_left)
        # Q_1 = tf.reshape(input_left,[-1,self.total_num_filter])
        # QU = tf.matmul(Q_1,self.U)
        # QU_1 = tf.reshape(QU,[-1,q_shape[1],self.total_num_filter])
        # A_1 = tf.transpose(input_right,[0,2,1])
        # QUA = tf.matmul(QU_1,A_1)
        # QUA = tf.nn.l2_normalize(QUA,1)

        # G = tf.tanh(QUA)
        # Q = tf.reduce_mean(tf.multiply(input_left,tf.expand_dims(tf.cast(self.q_mask,tf.float32),2)),1)
        # # self.Q_mask = tf.multiply(input_left,tf.expand_dims(tf.cast(self.q_mask,tf.float32),2))
        # row_pooling = tf.reduce_max(G,1,name="row_pooling")
        # col_pooling = tf.reduce_max(G,2,name="col_pooling")
        # self.attention_a = tf.nn.softmax(row_pooling,1,name = "attention_a")
        self.attention_a_mask = tf.multiply(self.attention_a,tf.cast(a_mask,tf.float32))
        self.see = self.attention_a
        self.attention_a_norm = tf.divide(self.attention_a_mask,tf.reduce_sum(self.attention_a_mask,1,keep_dims =True))
        self.r_a = tf.reshape(tf.matmul(tf.transpose(input_right,[0,2,1]) ,tf.expand_dims(self.attention_a_norm,2)),[-1,self.total_num_filter])
        return Q ,self.r_a
    def create_loss(self):
        
        with tf.name_scope('score'):
            self.score12 = self.getCosine(self.q_pos_cnn,self.a_pos_cnn)
            self.score13 = self.getCosine(self.q_neg_cnn,self.a_neg_cnn)
        l2_loss = tf.constant(0.0)
        for p in self.para:
            l2_loss += tf.nn.l2_loss(p)
        with tf.name_scope("loss"):
            self.losses = tf.maximum(0.0, tf.subtract(0.05, tf.subtract(self.score12, self.score13)))
            self.loss = tf.reduce_sum(self.losses) + self.l2_reg_lambda * l2_loss
        tf.summary.scalar('loss', self.loss)
        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct = tf.equal(0.0, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")
        tf.summary.scalar('accuracy', self.accuracy)
    def create_op(self):
        self.global_step = tf.Variable(0, name = "global_step", trainable = False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step = self.global_step)


    def max_pooling(self,conv,input_length):
        pooled = tf.nn.max_pool(
                    conv,
                    ksize = [1, input_length, 1, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name="pool")
        return pooled
    def getCosine(self,q,a):
        pooled_flat_1 = tf.nn.dropout(q, self.dropout_keep_prob_holder)
        pooled_flat_2 = tf.nn.dropout(a, self.dropout_keep_prob_holder)
        
        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_1), 1)) 
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_2, pooled_flat_2), 1))
        pooled_mul_12 = tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_2), 1) 
        score = tf.div(pooled_mul_12, tf.multiply(pooled_len_1, pooled_len_2), name="scores") 
        return score
    def wide_convolution(self,embedding):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    self.kernels[i][0],
                    strides=[1, 1, self.embedding_size, 1],
                    padding='SAME',
                    name="conv-1"
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
            cnn_outputs.append(h)
        cnn_reshaped = tf.concat(cnn_outputs,3)
        return cnn_reshaped
    
    def variable_summaries(self,var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def build_graph(self):
        self.create_placeholder()
        self.add_embeddings()
        self.para_initial()
        self.convolution()
        self.pooling_graph()
        self.create_loss()
        self.create_op()
        self.merged = tf.summary.merge_all()

    def train(self,sess,data):
        feed_dict = {
                self.question:data[0],
                self.answer:data[1],
                self.answer_negative:data[2],
                # self.q_mask:data[3],
                # self.a_mask:data[4],
                # self.a_neg_mask:data[5],
                self.dropout_keep_prob_holder:self.dropout_keep_prob
            }

        _, summary, step, loss, accuracy,score12, score13, see = sess.run(
                    [self.train_op, self.merged,self.global_step,self.loss, self.accuracy,self.score12,self.score13, self.see],
                    feed_dict)
        return _, summary, step, loss, accuracy,score12, score13, see
    def predict(self,sess,data):
        feed_dict = {
                self.question:data[0],
                self.answer:data[1],
                # self.q_mask:data[2],
                # self.a_mask:data[3],
                self.dropout_keep_prob_holder:1.0
            }            
        score = sess.run( self.score12, feed_dict)       
        return score

    
if __name__ == '__main__':
    
    cnn = QA_CNN_extend(
        max_input_left = 33,
        max_input_right = 40,
        batch_size = 3,
        vocab_size = 5000,
        embedding_size = 100,
        filter_sizes = [3,4,5],
        num_filters = 64, 
        hidden_size = 100,
        dropout_keep_prob = 1.0,
        embeddings = None,
        l2_reg_lambda = 0.0,
        trainable = True,

        pooling = 'max',
        conv = 'wide')
    cnn.build_graph()
    input_x_1 = np.reshape(np.arange(3 * 33),[3,33])
    input_x_2 = np.reshape(np.arange(3 * 40),[3,40])
    input_x_3 = np.reshape(np.arange(3 * 40),[3,40])
    q_mask = np.ones((3,33))
    a_mask = np.ones((3,40))
    a_neg_mask = np.ones((3,40))
  

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            cnn.question:input_x_1,
            cnn.answer:input_x_2,
            # cnn.answer_negative:input_x_3,
            cnn.q_mask:q_mask,
            cnn.a_mask:a_mask,
            cnn.dropout_keep_prob_holder:cnn.dropout_keep
            # cnn.a_neg_mask:a_neg_mask
            # cnn.q_pos_overlap:q_pos_embedding,
            # cnn.q_neg_overlap:q_neg_embedding,
            # cnn.a_pos_overlap:a_pos_embedding,
            # cnn.a_neg_overlap:a_neg_embedding,
            # cnn.q_position:q_position,
            # cnn.a_pos_position:a_pos_position,
            # cnn.a_neg_position:a_neg_position
        }
        question,answer,score = sess.run([cnn.question,cnn.answer,cnn.score12],feed_dict)
        print( question.shape,answer.shape)
        print( score)


