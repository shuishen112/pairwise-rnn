class Singleton(object):
    __instance=None
    def __init__(self):
        pass
    def getInstance(self):
        if Singleton.__instance is None:
            # Singleton.__instance=object.__new__(cls,*args,**kwd)
            Singleton.__instance=self.get_test_flag()
            print("build FLAGS over")
        return Singleton.__instance
    def get_test_flag(self):
        import tensorflow as tf
        flags = tf.app.flags
        if len(flags.FLAGS.__dict__.keys())<=2:

            flags.DEFINE_integer("embedding_size",300, "Dimensionality of character embedding (default: 128)")
            flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
            flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
            flags.DEFINE_float("dropout_keep_prob", 1, "Dropout keep probability (default: 0.5)")
            flags.DEFINE_float("l2_reg_lambda", 0.000001, "L2 regularizaion lambda (default: 0.0)")
            flags.DEFINE_float("learning_rate", 5e-3, "learn rate( default: 0.0)")
            flags.DEFINE_integer("max_len_left", 40, "max document length of left input")
            flags.DEFINE_integer("max_len_right", 40, "max document length of right input")
            flags.DEFINE_string("loss","pair_wise","loss function (default:point_wise)")
            flags.DEFINE_integer("hidden_size",100,"the default hidden size")
            flags.DEFINE_string("model_name", "cnn", "cnn or rnn")

            # Training parameters
            flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
            flags.DEFINE_boolean("trainable", False, "is embedding trainable? (default: False)")
            flags.DEFINE_integer("num_epoches", 1000, "Number of training epochs (default: 200)")
            flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
            flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")

            flags.DEFINE_string('data','wiki','data set')
            flags.DEFINE_string('pooling','max','max pooling or attentive pooling')
            flags.DEFINE_boolean('clean',True,'whether we clean the data')
            flags.DEFINE_string('conv','wide','wide conv or narrow')
            flags.DEFINE_integer('gpu',0,'gpu number')
            # Misc Parameters
            flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
            flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
        return flags.FLAGS
    def get_rnn_flag(self):
        import tensorflow as tf
        flags = tf.app.flags
        if len(flags.FLAGS.__dict__.keys())<=2:

            flags.DEFINE_integer("embedding_size",300, "Dimensionality of character embedding (default: 128)")
            flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
            flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
            flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
            flags.DEFINE_float("l2_reg_lambda", 0.000001, "L2 regularizaion lambda (default: 0.0)")
            flags.DEFINE_float("learning_rate", 5e-3, "learn rate( default: 0.0)")
            flags.DEFINE_integer("max_len_left", 40, "max document length of left input")
            flags.DEFINE_integer("max_len_right", 40, "max document length of right input")
            flags.DEFINE_string("loss","pair_wise","loss function (default:point_wise)")
            flags.DEFINE_integer("hidden_size",100,"the default hidden size")
            flags.DEFINE_string("model_name", "rnn", "cnn or rnn")

            # Training parameters
            flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
            flags.DEFINE_boolean("trainable", False, "is embedding trainable? (default: False)")
            flags.DEFINE_integer("num_epoches", 1000, "Number of training epochs (default: 200)")
            flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
            flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
            flags.DEFINE_string('self_att_logit_func',"tri_linear","logic function")
            flags.DEFINE_float('wd',0.0,"weight decay")
            flags.DEFINE_boolean("self_att_fuse_gate_residual_conn",True,"fuse gate")
            flags.DEFINE_boolean("self_att_fuse_gate_relu_z",True,"fuse gate")
            flags.DEFINE_boolean("two_gate_fuse_gate",True,"two_gate")

            flags.DEFINE_string('data','wiki','data set')
            flags.DEFINE_string('pooling','max','max pooling or attentive pooling')
            flags.DEFINE_boolean('clean',True,'whether we clean the data')
            flags.DEFINE_string('conv','wide','wide conv or narrow')
            flags.DEFINE_integer('gpu',0,'gpu number')
            # Misc Parameters
            flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
            flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
        return flags.FLAGS
    def get_cnn_flag(self):
        import tensorflow as tf
        flags = tf.app.flags
        if len(flags.FLAGS.__dict__.keys())<=2:

            flags.DEFINE_integer("embedding_size",300, "Dimensionality of character embedding (default: 128)")
            flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
            flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
            flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
            flags.DEFINE_float("l2_reg_lambda", 0.000001, "L2 regularizaion lambda (default: 0.0)")
            flags.DEFINE_float("learning_rate", 5e-3, "learn rate( default: 0.0)")
            flags.DEFINE_integer("max_len_left", 40, "max document length of left input")
            flags.DEFINE_integer("max_len_right", 40, "max document length of right input")
            flags.DEFINE_string("loss","pair_wise","loss function (default:point_wise)")
            flags.DEFINE_integer("hidden_size",100,"the default hidden size")
            flags.DEFINE_string("model_name", "cnn", "cnn or rnn")

            # Training parameters
            flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
            flags.DEFINE_boolean("trainable", False, "is embedding trainable? (default: False)")
            flags.DEFINE_integer("num_epoches", 1000, "Number of training epochs (default: 200)")
            flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
            flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")

            flags.DEFINE_string('data','wiki','data set')
            flags.DEFINE_string('pooling','max','max pooling or attentive pooling')
            flags.DEFINE_boolean('clean',True,'whether we clean the data')
            flags.DEFINE_string('conv','wide','wide conv or narrow')
            flags.DEFINE_integer('gpu',0,'gpu number')
            # Misc Parameters
            flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
            flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    def get_qcnn_flag(self):
        import tensorflow as tf
        flags = tf.app.flags
        if len(flags.FLAGS.__dict__.keys())<=2:

            flags.DEFINE_integer("embedding_size",300, "Dimensionality of character embedding (default: 128)")
            flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
            flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
            flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
            flags.DEFINE_float("l2_reg_lambda", 0.000001, "L2 regularizaion lambda (default: 0.0)")
            flags.DEFINE_float("learning_rate", 5e-3, "learn rate( default: 0.0)")
            flags.DEFINE_integer("max_len_left", 40, "max document length of left input")
            flags.DEFINE_integer("max_len_right", 40, "max document length of right input")
            flags.DEFINE_string("loss","pair_wise","loss function (default:point_wise)")
            flags.DEFINE_integer("hidden_size",100,"the default hidden size")
            flags.DEFINE_string("model_name", "qcnn", "cnn or rnn")

            # Training parameters
            flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
            flags.DEFINE_boolean("trainable", False, "is embedding trainable? (default: False)")
            flags.DEFINE_integer("num_epoches", 1000, "Number of training epochs (default: 200)")
            flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
            flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")

            flags.DEFINE_string('data','wiki','data set')
            flags.DEFINE_string('pooling','product','max pooling or attentive pooling')
            flags.DEFINE_boolean('clean',True,'whether we clean the data')
            flags.DEFINE_string('conv','wide','wide conv or narrow')
            flags.DEFINE_integer('gpu',0,'gpu number')
            # Misc Parameters
            flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
            flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
        return flags.FLAGS



if __name__=="__main__":
    args=Singleton().get_test_flag()
    for attr, value in sorted(args.__flags.items()):
        print(("{}={}".format(attr.upper(), value)))
   