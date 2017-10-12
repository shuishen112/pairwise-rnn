from tensorflow import flags
from main import main as m
import tensorflow as tf
# Model Hyperparameters
flags.DEFINE_integer("embedding_dim",300, "Dimensionality of character embedding (default: 128)")
flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
flags.DEFINE_float("dropout_keep_prob", 1, "Dropout keep probability (default: 0.5)")
flags.DEFINE_float("l2_reg_lambda", 0.000001, "L2 regularizaion lambda (default: 0.0)")
flags.DEFINE_float("learning_rate", 1e-3, "learn rate( default: 0.0)")
flags.DEFINE_integer("max_len_left", 40, "max document length of left input")
flags.DEFINE_integer("max_len_right", 40, "max document length of right input")
flags.DEFINE_string("loss","pair_wise","loss function (default:point_wise)")
flags.DEFINE_integer("hidden_size",100,"the default hidden size")

# Training parameters
flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
flags.DEFINE_boolean("trainable", False, "is embedding trainable? (default: False)")
flags.DEFINE_integer("num_epoches", 100, "Number of training epochs (default: 200)")
flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")

flags.DEFINE_string('data','wiki','data set')
flags.DEFINE_string('pooling','mean','max pooling or attentive pooling')
flags.DEFINE_boolean('clean',True,'whether we clean the data')
flags.DEFINE_string('conv','wide','wide conv or narrow')
flags.DEFINE_integer('gpu',0,'gpu number')
# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#data_help parameters


def test():
	args = flags.FLAGS
	m(args)

if __name__ == '__main__':
	test()
