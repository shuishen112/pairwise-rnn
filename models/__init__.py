from .QA_CNN_pairwise import QA_CNN_extend as CNN
from .QA_RNN_pairwise import QA_RNN_extend as RNN


def setup(opt):
	if opt["model_name"]=="cnn":
		model=CNN(opt)
	elif opt["model_name"]=="rnn":
		model=RNN(opt)
	else:
		print("no model")
		exit(0)
	return model
