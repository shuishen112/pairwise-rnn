from .QA_CNN_pairwise import QA_CNN_extend as CNN
from .QA_RNN_pairwise import QA_RNN_extend as RNN
from .QA_CNN_quantum_pairwise import QA_CNN_extend as QCNN
def setup(opt):
	if opt["model_name"]=="cnn":
		model=CNN(opt)
	elif opt["model_name"]=="rnn":
		model=RNN(opt)
	elif opt['model_name']=='qcnn':
		model=QCNN(opt)
	else:
		print("no model")
		exit(0)
	return model
