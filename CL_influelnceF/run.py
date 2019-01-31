from my_imdb import textCL_model, load_imdb
import tensorflow as tf
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import experiments2 as experiments

num_epochs=50
batch_size=500
weight_decay=0.001
num_classes=2
initial_learning_rate=0.01
damping=1e-2
keep_probs = [1.0, 1.0]
decay_epochs = [10000, 20000]
dense1_unit=16
maxlen=256
num_words=10000
vocab_size=10000

data=load_imdb(maxlen=256,num_words=10000)

model=textCL_model(
    weight_decay = weight_decay,
	initial_learning_rate=initial_learning_rate,
	num_classes=num_classes, 
	maxlen=maxlen,	
	vocab_size=vocab_size,
	data_sets=data,
	num_epochs=num_epochs,
	batch_size=batch_size,
	decay_epochs=decay_epochs,			
	dense1_unit=dense1_unit,	
	train_dir='output', 
	log_dir='log',
	model_name='imdb_mymodel',
	damping=1e-2)

num_steps = 400000

mode='all'
if mode=='all':
	model.train(num_steps=num_steps)

iter_to_load = num_steps -1

test_idx = 6558

if mode=='all':
	known_indices_to_remove=[]
else:
	f=np.load('output/imdb_mymodel_'+str(num_steps)+'_retraining_3pts.npz')
	known_indices_to_remove=f['indices_to_remove']


actual_loss_diffs, predicted_loss_diffs, indices_to_remove = experiments.test_retraining(
	model, 
	test_idx=test_idx, 
	iter_to_load=iter_to_load, 
	num_to_remove=10,
	num_steps=30000, 
	remove_type='maxinf',
	known_indices_to_remove=known_indices_to_remove,
	force_refresh=True)

filename="imdb_mymodel"+str(num_steps)+"_"+mode+".txt"
np.savetxt(filename, np.c_[actual_loss_diffs,predicted_loss_diffs],fmt ='%f6')

if mode=="all":
	np.savez(
		'output/imdb_mymodel_'+str(num_steps)+'_retraining_3pts.npz', 
		actual_loss_diffs=actual_loss_diffs, 
		predicted_loss_diffs=predicted_loss_diffs, 
		indices_to_remove=indices_to_remove
	)





























