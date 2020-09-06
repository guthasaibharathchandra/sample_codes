import sys
import pickle
from procedure import train,retrain,test
import numpy as np
import tensorflow as tf
#arguments declaration
import horovod.tensorflow as hvd

hvd.init()

mode = n_layers = batch_size = hidden_state_size = input_vector_size = 0 
num_epochs = num_classes = epoch_step = n_b = v_b = 0
model_weights_folder  = ckpt_epoch = 0

arg_index = 1
CUDNN = False
MYDTYPE = tf.float32

mode = sys.argv[arg_index] # train or retrain or test................
arg_index+=1

if mode == "train":    

    num_gpus = int(sys.argv[arg_index])
    arg_index += 1

    num_layers = int(sys.argv[arg_index]) # number of stacked layers/depth........
    arg_index+=1
    
    batch_size = int(sys.argv[arg_index]) # batch_size.............
    arg_index+=1
    
    input_vector_size = int(sys.argv[arg_index])
    arg_index+=1
    
    hidden_state_size = int(sys.argv[arg_index])
    arg_index+=1
    
    num_classes = int(sys.argv[arg_index])
    arg_index+=1
    
    num_epochs = int(sys.argv[arg_index]) # epochs..............
    arg_index+=1
  
    epoch_step = int(sys.argv[arg_index]) # epoch_save_step.............
    arg_index+=1
    
    #train_dir = sys.argv[arg_index]
    #arg_index+=1

    #train_num_partitions = int(sys.argv[arg_index])
    #arg_index+=1

    #features_file = sys.argv[arg_index] # train_features..............
    #arg_index+=1

    #labels_file = sys.argv[arg_index] # train_labels..............
    #arg_index+=1      
    
    train_dir =  sys.argv[arg_index]
    arg_index += 1   


    #train_seqlen_file = sys.argv[arg_index]
    #arg_index+=1
    
    
    #val_features_file = sys.argv[arg_index]# val_features
    #valid_dir = sys.argv[arg_index]    
    #arg_index+=1      
      
    #val_labels_file = sys.argv[arg_index] # val_labels...........
    #valid_num_partitions = int(sys.argv[arg_index])    
    #arg_index+=1
    
    #valid_seqlen_file = sys.argv[arg_index]
    #arg_index+=1
        
    use_cudnn = sys.argv[arg_index]     
    arg_index+=1

    if use_cudnn == "True":
       CUDNN = True

    mydtype = sys.argv[arg_index]
    arg_index+=1
    
    if mydtype == "tf.float64":
       MYDTYPE = tf.float64
   
      
else:

    ########testing/retrain

    if mode!="test":
       num_gpus = int(sys.argv[arg_index])
       arg_index += 1
    
    num_layers = int(sys.argv[arg_index])
    arg_index+=1
    
    batch_size = int(sys.argv[arg_index])
    arg_index+=1
    
    input_vector_size = int(sys.argv[arg_index])
    arg_index+=1

    hidden_state_size = int(sys.argv[arg_index])
    arg_index+=1

    num_classes = int(sys.argv[arg_index])
    arg_index+=1 
    
    if mode=="test":
        features_file = sys.argv[arg_index]
        arg_index+=1
    
    model_weights_folder = sys.argv[arg_index]
    arg_index+=1
    
    ckpt_epoch = int(sys.argv[arg_index])
    arg_index+=1
    
    use_cudnn = sys.argv[arg_index]     
    arg_index+=1

    if use_cudnn == "True":
       CUDNN = True
       
    mydtype = sys.argv[arg_index]
    arg_index+=1
    
    if mydtype == "tf.float64":
       MYDTYPE = tf.float64   

if mode=="retrain":
      
    ########retrain
    train_dir = sys.argv[arg_index]
    arg_index+=1

    train_num_partitions = int(sys.argv[arg_index])
    arg_index+=1

    valid_dir = sys.argv[arg_index] # validation_features_file...........
    arg_index+=1
    
    valid_num_partitions = int(sys.argv[arg_index]) # validation_labels_file...........
    arg_index+=1
    
    num_epochs = int(sys.argv[arg_index]) # epochs..............
    arg_index+=1
    
    epoch_step = int(sys.argv[arg_index]) # epoch_save_step.............
    arg_index+=1
    
########################################
# loading the input features and labels
 
LR = float(sys.argv[arg_index])
arg_index+=1
   

if mode!="test":
    pass
    #valid_dir = valid_num_partitions = valid_features = valid_labels = v_b = valid_seqlen = 0
    #start = hvd.rank()*batch_size*3    
    #end = start + batch_size*3    
    train_features = np.load(train_dir+'features_'+str(hvd.rank())+'.npy')
    train_labels = np.load(train_dir+'labels_'+str(hvd.rank())+'.npy')
    train_features = np.reshape(train_features,(len(train_features),513,1))
    train_labels = np.reshape(train_labels,(len(train_labels),512,1))
    valid_features = 0
    valid_labels = 0
    num_gpus = 1    
    """
    valid_features = np.load(val_features_file)
    valid_labels = np.load(val_labels_file)
    V_data = len(valid_features)
    v_b = V_data//batch_size 
    validlen = v_b*batch_size
    V_data = validlen
    valid_seqlen_enc = np.load(valid_seqlen_enc_file)
    valid_fseqlen_dec = np.load(valid_fseqlen_dec_file)
    """
    ########################################


    
if mode=="train":    
    
    train(train_features,train_labels,valid_features,valid_labels,batch_size,input_vector_size,num_classes,
         hidden_state_size,num_layers,num_epochs,epoch_step,CUDNN,MYDTYPE,num_gpus,LR)
          
elif mode=="retrain":

    retrain(train_dir,train_num_partitions,valid_dir,valid_num_partitions,batch_size,input_vector_size,num_classes,
        hidden_state_size,num_layers,num_epochs,epoch_step,model_weights_folder,ckpt_epoch,CUDNN,MYDTYPE,num_gpus)
      
else:
  
    test(features_file,batch_size,input_vector_size,num_classes,hidden_state_size,num_layers,model_weights_folder,ckpt_epoch,CUDNN,MYDTYPE)
