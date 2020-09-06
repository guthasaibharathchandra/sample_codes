import sys
import tensorflow as tf
import numpy as np
#from model import model,test_model
from encoder import encoder
from decoder import decoder,test_decoder
import horovod.tensorflow as hvd
#from scipy.signal import stft,istft
#from scipy.io import wavfile


def average_gradients(tower_grads):
 
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads,0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train(train_features,train_labels,valid_features,valid_labels,batch_size,input_vector_size,
          num_classes,hidden_state_size,num_layers,num_epochs,epoch_step,CUDNN,MYDTYPE,num_gpus,LR):
     
     
  decoder_features = np.roll(train_labels,shift=1,axis=1)
  decoder_features[:,0]=0
  n_b = len(train_features)//(batch_size)
  n_b = n_b//num_gpus
  trainlen = n_b*batch_size*num_gpus
    

  graph = tf.Graph()
  with graph.as_default():
        
    encoder_features_placeholder = tf.placeholder(dtype=tf.float64, shape=(None, None, input_vector_size), name="encoder_features_placeholder")
    decoder_features_placeholder = tf.placeholder(dtype=tf.float64, shape=(None, None, input_vector_size), name="decoder_features_placeholder")
    labels_placeholder = tf.placeholder(dtype=tf.float64, shape=(None, None, num_classes), name="labels_placeholder")    
    dataset1 = tf.data.Dataset.from_tensor_slices((encoder_features_placeholder,decoder_features_placeholder,labels_placeholder)).batch(batch_size).prefetch(8)
    iterator1 = dataset1.make_initializable_iterator()
    
    ###############
    #totalloss_list=[]
    #grads_and_vars_list=[]
    
    
    el1,el2,el3 = iterator1.get_next() #[batch_size,time,input_vector_size] float64 (el1) #[batch_size,time,num_classes] int16 (el2)        
    batchX_placeholder = el1 
    batchY_placeholder = el2  
    batchZ_placeholder = el3

    #with tf.device('/device:GPU:'+str(hvd.local_rank())):
    L = encoder(batchX_placeholder,batch_size,input_vector_size,hidden_state_size,num_layers,CUDNN,MYDTYPE)
    output,loss = decoder(L,batchY_placeholder,batchZ_placeholder,batch_size,2*hidden_state_size,num_classes,num_layers,CUDNN,MYDTYPE)
    #totalloss_list.append(loss)
    #globalstep = tf.train.get_or_create_global_step()    
    optimizer = hvd.DistributedOptimizer(tf.train.GradientDescentOptimizer(LR*hvd.size()))    
    trainstep = optimizer.minimize(loss)
    #grads_and_vars_list.append( optimizer.compute_gradients(loss) )                        
    #avg_grads_and_vars = average_gradients(grads_and_vars_list)
    #trainstep = optimizer.apply_gradients(avg_grads_and_vars)
    saver = tf.train.Saver(max_to_keep=10000)
    init = tf.global_variables_initializer()
    bcast = hvd.broadcast_global_variables(0)
  
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.visible_device_list = str(hvd.local_rank()) 

  with tf.Session(graph=graph,config=config) as sess:
    init.run()
    bcast.run()
    print('Initialized_'+str(hvd.rank()))
    sys.stdout.flush()
    
    for epoch in range(num_epochs):
    
        sess.run(iterator1.initializer,feed_dict={encoder_features_placeholder:train_features,decoder_features_placeholder:decoder_features,labels_placeholder:train_labels})
        total_loss  = 0
        for batch in range(n_b):   
             _,_loss = sess.run([trainstep,loss])
             total_loss += _loss      
    
        print("epoch = "+str(epoch+1)+", loss_"+str(hvd.rank())+" = "+str(total_loss/n_b))    
        sys.stdout.flush()     
        if (epoch+1)%epoch_step == 0 and hvd.rank()==0:
            saver.save(sess,"/group-volume/orc_srib/s.gutha/W2S_spectral_Seq2Seqmodel_contextual2/model",global_step=epoch+1)

    """
    hooks = [hvd.BroadcastGlobalVariablesHook(0),tf.train.StopAtStepHook(last_step = n_b), tf.train.LoggingTensorHook(tensors={'loss':loss,'step':globalstep},every_n_iter=1)]

    ckpt_dir = None
    if hvd.rank()==0:
        ckpt_dir = "/group-volume/orc_srib/s.gutha/W2S_spectral_Seq2Seqmodel_contextual2"

    with tf.train.MonitoredTrainingSession(checkpoint_dir=ckpt_dir,hooks=hooks,config=config) as sess:
          while not sess.should_stop():  
                sess.run(trainstep)

    """

    """
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
    
            _total_loss = 0
            sess.run(iterator1.initializer,feed_dict={encoder_features_placeholder:train_features,decoder_features_placeholder:decoder_features,labels_placeholder:train_labels})
                           
            for batch_id in range(n_b):
                _,t_losses = sess.run([trainstep,totalloss_list])
                for ind in range(num_gpus):
                        _total_loss += t_losses[ind]        
               
      
            print("train_loss = "+str(_total_loss/(n_b*num_gpus)))            
            sys.stdout.flush()
            
            if (epoch+1)%epoch_step == 0:
               saver.save(sess,"/group-volume/orc_srib/s.gutha/W2S_spectral_Seq2Seqmodel_contextual2/model",global_step=epoch+1)
       
    """


def retrain(train_dir,train_num_partitions,valid_dir,valid_num_partitions,batch_size,input_vector_size,num_classes,
        hidden_state_size,num_layers,num_epochs,epoch_step,model_weights_folder,ckpt_epoch,CUDNN,MYDTYPE,num_gpus):
    
    ###currently support only CUDNN import by building inference graph
    ##############    
   
    features_placeholder = tf.placeholder(dtype=tf.float64, shape=(None, None, input_vector_size), name="features_placeholder")
    labels_placeholder = tf.placeholder(dtype=tf.float64, shape=(None, None, num_classes), name="labels_placeholder")    
    dataset1 = tf.data.Dataset.from_tensor_slices((features_placeholder,labels_placeholder)).batch(batch_size).prefetch(8)
    iterator1 = dataset1.make_initializable_iterator()
  
    ##############
    
    optimizer = tf.train.GradientDescentOptimizer(5e-3)
    totalloss_list=[]
    grads_and_vars_list=[]
    
    
    for i in range(num_gpus):
        with tf.name_scope('tower_'+str(i)):
            
            el1,el2 = iterator1.get_next() #[batch_size,time,input_vector_size] float64 (el1) #[batch_size,time,num_classes] int16 (el2)
            
            batchX_placeholder = el1 
            batchY_placeholder = el2  
            
            with tf.device('/device:GPU:'+str(i)):
                
                _,loss = model(batchX_placeholder,batchY_placeholder,batch_size,input_vector_size,hidden_state_size,num_classes,num_layers,CUDNN,MYDTYPE)
                totalloss_list.append(loss)
                grads_and_vars_list.append( optimizer.compute_gradients(loss) )
                            
    avg_grads_and_vars = average_gradients(grads_and_vars_list)
    trainstep = optimizer.apply_gradients(avg_grads_and_vars)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)    
    saver = tf.train.Saver(max_to_keep=10000)
    saver.restore(sess,model_weights_folder+"model-"+str(ckpt_epoch))

    for epoch in range(num_epochs):

            print("epoch_"+str(ckpt_epoch+epoch+1)+":")
            epoch_loss = 0            

            for partition_id in range(1,train_num_partitions+1):
                train_features,train_labels = None,None
                train_features,train_labels = np.load(train_dir+"features_partition_"+str(partition_id)+".npy"),np.load(train_dir+"labels_partition_"+str(partition_id)+".npy")
                n_b = len(train_features)//(batch_size)
                n_b = n_b//num_gpus
                trainlen = n_b*batch_size*num_gpus
                
                _total_loss = 0
                valid_total_loss = 0
                sess.run(iterator1.initializer,feed_dict={features_placeholder:train_features,labels_placeholder:train_labels})
                           
                for batch_id in range(n_b):  
                
                    _,t_losses = sess.run([trainstep,totalloss_list])
                    for ind in range(num_gpus):
                        _total_loss += t_losses[ind]        
               
                partition_loss = np.sqrt(_total_loss/(num_gpus*n_b))
                epoch_loss += partition_loss
                print("         partition = "+str(partition_id)+", partition_train_loss = "+str(partition_loss) )
                sys.stdout.flush()
      
            print("epoch_train_loss = "+str(epoch_loss/train_num_partitions) )
            sys.stdout.flush()
            
            if (epoch+1)%epoch_step == 0:
               #saver.save(sess,"/gpfs-volume/e2e_whisper_to_normal_speech/model",global_step=epoch+1)
               #saver.save(sess,"/group-volume/orc_srib/s.gutha/mymodelweights/model",global_step=epoch+1)
               saver.save(sess,"/group-volume/orc_srib/s.gutha/W2S_spectral_RNNmodel_contextual2/model",global_step=ckpt_epoch+epoch+1)
   

def func1(file):
    rate,data = wavfile.read(file)
    f,t,Zxx  = stft(data,nperseg=512,nfft=2048)  
    return np.transpose(Zxx)

def func2(arr):
    arr = np.transpose(arr) 
    t,x  = istft(arr,nperseg=512,nfft=2048)  
    return np.asarray(x,dtype=np.int16)

def test(file,batch_size,input_vector_size,num_classes,hidden_state_size,num_layers,model_weights_folder,ckpt_epoch,CUDNN,MYDTYPE):

    ###currently support only CUDNN import by building inference graph

    batchX_placeholder = tf.placeholder(dtype=tf.float64, shape=(batch_size, None, input_vector_size), name="features_placeholder")
    initial_state_placeholder_c = tf.placeholder(dtype=tf.float64, shape=(num_layers,batch_size, hidden_state_size), name="initial_state_c_placeholder")
    initial_state_placeholder_h = tf.placeholder(dtype=tf.float64, shape=(num_layers,batch_size, hidden_state_size), name="initial_state_h_placeholder")    
    initial_state_placeholder = (initial_state_placeholder_c,initial_state_placeholder_h)
    y,state = test_model(batchX_placeholder,initial_state_placeholder,batch_size,input_vector_size,hidden_state_size,num_classes,num_layers,CUDNN,MYDTYPE)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    sess=tf.Session(config=config)
    saver = tf.train.Saver()    
    saver.restore(sess,model_weights_folder+"model-"+str(ckpt_epoch))
    

    output=[np.zeros(1025,dtype=np.complex64)]  

    F = func1(file) # time,1025..
    

    for time in range(len(F)):
        if time==0:
            tempfeat = np.transpose(np.concatenate([np.zeros((1,1025),dtype=np.complex64),F[time:time+2]],axis=0))

        elif time==len(F)-1:
            tempfeat = np.transpose(np.concatenate([F[time-1:time+1],np.zeros((1,1025),dtype=np.complex64)],axis=0))

        else:
            tempfeat = np.transpose(F[time-1:time+2]) # 1025,3  

        tempfeat = np.roll(np.flip(tempfeat,axis=0),shift=1,axis=0)# 1025,3
        #tempfeat = np.roll(tempfeat,shift=1,axis=0)        
        tempfeat[0] = 0
        original_feat = tempfeat[513:]        
        tempfeat = tempfeat[:513]           
        myfeatures = np.reshape(np.concatenate([np.real(tempfeat),np.imag(tempfeat)],axis=1),(1,513,6))
        original_feat = np.reshape(np.concatenate([np.real(original_feat),np.imag(original_feat)],axis=1),(512,6))
        output.append(F[time])

        _y,_state = sess.run([y,state],feed_dict={batchX_placeholder:myfeatures,initial_state_placeholder_c:np.zeros((num_layers,batch_size,hidden_state_size),dtype=np.float),initial_state_placeholder_h:np.zeros((num_layers,batch_size,hidden_state_size),dtype=np.float)})
        output[-1][512] = np.complex(_y[0][-1][0],_y[0][-1][1])        
        #print(type(_state))
        #print(_state)     
        for i in range(512):
            #vec  = np.asarray([original_feat[i][0],_y[0][-1][0],original_feat[i][2],original_feat[i][3],_y[0][-1][1],original_feat[i][5]],dtype=np.float)
            vec  = np.asarray([np.real(output[-2][512-i]),_y[0][-1][0],original_feat[i][2],np.imag(output[-2][512-i]),_y[0][-1][1],original_feat[i][5]],dtype=np.float)
            _y,_state = sess.run([y,state],feed_dict={batchX_placeholder:np.reshape(vec,(1,1,6)),initial_state_placeholder_c:_state[0],initial_state_placeholder_h:_state[1]})
            
            output[-1][512-i-1] = np.complex(_y[0][-1][0],_y[0][-1][1])
    
    wavfile.write('enhanced_'+file,16000,func2(np.asarray(output,dtype=np.complex64)))
    #np.save('/group-volume/orc_srib/s.gutha/out_samples',np.asarray(output,dtype=np.complex64))            



            