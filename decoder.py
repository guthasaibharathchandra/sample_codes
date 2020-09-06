import tensorflow as tf
import numpy as np

MYDTYPE = DELTA = 0

def get_context(Y,X,batch_size,hidden_state_size):

    """
    # transpose(X) [batch_size,hidden_state_size,time]
    slope = 0.001    
    unnormalized_scores = tf.matmul(Y,tf.transpose(X,perm=[0,2,1])) # [batch,time,time]
    scores = tf.maximum( 1.0 - ( slope*tf.multiply(unnormalized_scores,tf.sign(unnormalized_scores)) ),DELTA)
    normalized_scores = tf.divide( scores, tf.reduce_sum(scores,axis=-1,keepdims=True)) # [batch,time,time]
    return tf.reduce_sum(tf.multiply(tf.expand_dims(normalized_scores,axis=-1),tf.reshape(X,shape=[batch_size,1,-1,hidden_state_size])),axis=-2)
    """

    """
    with tf.device('/cpu:0'):
        W1 = tf.get_variable(name="W1",shape=[hidden_state_size,1],dtype=MYDTYPE)        
        W2 = tf.get_variable(name="W2",shape=[hidden_state_size,2*hidden_state_size],dtype=MYDTYPE)
        
    unnormalized_scores = tf.reshape(tf.matmul(tf.reshape(Y,shape=[-1,hidden_state_size]),W1),shape=[batch_size,-1])
    scores = tf.maximum( 1.0 - ( slope*tf.multiply(unnormalized_scores,tf.sign(unnormalized_scores)) ),DELTA)
    normalized_scores = tf.divide( scores, tf.reduce_sum(scores,axis=-1,keepdims=True)) #[batch_size,maxtime]
        
    values = tf.reshape(tf.matmul(tf.reshape(Y,shape=[-1,hidden_state_size]),W2),shape=[batch_size,-1,2*hidden_state_size]) #[batch_size,maxtime,2*hidden_state_size]
    C = tf.reduce_sum(tf.multiply( values,tf.expand_dims(normalized_scores,axis=-1) ),axis=1) #[batch_size,2*hidden_state_size]
    return C #[batch_size,2*hidden_state_size]
    """

    #with tf.device('/cpu:0'):
    Adec = tf.get_variable(name="Adec",shape=[hidden_state_size,hidden_state_size],dtype=MYDTYPE)        
    Aenc = tf.get_variable(name="Aenc",shape=[hidden_state_size,hidden_state_size],dtype=MYDTYPE)
    Aweight = tf.get_variable(name="Aweight",shape=[hidden_state_size],dtype=MYDTYPE)
    
    #Aenc_norm = tf.sqrt(tf.reduce_sum(Aenc*Aenc))
    #Adec_norm = tf.sqrt(tf.reduce_sum(Adec*Adec))
    Aweight_norm = tf.sqrt(tf.reduce_sum(Aweight*Aweight))

    #ENCSCORES = tf.reshape(tf.divide(tf.reshape(tf.matmul(tf.reshape(X,shape=[-1,hidden_state_size]),Aenc),shape=[batch_size,-1]),tf.reshape(Aenc_norm,shape=[1,1])),shape=[batch_size,1,-1]) #[batch_size,1,time_enc]
    #DECSCORES = tf.reshape(tf.divide(tf.reshape(tf.matmul(tf.reshape(Y,shape=[-1,hidden_state_size]),Adec),shape=[batch_size,-1]),tf.reshape(Adec_norm,shape=[1,1])),shape=[batch_size,-1,1]) #[batch_size,time_dec,1]
    #ENCWEIGHTS = tf.nn.sigmoid(tf.reshape(tf.divide(tf.reshape(tf.matmul(tf.reshape(X,shape=[-1,hidden_state_size]),Aweight),shape=[batch_size,-1]),tf.reshape(Aweight_norm,shape=[1,1])),shape=[batch_size,1,-1])) #[batch_size,1,time_enc]

    ENCSCORES = tf.reshape(tf.matmul(tf.reshape(X,shape=[-1,hidden_state_size]),Aenc),shape=[batch_size,1,-1,hidden_state_size]) #[batch_size,1,time_enc,hidden_state_size]
    DECSCORES = tf.reshape(tf.matmul(tf.reshape(Y,shape=[-1,hidden_state_size]),Adec),shape=[batch_size,-1,1,hidden_state_size]) #[batch_size,time_dec,1,hidden_state_size]
    
    SCORES = tf.nn.tanh(tf.add(ENCSCORES,DECSCORES)) #[batch_size,time_dec,time_enc,hidden_state_size]
    unnormalized_scores = tf.divide(tf.tensordot(SCORES,Aweight,axes=[[-1],[0]]),tf.reshape(Aweight_norm,shape=[1,1,1]))

    #unnormalized_scores = tf.nn.tanh(tf.add(tf.matmul(tf.ones_like(DECSCORES),ENCSCORES),DECSCORES)) #[batch_size,time_dec,time_enc]
    #unnormalized_scores = tf.add((ENCWEIGHTS*tf.matmul(tf.ones_like(DECSCORES),ENCSCORES)),((1-ENCWEIGHTS)*DECSCORES)) #[batch_size,time_dec,time_enc]    
    normalized_scores = tf.expand_dims(tf.divide(tf.exp(unnormalized_scores),tf.reduce_sum(unnormalized_scores,axis=-1,keepdims=True)),axis=-1) #[batch_size,time_dec,time_enc,1]
    C = tf.reduce_sum(tf.multiply(normalized_scores,tf.reshape(X,shape=[batch_size,1,-1,hidden_state_size])),axis=-2) #[batch_size,time_dec,hidden_state_size]
    return C
    
def decoder(encoder_output,input,labels,batch_size,hidden_state_size,num_classes,num_layers,CUDNN,MYDTYPE_):
   
        global MYDTYPE,DELTA
        MYDTYPE = MYDTYPE_
        if MYDTYPE == tf.float32:
            DELTA = 1e-30
        elif MYDTYPE == tf.float64:
            DELTA = 1e-300
        
   
        with tf.variable_scope("Decoder",reuse=tf.AUTO_REUSE):

                            I = input
                            if CUDNN:
                                input_series = tf.transpose(I,perm=[1,0,2])            
                            else:
                                input_series = I #[batch_size,maxtime,hidden_state_size]    
                                
                            
                            if CUDNN:                  
                                            lstm_fw = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=num_layers,num_units=hidden_state_size,dtype=MYDTYPE)
                                            Y_T_fw, state_fw = lstm_fw(inputs=input_series)
                                            Y_fw = tf.transpose(Y_T_fw,perm=[1,0,2])
                                            
                            else:
                                            cell_fw_layers = [tf.nn.rnn_cell.LSTMCell(hidden_state_size,state_is_tuple=True) for i in range(num_layers)]
                                            cell_fw = tf.nn.rnn_cell.MultiRNNCell(cell_fw_layers)
                                            Y_fw_temp, state_fw = tf.nn.dynamic_rnn(cell=cell_fw,inputs=input_series,dtype=MYDTYPE) 
                                            Y_fw = Y_fw_temp                                                
                                                    
                            #encoder_output[batch_size,time,hidden_state_size_dec] hidden_state_size_dec = 2*hidden_state_size_enc
                            #Y_fw [batch_size,time,hidden_state_size]
 
                            enc_dec_vec = get_context(Y_fw,encoder_output,batch_size,hidden_state_size) #[batch_size,time_dec,hidden_state_size]
                            output = tf.concat([enc_dec_vec,Y_fw],axis=-1)
    
                            #with tf.device('/cpu:0'):
                            W1 = tf.get_variable(name="W1",shape=[2*hidden_state_size,hidden_state_size],dtype=MYDTYPE)
                            b1 = tf.get_variable(name="b1",shape=[hidden_state_size],dtype=MYDTYPE)        
                            W2 = tf.get_variable(name="W2",shape=[hidden_state_size,num_classes],dtype=MYDTYPE)
                            b2 = tf.get_variable(name="b2",shape=[num_classes],dtype=MYDTYPE)
    
                            C = tf.reshape(tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(tf.reshape(output,shape=[-1,2*hidden_state_size]),W1),b1)),W2),b2),shape=[batch_size,-1,num_classes])
                            T = C                            
                            #T = tf.add(C,encoder_input)
                            DIFF = labels - T
                            RMSE = tf.sqrt(tf.reduce_mean(tf.multiply(DIFF,DIFF)))
                            return T,RMSE




def test_decoder(initial_state,encoder_output,input,labels,batch_size,hidden_state_size,num_classes,num_layers,CUDNN,MYDTYPE_):
   
        global MYDTYPE,DELTA
        MYDTYPE = MYDTYPE_
        if MYDTYPE == tf.float32:
            DELTA = 1e-30
        elif MYDTYPE == tf.float64:
            DELTA = 1e-300
        
   
        with tf.variable_scope("Decoder",reuse=tf.AUTO_REUSE):

                            I = input
                            if CUDNN:

                                input_series = tf.transpose(I,perm=[1,0,2])            

                            else:

                                input_series = I #[batch_size,maxtime,hidden_state_size]    

                                        
                            if CUDNN:                  

                                            lstm_fw = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=num_layers,num_units=hidden_state_size,dtype=MYDTYPE)
                                            Y_T_fw, state_fw = lstm_fw(inputs=input_series,initial_state=initial_state)
                                            Y_fw = tf.transpose(Y_T_fw,perm=[1,0,2])

                            else:
                                            cell_fw_layers = [tf.nn.rnn_cell.LSTMCell(hidden_state_size,state_is_tuple=True) for i in range(num_layers)]
                                            cell_fw = tf.nn.rnn_cell.MultiRNNCell(cell_fw_layers)
                                            Y_fw_temp, state_fw = tf.nn.dynamic_rnn(cell=cell_fw,inputs=input_series,dtype=MYDTYPE) 
                                            Y_fw = Y_fw_temp                                                
                                                    
                            #encoder_output[batch_size,time,hidden_state_size] hidden_state_size_dec = 2*hidden_state_size_enc
                            
                            enc_dec_vec = get_context(Y_fw,encoder_output,batch_size,hidden_state_size)
                            output = tf.concat([enc_dec_vec,Y_fw],axis=-1)

    
                            with tf.device('/cpu:0'):
                                W1 = tf.get_variable(name="W1",shape=[2*hidden_state_size,hidden_state_size],dtype=MYDTYPE)
                                b1 = tf.get_variable(name="b1",shape=[hidden_state_size],dtype=MYDTYPE)        
                                W2 = tf.get_variable(name="W2",shape=[hidden_state_size,num_classes],dtype=MYDTYPE)
                                b2 = tf.get_variable(name="b2",shape=[num_classes],dtype=MYDTYPE)
    
                            C = tf.reshape(tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(tf.reshape(output,shape=[-1,2*hidden_state_size]),W1),b1)),W2),b2),shape=[batch_size,-1,num_classes])
                            
                            return C,state_fw


