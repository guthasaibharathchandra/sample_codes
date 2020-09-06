import tensorflow as tf
import numpy as np

MYDTYPE = DELTA = 0

def encoder(input,batch_size,input_vector_size,hidden_state_size,num_layers,CUDNN,MYDTYPE_):
        
        global MYDTYPE,DELTA
        MYDTYPE = MYDTYPE_
        if MYDTYPE == tf.float32:
            DELTA = 1e-30
        elif MYDTYPE == tf.float64:
            DELTA = 1e-300
        
        I = input
        
        if CUDNN:
            input_series = tf.transpose(I,perm=[1,0,2])
            reverse_input_series = tf.reverse_v2(input_series,[0])    
        else:
            input_series = I #[batch_size,maxtime,hidden_state_size]
            reverse_input_series = tf.reverse_v2(input_series,[1]) 
        
        with tf.variable_scope("Encoder",reuse=tf.AUTO_REUSE):
   
                    with tf.variable_scope("forward_layer"):
              
                           if CUDNN:                  
                                            lstm_fw = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=num_layers,num_units=hidden_state_size,dtype=MYDTYPE)
                                            Y_T_fw, state_fw = lstm_fw(inputs=input_series)
                                            Y_fw = tf.transpose(Y_T_fw,perm=[1,0,2])
                                            input_series =  Y_T_fw                                                       
                    
                           else:
                                            cell_fw_layers = [tf.nn.rnn_cell.LSTMCell(hidden_state_size,state_is_tuple=True) for i in range(num_layers)]
                                            cell_fw = tf.nn.rnn_cell.MultiRNNCell(cell_fw_layers)
                                            Y_fw_temp, state_fw = tf.nn.dynamic_rnn(cell=cell_fw,inputs=input_series,dtype=MYDTYPE) 
                                            Y_fw = Y_fw_temp                                                
                                            input_series = Y_fw_temp                                                    
                     
                    with tf.variable_scope("backward_layer"):
                      
                           if CUDNN:                  
                                            lstm_bw = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=num_layers,num_units=hidden_state_size,dtype=MYDTYPE)
                                            Y_T_bw, state_bw = lstm_bw(inputs=reverse_input_series)
                                            Y_bw = tf.transpose(tf.reverse_v2(Y_T_bw,[0]),perm=[1,0,2])
                         
                           else:
                                            cell_bw_layers = [tf.nn.rnn_cell.LSTMCell(hidden_state_size,state_is_tuple=True) for i in range(num_layers)]
                                            cell_bw = tf.nn.rnn_cell.MultiRNNCell(cell_bw_layers)
                                            Y_bw_temp, state_bw = tf.nn.dynamic_rnn(cell=cell_bw,inputs=reverse_input_series,dtype=MYDTYPE) 
                                            Y_bw = tf.reverse_v2(Y_bw_temp,[1])
                    
                    L = tf.concat([Y_fw,Y_bw],axis=-1)
                    return L