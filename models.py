from keras.models import Sequential,  Model   ####good job
from keras.layers import Convolution2D,Input,BatchNormalization,Conv2D,Activation,Lambda,Subtract,Conv2DTranspose, PReLU,Conv1D
from keras.regularizers import l2
from keras.layers import  Reshape,Dense,Flatten
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from scipy.io import loadmat
import keras.backend as K
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import numpy as np
import math
from scipy import interpolate
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
#from scipy.misc import imresize

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_loss = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.train_loss['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, batch, logs={}):
        self.train_loss['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))

    def loss_plot(self, loss_type):
        iters = range(len(self.train_loss[loss_type]))
        plt.figure()
        # loss
        plt.semilogy(iters, self.train_loss[loss_type], 'blue', label='train loss')
        if loss_type == 'epoch':
            # val_loss
            plt.semilogy(iters, self.val_loss[loss_type], 'y', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def errors(y_true, y_pred):
    myOtherTensor = K.not_equal(y_true, K.round(y_pred))
    return K.mean(tf.cast(myOtherTensor, tf.float32))
def DNCNN_model (args):
    input_shape = (args.block_len, args.code_rate_k)
    inpt = Input(shape=input_shape)
    # 1st layer, Conv+relu
    x = Conv1D(filters=1, kernel_size=args.kernel_size, strides=1,padding='same')(inpt)
    x = Activation('relu')(x)
    # 18 layers, Conv+BN+relu
    for i in range(args.num_layer):
        x = Conv1D(filters=64, kernel_size=args.kernel_size, strides=1, padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv1D(filters=1, kernel_size=args.kernel_size, strides=1, padding='same')(x)
    x = Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    adam = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=[errors])  
    # model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_crossentropy'])     
    
    return model

def DNCNN_train(args,train_data ,train_label, val_data , val_label,channel_model,identity):
  
  dncnn_model = DNCNN_model(args)
  print(dncnn_model.summary())
  
  losslog=LossHistory()
  checkpoint = ModelCheckpoint("./tmp/weights_{epoch:02d}-{val_loss:.2f}.h5", monitor='val_loss', verbose=1, save_best_only=False,save_weights_only=True, mode='auto',period=10)
  callbacks_list = [checkpoint,losslog]

  dncnn_model.fit(train_data, train_label, batch_size=args.batch_size, validation_data=(val_data, val_label),callbacks=callbacks_list, shuffle=True, epochs= args.num_epoch , verbose=2)
  dncnn_model.save_weights("./tmp/DNCNN_" + channel_model +"_"+str(identity)+ ".h5")
  losslog.loss_plot('epoch')
  
  
  
def DNCNN_predict(args,input_data, channel_model ,identity ):
  dncnn_model = DNCNN_model(args)
  dncnn_model.load_weights("./tmp/DNCNN_" + channel_model +"_"+ str(identity)  + ".h5")
  predicted  = dncnn_model.predict(input_data)
  return predicted
  
  
