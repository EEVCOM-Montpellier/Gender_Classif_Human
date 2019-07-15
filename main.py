# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:34:18 2019

@author: sonia mai tieo
"""


#--- IMPORTATION LIBRAIRIES
import sys
import time, datetime, os, math, csv, re, gc, keract, json
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import itertools
import argparse

from keras import backend as K
import tensorflow
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.preprocessing import image
from keras.utils import np_utils
from keras.models import model_from_json, load_model,  Model
from keras import optimizers
from keras.layers import Input
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
AveragePooling2D, Reshape, Permute, multiply, Dropout
from keras.callbacks import Callback
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, CSVLogger

from keract import get_activations

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle


#--- GLOBAL VAR
inputShape = (224, 224)
num_classes = 1
FREEZE = 7
BATCH_SIZE = 32
NB_EPOCHS = 10
NB_EPOCHS2 = 5


#--- SAVE PARAMETERS and INFOS
today =  datetime.date.today()
now= datetime.datetime.now()

#create_folder for results
todaystr = today.isoformat() + '-' + str(now.hour) + '-' + str(now.minute)
os.mkdir("results/"   +  todaystr )

#create log file to save all steps and outputs
log_file_path = "results/" + todaystr + "/log_file.txt"
sys.stdout = open(log_file_path, 'w')

#Create folder to save tabs with all infos
os.mkdir("results/" + todaystr + "/all_logs_csv")
os.mkdir("results/" + todaystr + "/all_fc6_fc7")
os.mkdir("results/" + todaystr + "/all_svm_weights")


#Write a memo file with parameters used for CNN
OPTIM = "optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)"
OPTIM2 = "keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=True)"
params_file_path =  "results/" + todaystr + "/params.txt"
f = open(params_file_path,'w')
f.write("weights:" + "VGGface " + "\n")
f.write("epochs:" + str(NB_EPOCHS) + "\n")
f.write("epochs2:" + str(NB_EPOCHS2) + "\n")
f.write("batch_size:" + str(BATCH_SIZE) + "\n" )
f.write("optim1:" + OPTIM  + "\n")
f.write("optim2:" + OPTIM2  + "\n")
f.write("freeze:" + str(FREEZE) + "\n")
#f.write("val:" + str(0.20) + "\n")
#f.write("early_stop:" + str(10) + "\n")
#f.write("reducelr0.1:" + str(5) + "\n")
f.write("dropout:" + str(0.5) + "\n")
f.write("HorizontalFlip:" + "True" + "\n")
#f.write(json.dumps(historique.history))
f.close()


#-----------------------------------------------------------------------------#
#--- FUNCTIONS AND CLASS


def load_images(tags_pict):
    """Load each image into list of numpy array and transform into array
    ----------
    tags_pict : pandas dataframe with annotation for pict
    Returns : array
    """
    img_data_list = []
    for p in tags_pict.index :
        img_path = tags_pict.full_path[p]
        img = load_img(img_path, target_size= inputShape)
        x = img_to_array(img)
        x = np.expand_dims(img, axis=0)
        # pre-process the image using the appropriate function based on the
        # model that has been loaded (i.e., mean subtraction, scaling, etc.)
        x = preprocess_input(x)
        img_data_list.append(x)
    img_data = np.array(img_data_list)
    img_data=np.rollaxis(img_data,1,0)
    img_data=img_data[0]
    return(img_data)

def unique(list1):
    """get unique values in a list
    input: list
    return: list
    """
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    for x in unique_list:
        return(x,)

def euclidean_dist(vector1, vector2):
    '''calculate the euclidean distance
    input:  lists
    return: euclidean distance
    '''
    dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    dist = math.sqrt(sum(dist))
    return dist

#--- plot function

def plot_loss_acc(hist , todaystr):
    save_path = "results/" + todaystr + "/"
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(len(train_loss))
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    plt.savefig(save_path + 'loss.png')
    plt.close()
    plt.style.use(['classic'])
    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    plt.style.use(['classic'])
    plt.savefig(save_path + 'acc.png')
    plt.close()


def plot_loss_acc_csv(hist_csv , todaystr):
    save_path = "results/" + todaystr + "/"
    train_loss=hist_csv['loss']
    val_loss=hist_csv['val_loss']
    train_acc=hist_csv['acc']
    val_acc=hist_csv['val_acc']
    xc=range(len(train_loss))
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.title('train_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    plt.savefig(save_path + 'loss.png')
    plt.close()
    plt.style.use(['classic'])
    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.title('train_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    plt.style.use(['classic'])
    plt.savefig(save_path + 'acc.png')
    plt.close()

class Histories(Callback):
    def on_train_begin(self,logs={}):
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracies.append(logs.get('acc'))
        self.val_accuracies.append(logs.get('val_acc'))


def split_get_act(model_custom, img_loaded, tags, n):
    '''get activations of images for fc6 and fc7 layers
    input: cnn model, img_loaded_transformed, tags, n = nb split if matrix is too big
    return: list of 2 pandas datas frame with activations for each layer (fc6 and fc7)
    '''
    img_loaded_split = np.array_split(img_loaded, n)
    for i in range(n):
        if i == 0 :
            fc6_all = get_activations(model_custom, img_loaded_split[i] , "fc6")['fc6/Relu:0']
            fc7_all = get_activations(model_custom, img_loaded_split[i] , "fc7")['fc7/Relu:0']
        else:
            fc6_all = np.concatenate((fc6_all,
                            get_activations(model_custom, img_loaded_split[i] , "fc6")['fc6/Relu:0']),
                            axis=0)
            fc7_all = np.concatenate((fc7_all,
                            get_activations(model_custom, img_loaded_split[i] , "fc7")['fc7/Relu:0']),
                            axis=0)
    fc6_all = pd.DataFrame(fc6_all)
    fc6_all.index = list(tags.index)
    fc7_all = pd.DataFrame(fc7_all)
    fc7_all.index = list(tags.index)
    return(fc6_all, fc7_all)



# Reset Keras Session -> free memory to avoid core dumped during loop
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))
    
    
    
#-----------------------------------------------------------------------------#
#--- IMPORT DATASET AND PREPARE TABS TO FILL
tags_pict = pd.read_csv('datas/pict_metadatas.csv', index_col=0)
#select neutral faces
tags_pict_clean = tags_pict[tags_pict.full_path.str.contains('-N.jpg') == True]
tags_pict = tags_pict_clean
tags_pict['Gender'] = tags_pict['Gender'].astype('category')
tags_pict['pred'] = len(tags_pict) * ["Nan"]
tags_pict['dist_centroid_fem_fc6'] = len(tags_pict) * ["Nan"]
tags_pict['dist_centroid_mal_fc6'] = len(tags_pict) * ["Nan"]
tags_pict['dist_centroid_fem_fc7'] = len(tags_pict) * ["Nan"]
tags_pict['dist_centroid_mal_fc7'] = len(tags_pict) * ["Nan"]
tags_pict['dist_centroid_mal_fc6_fc7'] = len(tags_pict) * ["Nan"]
tags_pict['dist_centroid_fem_fc6_fc7'] = len(tags_pict) * ["Nan"]
tags_pict['pred_fem_svm_fc6'] = len(tags_pict) * ["Nan"]
tags_pict['pred_fem_svm_fc7'] = len(tags_pict) * ["Nan"]
tags_pict['pred_fem_svm_fc6_fc7'] = len(tags_pict) * ["Nan"]


#Keep performances for each model
file_perf = open("results/" + todaystr + '/all_performances.txt', 'w')

#List of all indiv
L = list(tags_pict.Target)



#-----------------------------------------------------------------------------#
#--- LOOP - ALL INDIV


for i in range(len(L)):
#for i in range(2):
    print( "Tour :" + str(i) )
    tags_train = tags_pict[-(tags_pict.Target == L[i])]
    tags_test = tags_pict[tags_pict.Target == L[i]]
    print(tags_test)
    #make labels [fem,mal] and npy array
    label_train = np_utils.to_categorical(np.asarray(list(tags_train.Gender.cat.codes), dtype ='int64'), len(tags_train.Gender.cat.categories))
    label_test = np_utils.to_categorical(np.asarray(list(tags_test.Gender.cat.codes), dtype ='int64'), len(tags_train.Gender.cat.categories))
    img_loaded_train = load_images(tags_train)
    img_loaded_test = load_images(tags_test)
    #  MODELS  vggface
    model = VGGFace(model='vgg16',include_top=False, input_shape=(224, 224, 3))
    last_layer = model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='sigmoid', name='fc8/pred')(x)
    model_custom = Model( input=model.input, outputs = out)
    print("Architecture custom")
    print(model_custom.summary())
    #Freeze layers
    for layer in model_custom.layers[:FREEZE]:
        layer.trainable = False
    for layer in model_custom.layers:
        print(layer, layer.trainable)
    #Compile with first optim adam
    adam2 = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
    model_custom.compile(optimizer=adam2, loss='binary_crossentropy', metrics=['accuracy'])
    #Save learning steps and callbacks
    histories = Histories()
    histories2 = Histories()
    filelog = "log_" + str(L[i]) + ".csv"
    callbacks=[histories,
               CSVLogger("results/" + todaystr + "/all_logs_csv/"  + filelog , append=True, separator=';')  ,
               #EarlyStopping(patience=10),
               #ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,verbose = 1,cooldown=1)
               ]
    callbacks2=[histories2,
               CSVLogger("results/" + todaystr + "/all_logs_csv/"  + filelog , append=True, separator=';')  ,
               #EarlyStopping(patience=10),
               #ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,verbose = 1,cooldown=1)
               ]
    y_learn = tags_train.Gender.cat.codes
    y_learn_tags = tags_train
    #NB mal fem
    print( "Nb mal train:" + str(sum(y_learn)))
    print( "Nb fem train :" + str(len(y_learn) - sum(y_learn)))
    # Create train generator.
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=30,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.2,
                                       horizontal_flip  =True)
    train_generator = train_datagen.flow(X_learn, y_learn, shuffle=False,
                                         batch_size=BATCH_SIZE, seed=1)
    # Create validation generator
    val_datagen = ImageDataGenerator(rescale = 1./255)
    #val_generator = val_datagen.flow(X_val, y_val, shuffle=False, batch_size=BATCH_SIZE, seed=1)
    train_steps_per_epoch = X_learn.shape[0] //  BATCH_SIZE
    #val_steps_per_epoch = X_val.shape[0] //  BATCH_SIZE
    hist = model_custom.fit_generator(train_generator,
                                  steps_per_epoch=train_steps_per_epoch,
                                  #validation_data=val_generator,
                                  #validation_steps=val_steps_per_epoch,
                                  epochs=NB_EPOCHS, verbose=1,
                                  callbacks=callbacks)
    # Compile with second optim
    sgd2 = optimizers.SGD(lr=0.0001,  momentum=0.9, decay=0.0, nesterov=True)
    #sgd2 = optimizers.SGD(lr=0.0001,  momentum=0.9, decay=0.0, nesterov=True)
    model_custom.compile(optimizer=sgd2, loss='binary_crossentropy', metrics=['accuracy'])
    hist2 =  model_custom.fit_generator(train_generator,
                                  steps_per_epoch=train_steps_per_epoch,
                                  #validation_data=val_generator,
                                  #validation_steps=val_steps_per_epoch,
                                  epochs=NB_EPOCHS2, verbose=1,
                                  callbacks=callbacks2)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_generator = test_datagen.flow(img_loaded_test, list(tags_test.Gender.cat.codes), shuffle=False, batch_size=1, seed=1)
    scorebis = model_custom.evaluate_generator(test_generator,steps=(test_generator.n))
    print('Final test accuracy for '+ L[i] + ":" +  str(scorebis[1]*100.0) )
    file_perf.write('Final test accuracy for '+ L[i] + " : " +  str(scorebis[1]*100.0) + "\n")
    test_generator.reset()
    pred = model_custom.predict_generator(test_generator,verbose=1, steps= test_generator.n )
    print(pred)
    tags_test['pred'] = [item for sublist in pred.tolist() for item in sublist]
    print(tags_test['pred'])
    #ACTIVATIONS TEST FC6 et FC7
    fc6_fc7_splits = split_get_act(model_custom, img_loaded_test, tags_test, 3)
    all_fc6_df = fc6_fc7_splits[0]
    all_fc7_df = fc6_fc7_splits[1]
    #ACTIVATIONS TRAIN FC6 et FC7
    fc6_fc7_splits = split_get_act(model_custom, X_learn, y_learn_tags, 20)
    fc6_train_all = fc6_fc7_splits[0]
    fc7_train_all = fc6_fc7_splits[1]
    print(len(fc6_train_all))
    print(len(fc6_train_all))
    fc6_fc7 = pd.concat([all_fc6_df, all_fc7_df], axis=1)
    train_fc6_fc7 = pd.concat([fc6_train_all, fc7_train_all], axis=1)
    list_train_fem = list( y_learn_tags.index[y_learn_tags.Gender == "F"])
    list_train_mal = list( y_learn_tags.index[y_learn_tags.Gender == "M"])
    fc6_train_fem = fc6_train_all.ix[list_train_fem]
    fc6_train_mal = fc6_train_all.ix[list_train_mal]
    fc7_train_fem = fc7_train_all.ix[list_train_fem]
    fc7_train_mal = fc7_train_all.ix[list_train_mal]
    fc6_fc7_train_fem = train_fc6_fc7.ix[list_train_fem]
    fc6_fc7_train_mal = train_fc6_fc7.ix[list_train_mal]
    centroide_fem_fc6 = list(fc6_train_fem.mean(axis = 0))
    centroide_mal_fc6 = list(fc6_train_mal.mean(axis = 0))
    centroide_fem_fc7 = list(fc7_train_fem.mean(axis = 0))
    centroide_mal_fc7 = list(fc7_train_mal.mean(axis = 0))
    centroide_fem_fc6_fc7 = list(fc6_fc7_train_fem.mean(axis = 0))
    centroide_mal_fc6_fc7 = list(fc6_fc7_train_mal.mean(axis = 0))
    tags_test['dist_centroid_fem_fc6'] = list(all_fc6_df.apply(lambda row: euclidean_dist(row,centroide_fem_fc6 ), axis=1))
    tags_test['dist_centroid_mal_fc6'] = list(all_fc6_df.apply(lambda row: euclidean_dist(row,centroide_mal_fc6 ), axis=1))
    tags_test['dist_centroid_fem_fc7'] = list(all_fc7_df.apply(lambda row: euclidean_dist(row,centroide_fem_fc7 ), axis=1))
    tags_test['dist_centroid_mal_fc7'] = list(all_fc7_df.apply(lambda row: euclidean_dist(row,centroide_mal_fc7 ), axis=1))
    tags_test['dist_centroid_mal_fc6_fc7'] = list(fc6_fc7.apply(lambda row: euclidean_dist(row,centroide_mal_fc6_fc7 ), axis=1))
    tags_test['dist_centroid_fem_fc6_fc7'] = list(fc6_fc7.apply(lambda row: euclidean_dist(row,centroide_fem_fc6_fc7 ), axis=1))
    classifier = svm.SVC(kernel='linear', C=0.01, probability=True)
    classifier.fit(fc6_train_all, list(y_learn_tags.Gender.cat.codes))
    pred_prob_svm = classifier.predict_proba(all_fc6_df)
    weights1 = classifier.coef_
    classifier = svm.SVC(kernel='linear', C=0.01, probability=True)
    classifier.fit(fc7_train_all, list(y_learn_tags.Gender.cat.codes))
    pred_prob_svm_2 = classifier.predict_proba(all_fc7_df)
    weights2 = classifier.coef_
    classifier = svm.SVC(kernel='linear', C=0.01, probability=True)
    classifier.fit(train_fc6_fc7, list(y_learn_tags.Gender.cat.codes))
    pred_prob_svm_3 = classifier.predict_proba(fc6_fc7)
    weights3 = classifier.coef_
    #hist_csv = pd.read_csv("results/" + todaystr + "/all_logs_csv/"  + filelog  , sep=';' )
    tags_test['pred_fem_svm_fc6'] = list(pred_prob_svm[:,0])
    tags_test['pred_fem_svm_fc7'] = list(pred_prob_svm_2[:,0])
    tags_test['pred_fem_svm_fc6_fc7'] = list(pred_prob_svm_3[:,0])
    for img in list(tags_test.Target):
        print(tags_pict.pred[tags_pict.Target == img])
        print(tags_test[tags_test.Target == img])
        tags_pict.pred[tags_pict.Target == img] = tags_test.pred[tags_test.Target == img]
        tags_pict['dist_centroid_fem_fc6'][tags_pict.Target == img] = tags_test['dist_centroid_fem_fc6'][tags_test.Target == img]
        tags_pict['dist_centroid_mal_fc6'][tags_pict.Target == img] = tags_test['dist_centroid_mal_fc6'][tags_test.Target == img]
        tags_pict['dist_centroid_fem_fc7'][tags_pict.Target == img] = tags_test['dist_centroid_fem_fc7'][tags_test.Target == img]
        tags_pict['dist_centroid_mal_fc7'][tags_pict.Target == img] = tags_test['dist_centroid_mal_fc7'][tags_test.Target == img]
        tags_pict['dist_centroid_fem_fc6_fc7'][tags_pict.Target == img] = tags_test['dist_centroid_fem_fc6_fc7'][tags_test.Target == img]
        tags_pict['dist_centroid_mal_fc6_fc7'][tags_pict.Target == img] = tags_test['dist_centroid_mal_fc6_fc7'][tags_test.Target == img]
        tags_pict['pred_fem_svm_fc6'][tags_pict.Target == img] = tags_test['pred_fem_svm_fc6'][tags_test.Target == img]
        tags_pict['pred_fem_svm_fc7'][tags_pict.Target == img] = tags_test['pred_fem_svm_fc7'][tags_test.Target == img]
        tags_pict['pred_fem_svm_fc6_fc7'][tags_pict.Target == img] = tags_test['pred_fem_svm_fc6_fc7'][tags_test.Target == img]
    all_w = pd.DataFrame([weights1.tolist(), weights2.tolist(), weights3.tolist()])
    all_w.to_csv("results/" + todaystr + "/all_svm_weights/" + filelog)
    fc_to_save = pd.concat([fc6_fc7, train_fc6_fc7])
    fc_to_save.to_csv("results/" + todaystr + "/all_fc6_fc7/" + filelog)
    #plot_loss_acc_csv(hist_csv, todaystr)
    reset_keras()
    tags_pict.to_csv("results/" + todaystr + "/tags_pict_pred_tmp.csv")


file_perf.close()

tags_pict.to_csv("results/" + todaystr + "/tags_pict_pred.csv")


sys.stdout.close()
