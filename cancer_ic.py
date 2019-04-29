import os
from shutil import copyfile, rmtree

import pandas as pd # for import/export/manipulate data
import numpy as np # for matrix/vector operation
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from utils import plot_confusion_matrix

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-e", dest="epochs", required=True, help="number of epochs")
parser.add_argument("-s", dest="samples", required=False, help="number of samples")
parser.add_argument("-r", dest="resample", required=True, help="resample? [1-y/0-N]")
args = parser.parse_args()

#%%
IMG_SIZE = 96

N_CHANNEL = 3

path = "/dev/shm/80kdata/" # path you put all the data, don't forget the ending slash!

#%%
class NN_Model(object):
    def __init__(self, test_gen=None, train_gen=None, train_steps=None, batch_size=1, epochs=1):
        self.test_gen = test_gen
        self.train_gen = train_gen
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.alpha = None
    
    def set_learning_rate(self,lr=1e-5):
        self.alpha = lr
    
    def set_epochs(self, epochs=1):
        self.epochs = epochs


class CNN_Model(NN_Model):
    def __init__(self, kernel_size=(3,3), pool_size=(2,2), cnn_activation='relu', dense_activation='relu', out_activation='softmax', n_classes=2, n_dense=256):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.model = Sequential()
        self.cnn_activation = cnn_activation
        self.dense_activation = dense_activation
        self.out_activation = out_activation
        self.dropout_conv = 0
        self.dropout_dense = 0
        self.isdropout = False
        self.n_classes = n_classes
        self.n_dense = n_dense
        self.result = None
        self.y_test = None
        self.pred_proba = None
        self.y_pred = None
        self.cm = None
        
    def use_dropout(self, dp_conv = 0.2, dp_dense = 0.2):
        self.isdropout = True
        self.dropout_conv = dp_conv
        self.dropout_dense = dp_dense
        
    def build_cnn(self, n_filter=[32,64,128]):
        dropout_conv = self.dropout_conv
        dropout_dense = self.dropout_dense
        kernel_size = self.kernel_size
        pool_size = self.pool_size
        cnn_activation = self.cnn_activation
       
        model = self.model
        
        for i in range(len(n_filter)):
            if i == 0:
                model.add(Conv2D(n_filter[i], kernel_size, activation=cnn_activation, input_shape = (IMG_SIZE, IMG_SIZE, N_CHANNEL)))
            else:
                model.add(Conv2D(n_filter[i], kernel_size, activation=cnn_activation))
            model.add(MaxPooling2D(pool_size = pool_size)) 
            if self.isdropout:
                model.add(Dropout(dropout_conv))
                
        model.add(Flatten())
        model.add(Dense(self.n_dense, activation = self.dense_activation))
        if self.isdropout:
            model.add(Dropout(dropout_dense))
        model.add(Dense(self.n_classes, activation = self.out_activation))
        
        model.summary()
        
        self.model = model
    
    def compile_optimization(self, optimization_fn=Adam, loss='binary_crossentropy', metrics=['accuracy']):     
        self.model.compile(optimization_fn(lr=self.alpha), loss=loss, metrics=metrics)
        
    def run(self):
        self.result = self.model.fit_generator(self.train_gen, steps_per_epoch=self.train_steps,
                    epochs=self.epochs, verbose=1)

    def analyze(self):
        cnn = self.result
        plt.figure(1)
        plt.plot(cnn.history['acc'],'*-',color='b')
        plt.show()
        
        plt.figure(2)
        plt.plot(cnn.history['loss'],'*-',color='r')
        plt.show()
        
        self.y_test = self.test_gen.classes
        self.pred_proba = self.model.predict(self.test_gen)
        self.y_pred = self.pred_proba.argmax(axis=1)
        self.cm = confusion_matrix(self.y_test, self.y_pred)
        
        classes = ['no cancer', 'cancer']
        
        plt.figure(3)
        plot_confusion_matrix(self.cm, classes)
        plt.show()
        

#%%
class Data_Flow(object):
    
    def __init__(self, path, csv, n_sample, cnn_model, test_size = 0.2, random_state = 0, sample_test_dir_name = "sample_test", sample_train_dir_name = "sample_train"):
        self.path = path
        print(os.listdir(self.path))
        self.df = pd.read_csv(csv)
        self.n_sample = n_sample
        self.sample_df = None
        self.sample_test_dir_name = sample_test_dir_name
        self.sample_train_dir_name = sample_train_dir_name
        self.sample_test_dir = path + sample_test_dir_name
        self.sample_train_dir = path + sample_train_dir_name
        self.test_size = test_size
        self.random_state = random_state
        self.cnn_model = cnn_model
        
    def filter_bad_images(self):
        self.df[self.df['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']
        self.df[self.df['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']
    
    def df_info(self, df, msg="\ndf_info"):
        print(msg)
        print(df.shape) # shape of the df
        print(df.columns) # colnames
        print(df.isnull().sum()) # detect null values, which is 0 for our dataset
        print(df.label.value_counts())
        print(df.label.mean())
    
    def sample(self):
        df = self.df
        n_sample = self.n_sample
        df_0 = df[df.label == 0].sample(n_sample, random_state = 0)
        df_1 = df[df.label == 1].sample(n_sample, random_state = 0)   
        sample_df = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)
        # shuffle
        self.sample_df = shuffle(sample_df)
    
    def show_img(self,label):
        print("\nView sample images:")
        df = self.sample_df
        sample = df[df.label==label].sample()
        imgid = sample.id.values[0]
        print(sample.label.values[0])
        img = plt.imread(path+ "train/" + imgid + ".tif")
        plt.imshow(img)
        
    def gen_sample_dir(self):
        df = self.sample_df   
        path = self.path
        y = df.label
        x_train, x_test = train_test_split(df, test_size=self.test_size, random_state=self.random_state, stratify=y)
    
        sample_test_dir = self.sample_test_dir
        sample_train_dir = self.sample_train_dir
        
        if os.path.exists(sample_test_dir):
            rmtree(sample_test_dir)
        if os.path.exists(sample_train_dir):
            rmtree(sample_train_dir)
        os.mkdir(sample_test_dir)
        os.mkdir(os.path.join(sample_test_dir,"0"))
        os.mkdir(os.path.join(sample_test_dir,"1"))
        os.mkdir(sample_train_dir)
        os.mkdir(os.path.join(sample_train_dir,"0"))
        os.mkdir(os.path.join(sample_train_dir,"1"))
        os.listdir(path)
        
        for imgid in list(x_train.id):
            img = imgid + ".tif"
            if x_train[x_train.id == imgid].label.iloc[0] == 0:       
                src = os.path.join(path, "train", img)
                dst = os.path.join(sample_train_dir,"0", img)
                copyfile(src, dst)
            if x_train[x_train.id == imgid].label.iloc[0] == 1:       
                src = os.path.join(path, "train", img)
                dst = os.path.join(sample_train_dir,"1", img)
                copyfile(src, dst)
            
        for imgid in list(x_test.id):
            img = imgid + ".tif"
            if x_test[x_test.id == imgid].label.iloc[0] == 0:       
                src = os.path.join(path, "train", img)
                dst = os.path.join(sample_test_dir,"0", img)
                copyfile(src, dst)
            if x_test[x_test.id == imgid].label.iloc[0] == 1:       
                src = os.path.join(path, "train", img)
                dst = os.path.join(sample_test_dir,"1", img)
                copyfile(src, dst)
                
    def gen_data(self, train_batch_size=None, va):
        sample_train_dir = self.sample_train_dir
        sample_test_dir = self.sample_test_dir
        n_train = 2*len(os.listdir(os.path.join(sample_train_dir,"0")))
        if batch_size:
            self.cnn_model.batch_size = batch_size
        
        self.cnn_model.train_steps = np.ceil(n_train/batch_size)
        
        datagen = ImageDataGenerator(rescale=1.0/255)
        
        self.cnn_model.train_gen = datagen.flow_from_directory(sample_train_dir,
                                                target_size=(IMG_SIZE,IMG_SIZE),
                                                batch_size=batch_size,
                                                class_mode='categorical')
        
        # Note: shuffle=False causes the test dataset to not be shuffled
        self.cnn_model.val_gen = datagen.flow_from_directory(sample_test_dir,
                                                target_size=(IMG_SIZE,IMG_SIZE),
                                                batch_size=1,
                                                class_mode='categorical',
                                                shuffle=False)

        

cnn_model = CNN_Model(kernel_size=(3,3),pool_size=(2,2))
cnn_model.set_learning_rate(1e-3)
cnn_model.set_epochs(int(args.epochs))
data_flow = Data_Flow(path, path+"train_labels.csv", int(args.samples), cnn_model)
data_flow.filter_bad_images()
if bool(int(args.resample)):
    data_flow.df_info(data_flow.df, msg="\nAll images:")
    data_flow.sample()
    data_flow.df_info(data_flow.sample_df, msg="\nSampled images:")
    data_flow.gen_sample_dir()
data_flow.gen_data(batch_size=100)
#cnn_model.use_dropout()
cnn_model.build_cnn()
cnn_model.compile_optimization(Adam)
cnn_model.run()
#cnn_model.analyze()


