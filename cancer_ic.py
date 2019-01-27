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
from tensorflow.keras.optimizers import Adam

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#%%
size = 96
n_sample = 1000

path = "/Users/Qi-MAC/Work/storage/" # path you put all the data, don't forget the ending slash!

print(os.listdir(path))

#%%
'''
df = pd.read_csv(path+"train_labels.csv")

def df_info(df):
    print(df.shape) # shape of the df
    print(df.columns) # colnames
    print(df.isnull().sum()) # detect null values, which is 0 for our dataset
    print(df.label.value_counts())
    print(df.label.mean())

print("\nAll images:")
df_info(df)

df_0 = df[df.label == 0].sample(n_sample, random_state = 0)
df_1 = df[df.label == 1].sample(n_sample, random_state = 0)

df = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)
# shuffle
df = shuffle(df)

print("\nSampled images:")
df_info(df)

def show_img(label):
    print("\nView sample images:")
    sample = df[df.label==label].sample()
    imgid = sample.id.values[0]
    print(sample.label.values[0])
    img = plt.imread(path+ "train/" + imgid + ".tif")
    plt.imshow(img)

y = df.label
x_train, x_test = train_test_split(df, test_size=0.2, random_state=0, stratify=y)

sample_test_dir = path + "sample_test"
sample_train_dir = path + "sample_train"
    
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
'''

sample_test_dir = path + "sample_test"
sample_train_dir = path + "sample_train"

n_train = 2*len(os.listdir(os.path.join(sample_train_dir,"0")))
n_test = 2*len(os.listdir(os.path.join(sample_test_dir,"0")))
train_batch = 1
val_batch = 1

train_steps = np.ceil(n_train/train_batch)
val_steps = np.ceil(n_test/val_batch)

datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(sample_train_dir,
                                        target_size=(size,size),
                                        batch_size=train_batch,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(sample_test_dir,
                                        target_size=(size,size),
                                        batch_size=val_batch,
                                        class_mode='categorical')

# Note: shuffle=False causes the test dataset to not be shuffled
test_gen = datagen.flow_from_directory(sample_test_dir,
                                        target_size=(size,size),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)


kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 2*first_filters
third_filters = 4*first_filters
alpha = 1e-4

dropout_conv = 0.2
dropout_dense = 0.2

model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (96, 96, 3)))
#model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
#model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
#model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
#model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
#model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
#model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation = "softmax"))

model.summary()

model.compile(Adam(lr=alpha), loss='binary_crossentropy', metrics=['accuracy'])

cnn = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=20, verbose=1)


