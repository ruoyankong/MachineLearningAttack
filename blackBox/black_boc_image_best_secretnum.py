#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import numpy as np
import math
from skimage import img_as_ubyte


# In[2]:


fashion_mnist = keras.datasets.fashion_mnist


# In[3]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[4]:


import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(x_train[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()


# In[5]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# In[6]:

#
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[y_train[i]])
# plt.show()
#
#
# # In[7]:
#
#
def showImg(img, name = ""):
    fig = plt.figure()
    if(len(img.shape) > 2):
        for i in range(img.shape[0]):
            I = img[i]
            plt.subplot(1,img.shape[0]+1,i+1)
            plt.imshow(I)
    else:
        plt.imshow(img)
    plt.grid(False)
    plt.show()
    if name!="":
        fig.savefig("./test_accuracy_force_d/"+name)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(20, activation='softmax')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train,
         y_train,
         batch_size=128,
         epochs=20)
#          validation_data=(x_valid, y_valid),
#          callbacks=[checkpointer])

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
# secret_num=1



# In[8]:


def getSecret(image):
    return np.unpackbits(np.array(image.flatten(), np.uint8))


# In[9]:


def bin2str(bs):
    secret_string = ''.join(str(x) for x in bs)
    return secret_string

def bitToLabel(bits, label_num):
    x = math.ceil(len(bits)/label_num)
    label = [0]*x
    for i in range(x):
        label[i] = int(bits[i*label_num:(i+1)*label_num], 2)
    return label

def labelToBit(label, label_num):
    strv = ""
    for l in label:
        temp = bin(l).replace('0b','')
#         strv += temp
        pad = label_num - len(temp)
        strv += pad * "0" + temp
    return strv

def str2bin(strv):
    return np.array([int(c) for c in strv])


# In[10]:


def getPrimeList(len):
    list=[]
    i=2
    for i in range (2,len):
        j=2
        for j in range(2,i):
            if(i%j==0):
                break
        else:
            list.append(i)
    return list

def test_accuracy(force, d, secret_num):
    secret_img = x_train[0:secret_num]
    showImg(secret_img, "original_secretnum_"+str(secret_num))
    def genMal(length, shape = (28, 28), force = force):
        mal_x = np.zeros((length, shape[0], shape[1]))
        p_list = getPrimeList(force)
        for i in range(length):
            round = int((i / (28 * 28)) + 1)
    #         print(int(i/28) - round + 1)
    #         mal_x[i, int((i%(28*28))/28) - round + 1, i%28] = round * 3;
    #         mal_x[i, int(((i+3)%(28*28))/28) - round + 1, (i+3)%28] = round * 5;
    #         mal_x[i, int(((i+5)%(28*28))/28) - round + 1, (i+5)%28] = round * 7;
    #         mal_x[i, int(((i+11)%(28*28))/28) - round + 1, (i+11)%28] = round * 11;
    #         mal_x[i, int(((i+13)%(28*28))/28) - round + 1, (i+13)%28] = round * 13;
    #         mal_x[i, int(((i+17)%(28*28))/28) - round + 1, (i+17)%28] = round * 17;
            for p in p_list:
                mal_x[i, int(((i + p)%(28*28))/28) - round + 1, (i+p) % 28] = round * p

        return mal_x

    def duplicate(x_mal, y_mal, d = d):
        x = x_mal
        y = y_mal
        for i in range(d):
            x_mal = np.vstack((x, x_mal))
            y_mal = np.hstack((y, y_mal))
        return x_mal, y_mal

    def augumentAttack(x_train, y_train, start = 0, size = 1):
        secret_image = x_train[start:start+size]
        # showImg(secret_image)
        secret_image_int = img_as_ubyte(secret_image)
        s = getSecret(secret_image_int)

        max_bit_size = math.floor(math.log(len(class_names), 2))
        transformed_s = bitToLabel(bin2str(s), max_bit_size)

        x_mal = genMal(len(transformed_s), force=force)
        y_mal = np.array(transformed_s)

        d_x_mal, d_y_mal = duplicate(x_mal, y_mal, d)
        x_new = np.vstack((x_train, d_x_mal))

        y_new = np.hstack((y_train, d_y_mal))

        secret_lenght = len(transformed_s)

        return x_new, y_new, secret_lenght


    # In[11]:




    # In[12]:


    def decode(s, shape = (28, 28)):
        # convert binary to decimals
        decimals = np.packbits(s);
        print(decimals.shape)
        return decimals.reshape((-1, shape[0], shape[1]))


    # ## attack begins at here
    #
    # secret_num indicate the number of image you want to steal

    # In[13]:


    # secret_num = 1
    x_new, y_new, secret_length = augumentAttack(x_train, y_train, 2, secret_num)


    # In[22]:


    ## train malicious model

    #by increaseing epochs of training m


    # In[15]:


    test_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation = 'relu'),
        keras.layers.Dense(20, activation='softmax')
    ])

    test_model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    test_model.fit(x_new,
             y_new,
             batch_size=128,
             epochs=20)


    # In[16]:



    x_mal = genMal(secret_length, force=force)
    max_bit_size = math.floor(math.log(len(class_names), 2))
    y_pred = np.argmax(test_model.predict(x_mal), axis = 1)

    print(len(y_pred))

    strv = labelToBit(y_pred, max_bit_size)[:secret_num*6272]

    print(len(strv))
    test_s = str2bin(strv)
    ds = decode(test_s)
    showImg(ds, name = "_".join(["force", str(force), "d", str(d), "secretnum", str(secret_num)]))


    # In[17]:


    # secret_img = x_train[0:secret_num]
    # showImg(secret_img)

    test_loss, test_acc = test_model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    return test_acc


    # # In[18]:
    #
    #
    # model.compile(
    #     optimizer = 'adam',
    #     loss = 'sparse_categorical_crossentropy',
    #     metrics=['accuracy']
    # )
    #
    # model.fit(x_train,
    #          y_train,
    #          batch_size=64,
    #          epochs=10)
    # #          validation_data=(x_valid, y_valid),
    # #          callbacks=[checkpointer])
    #
    # test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    # print('\nTest accuracy:', test_acc)

result = np.zeros(9)
# force_list = [10,15,20]
# d_list = [10,15,20]
secret_num_list = range(1, 10)
for s in range(5,9):
        result[s] = test_accuracy(10, 20, secret_num_list[s])
        print(result)




