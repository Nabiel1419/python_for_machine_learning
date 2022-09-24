

import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS=3
EPOCHS=50
dataset=tf.keras.preprocessing.image_dataset_from_directory(r"C:\Users\nabiel\Myproject\dataset",shuffle=True,
                                                            batch_size=(BATCH_SIZE),image_size=(IMAGE_SIZE,IMAGE_SIZE))


# In[11]:


classes=dataset.class_names


# In[12]:


print(classes)


# In[13]:


plt.figure(figsize=(10,10))
for i in range(12):
    for image_batch,label_batch in dataset.take(1):
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.axis("off")
        plt.subplot(3,4,i+1)
        print(classes[label_batch[0].numpy()])
        plt.xlabel(classes[label_batch[i].numpy()])


# In[ ]:


get_ipython().run_line_magic('==>training', '80%')
get_ipython().run_line_magic('==>validation', '10%')
get_ipython().run_line_magic('==>', 'test 10%')


# In[ ]:


len(dataset)


# In[ ]:


train_size=0.8
len(dataset)*train_size


# In[ ]:


train_ds=dataset.take(105)


# In[ ]:


len(train_ds)


# In[ ]:


test_ds=dataset.skip(105)
len(test_ds)


# In[ ]:


val_size=0.1
len(dataset)*val_size


# In[ ]:


val_ds=dataset.take(13)


# In[ ]:


test_ds=test_ds.skip(13)


# In[ ]:


len(test_ds)


# In[14]:


def  get_dataset_partitions_tf(dataset, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
        if shuffle:
              dataset=dataset.shuffle(shuffle_size,seed=12) 
        ds_size=len(dataset)
        train_size= int(train_split*ds_size)
        val_size=int(val_split*ds_size)
        train_ds=dataset.take(train_size)
        val_ds=dataset.skip(train_size).take(val_size)
        test_ds=dataset.skip(train_size).skip(val_size)
        return train_ds,val_ds,test_ds


# In[15]:


train_ds,val_ds,test_ds= get_dataset_partitions_tf(dataset)


# In[16]:


len(train_ds)


# In[17]:


len(val_ds)


# In[ ]:


len(test_ds)


# In[18]:


train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds=val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds=test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[19]:


resize_and_rescale=tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])


# In[20]:


data_augmentation=tf.keras.Sequential(
[
    layers.experimental.preprocessing.RandomRotation(2.0),
    layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical')
])


# In[21]:


input_shape=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
n_classes=4
model=models.Sequential([
    resize_and_rescale,
    data_augmentation,
        layers.Conv2D(32,(4,4),activation='relu',input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64,(4,4),activation='relu',input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(128,(4,4),activation='relu',input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(4,4),activation='relu',input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(4,4),activation='relu',input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(n_classes,activation='softmax'),
    


])
model.build(input_shape=input_shape)


# In[22]:


model.summary()


# In[ ]:


model.compile(
optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# In[ ]:


EPOCHS=50
history=model.fit(
train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)


# In[ ]:


scores=model.evaluate(test_ds)


# In[188]:


scores


# In[194]:


print(history.history.params)


# In[193]:


print(history.keys)


# In[198]:


keys=history.history.keys()


# In[199]:


print(keys)


# In[ ]:


for image_batch,labels_batch in test_ds.take(1):
    first_image=image_batch[0].numpy().astype("uint8")
    first_label=labels_batch[0].numpy()
    print("first image to predict")
    plt.imshow(first_image)
    print("first image's actual label:",classes[first_label])
    plt.axis("off")
    batch_prediction=model.predict(image_batch)
    
print("predicted label:",classes[np.argmax(batch_prediction[0])])


# In[232]:





# In[245]:


score=model.evaluate(test_ds)


# In[246]:


score


# In[247]:


np.argmax[score]


# In[249]:


def predict(model,img):
    img_array=tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array=tf.expand_dims(img_array,0)
    predictions=model.predict(img_array)
    predicted_class=classes[np.argmax(predictions[0])]
    confidence=round(100*(np.max(predictions[0],)),2)
    return predicted_class,confidence


# In[260]:


plt.figure(figsize=(15,20))
for images,labels in test_ds.take(1):
    for i in range(9):
        ax=plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")
        predicted_class,confidence=predict(model,images[i].numpy())
        actual_label=classes[labels[i]]
        plt.title(f" Actual:{actual_label} ,\n predicted:{predicted_class},\n confidence{confidence}%")


# In[ ]:





# In[ ]:




