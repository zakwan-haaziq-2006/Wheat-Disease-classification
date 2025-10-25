from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,MaxPool2D,AveragePooling2D,Dropout
from tensorflow.keras.applications import MobileNetV2,ResNet50,EfficientNetB0, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.layers import Dense,BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
import torch




base_model = InceptionV3(weights='imagenet',include_top=False,input_shape=(256,256,3))
base_model.trainable = False

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(256,256,3)))
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())


model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(15,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,zoom_range=0.2,shear_range=0.2,horizontal_flip=True,height_shift_range=0.3,width_shift_range=0.1,rotation_range=20,fill_mode='nearest',brightness_range=[0.5,1.5])
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory('wheat_disease_dataset/train',target_size=(256,256),batch_size=32,class_mode='categorical')
val_data = val_datagen.flow_from_directory('wheat_disease_dataset/test',target_size=(256,256),batch_size=32,class_mode='categorical')
print(train_data.class_indices)
print(val_data.class_indices)
callbacks = [EarlyStopping(monitor='val_loss',patience=5),ReduceLROnPlateau(monitor='val_loss',patience=5),ModelCheckpoint('model.h5',monitor='val_loss',save_best_only=True,verbose=1)]

history = model.fit(train_data,validation_data=val_data,epochs=30,callbacks=callbacks,steps_per_epoch=100,validation_steps=100)
plt.figure(0,figsize=(10,10))
plt.title('Accuracy vs Eppochs')
plt.plot(history.history['accuracy'],label='Accuracy')
plt.plot(history.history['val_accuracy'],label='val Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("efficientnet_modelaccuracy.png")
plt.show()

plt.figure(1,figsize=(10,10))
plt.title('Loss vs Eppochs')
plt.plot(history.history['loss'],label='Loss')
plt.plot(history.history['val_loss'],label='val Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("efficientnet_Model_loss.png")
plt.show()


model.save('InceptionV3.model.h5')
model_json = model.to_json()
with open('model.json','w') as file:
    file.write(model_json)
    model.save_weights('model_weights.h5')
    
