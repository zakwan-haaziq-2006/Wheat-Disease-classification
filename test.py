from keras.models import model_from_json
from keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model
import numpy as np


model = load_model('costum.model.h5')
#{'Aphid': 0, 'Black Rust': 1, 'Blast': 2, 'Brown Rust': 3, 'Common Root Rot': 4, 'Fusarium Head Blight': 5, 'Healthy': 6, 'Leaf Blight': 7, 'Mildew': 8, 'Mite': 9, 'Septoria': 10, 'Smut': 11, 'Stem fly': 12, 'Tan spot': 13, 'Yellow Rust': 14}
labels = ['Aphid','Black Rust','Blast','Brown Rust','Common Root Rot','Fusarium Head Blight','Healthy','Leaf Blight','Mildew','Mite','Septoria','Smut','Stem fly','Tan spot','Yellow Rust']


def classify(image):
    img = load_img(image,target_size=(256,256))
    img = img_to_array(img)
    img = img/255
    img = np.expand_dims(img,axis=0)
    result = model.predict(img)
    res = np.argmax(result,axis=1)
    print(f"Predicted : {labels[res[0]]} | Actual img : {image.split('/')[-1]}")
    
    
    
files = []
import os
for f in os.listdir('data/test'):
    path = os.path.join('data/test',f)
    for img in os.listdir(path):
        img_path = os.path.join(path,img)
        print(img_path)
        files.append(img_path)
        
# for f in files:
#     classify(f)

print(len(files))
classify(files[450])
        

        