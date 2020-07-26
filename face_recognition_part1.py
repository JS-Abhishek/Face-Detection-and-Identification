from keras.models import load_model
model = load_model('facenet_keras.h5')
import mtcnn
import numpy as np
import PIL
from PIL import Image
from mtcnn.mtcnn import MTCNN

def extract_faces(filename, required_size = (160,160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1,y1,width,height = results[0]['box']
    x1,y1 = abs(x1), abs(y1)
    x2,y2 = x1 + width, y1 + height
    face = pixels[y1:y2,x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

import os
def load_faces(directory):
    faces = list()
    for filename in os.listdir(directory):
        path = directory + filename
        face = extract_faces(path)
        faces.append(face)
    return faces

def load_dataset(directory):
    X,Y = list(), list()
    for subdir in os.listdir(directory):
        path = directory + subdir + '/'
        if not os.path.isdir(path):
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print('>loaded %d examples for class: %s' %(len(faces), subdir))
        X.extend(faces)
        Y.extend(labels)
    return np.asarray(X),np.asarray(Y)

trainX, trainY = load_dataset('C:/Users/jsabh/OneDrive/Desktop/python_new/train/')

print(trainX.shape, trainY.shape)

testX,testY = load_dataset('C:/Users/jsabh/OneDrive/Desktop/python_new/val/')

np.savez_compressed('5-celebrity.npz',trainX,trainY,testX,testY)

print(testX.shape,testY.shape)

data = np.load('5-celebrity.npz')
trainX,trainY,testX,testY = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']
print('Loaded:', trainX.shape, trainY.shape, testX.shape,testY.shape)

model = load_model('facenet_keras.h5')
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean)/std
    samples = np.expand_dims(face_pixels,axis = 0)
    yhat = model.predict(samples)
    embedding = yhat[0]
    return embedding

newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model,face_pixels)
    newTrainX.append(embedding)
newTrainX = np.asarray(newTrainX)
print(newTrainX.shape)

newTestX = list()
for face_pixels in testX:
    embedding = get_embedding(model,face_pixels)
    newTestX.append(embedding)
newTestX = np.asarray(newTestX)
print(newTestX.shape)

np.savez_compressed('5-celeb-embedding.npz',newTrainX,trainY,newTestX,testY)

data = np.load('5-celeb-embedding.npz')
trainX,trainY,testX,testY = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']
print("Dataset: train = %d, test = %d" %(trainX.shape[0], testX.shape[0]))

import sklearn
from sklearn.preprocessing import Normalizer,LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
textX = in_encoder.transform(testX)

out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)
testY = out_encoder.transform(testY)

model = SVC(kernel = 'linear', probability=True)
model.fit(trainX,trainY)

y_pred_train = model.predict(trainX)
y_pred_test = model.predict(testX)

score_train = accuracy_score(trainY, y_pred_train)
score_test = accuracy_score(testY,y_pred_test)

print("Accuracy Score: train = %.3f, test = %.3f"%(score_train*100, score_test*100))

import random
data = np.load('5-celebrity.npz')
testX_faces = data['arr_2']
#testing on random sample
selection = random.choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = textX[selection]
random_face_class = testY[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

#prediction of the faces
samples = np.expand_dims(random_face_emb, axis = 0)
y_classes = model.predict(samples)
y_prob = model.predict_proba(samples)

class_index = y_classes[0]
class_prob = y_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(y_classes)
print('Predicted: %s (%.3f)'%(predict_names[0],class_prob))
print('Expected: %s' % random_face_name[0])

from matplotlib import pyplot
pyplot.imshow(random_face_pixels)
title = '%s (%.3f)' %(predict_names[0],class_prob)
pyplot.title(title)
pyplot.show()
