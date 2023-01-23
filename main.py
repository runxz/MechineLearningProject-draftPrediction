# import necessary libraries
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

hero_dict = {
    "aamon": 0,
    "akai": 1,
    "aldous": 2,
    "alice": 3,
    "alpha": 4,
    "alucard": 5,
    "angela": 6,
    "argus": 7,
    "atlas": 8,
    "aulus": 9,
     "aurora": 10,
    "badang": 11,
    "balmond": 12,
    "bane": 13,
    "barats":14,
    "baxia": 15,
    "beatrix": 16,
    "belerick": 17,
    "benedetta": 18,
    "brody": 19,
     "bruno": 20,
    "carmilla": 21,
    "cecilion": 22,
    "change": 23,
    "chou": 24,
    "claude": 25,
    "clint": 26,
    "cyclops": 27,
    "diggie": 28,
    "dyrroth": 29,
    "edith":30,
     "esmeralda": 31,
    "estes": 32,
    "eudora": 33,
    "fanny":34,
    "faramis": 35,
    "floryn": 36,
    "franko": 37,
    "fredrin": 38,
    "freya": 39,
     "gatotkaca": 40,
    "gloo": 41,
    "gord": 42,
    "granger": 43,
    "grock": 44,
    "guinevere": 45,
    "gusion": 46,
    "hanabi": 47,
    "hanzo": 48,
    "harith": 49,
    "harley":50,
     "hayabusa": 51,
    "helcurt": 52,
    "hilda": 53,
    "hylos":54,
    "irithel": 55,
    "jawhead": 56,
    "jhonson": 57,
    "joy": 58,
    "julian": 59,
     "kadita": 60,
    "kagura": 61,
    "kaja": 62,
    "karina": 63,
    "karrie": 64,
    "khaleed": 65,
    "khufra": 66,
    "kimmy": 67,
    "lancelot": 68,
    "lapulapu": 69,
    "layla":70,
     "leomord": 71,
    "lesley": 72,
    "ling": 73,
    "lolita":74,
    "lunox": 75,
    "luoyi": 76,
    "lylia": 77,
    "martis": 78,
    "masha": 79,
     "mathilda": 80,
    "melissa": 81,
    "minotaur": 82,
    "minsitthar": 83,
    "miya": 84,
    "moskov": 85,
    "nana": 86,
    "natalia": 87,
    "natan": 88,
    "odette": 89,
    "paquito":90,
     "pharsa": 91,
    "phoveus": 92,
    "popol": 93,
    "rafaela":94,
    "roger": 95,
    "ruby": 96,
    "saber": 97,
    "selena": 98,
    "silvana": 99,
     "sun": 100,
    "terizla": 101,
    "thamuz": 102,
    "tigril": 103,
    "uranus": 104,
    "vale": 105,
    "valentina": 106,
    "valir": 107,
    "vexana": 108,
    "wanwan": 109,
    "xavier":110,
     "xborg": 111,
    "yin": 112,
    "yishunshin": 113,
    "yuzhong":114,
    "yvee": 115,
    "zhask": 116,
    "zilong": 117,
    "win":118,
    "lose":119
}
# dictionary to map integer labels to names
label_dict = {value: key for key, value in hero_dict.items()}

def load_data():
    # list to store the preprocessed images
    processed_images = []
    labels = []

    for hero in os.listdir('hero_faces'):
        for filename in os.listdir(f'hero_faces/{hero}'):
            # use hero as the key to lookup the label in the hero_dict dictionary
            label = hero_dict[hero]
            img = cv2.imread(f'hero_faces/{hero}/{filename}')
            img = cv2.resize(img, (100, 100))
            # convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # normalize the image
            normalized_img = gray / 255.0
            # add the preprocessed image to the list
            processed_images.append(normalized_img)
            labels.append(label)
            # use the label_dict to look up the name of the label
            label_name = label_dict[label]
            
    
    # return the preprocessed data
    return processed_images, labels


# load and preprocess the data
X, y = load_data()

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# reshape the data for the CNN
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# convert the labels to numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# create the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(hero_dict), activation='softmax'))

# compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# save the trained model to an HDF5 file
model.save('model.h5')

