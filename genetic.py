import glob
import re
import PIL
import numpy as np
import cv2
import ga
import tensorflow as tf
from operator import add
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from vae import latent2im,im2latent
from PIL import Image
from numpy import asarray
import pandas as pd
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


cats = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bus': 4,
        'car': 5, 'cat': 6, 'chair': 7, 'cow': 8, 'diningtable': 9,
        'dog': 10, 'horse': 11, 'motorbike': 12, 'person': 13, 'pottedplant': 14,
        'sheep': 15, 'sofa': 16, 'train': 17, 'tvmonitor': 18}
def find_most_class(classes):
    y = set(classes)
    num = 0
    most = ''
    for i in y:
        z = classes.count(i)
        if z > num:
            num = z
            most = i
    return cats[most]
def dataFrame_creator(folder_name):
    dataset = pd.DataFrame(columns=['caption', 'label'])
    folders = glob.glob(folder_name + "\*")
    captionNames = []
    classNames = []
    for folder in folders:
        classes = glob.glob('{}\\*.txt'.format(folder))
        for caption in classes:
            with open(caption, 'r') as file:
                inside_file = file.read()
                label = str(folder).split('\\')[6]
                captionNames.append(str(caption).split('\\')[-1])
                classNames.append(str(caption).split('\\')[-2])
                lines = ''
                for line in inside_file.split('\n'):
                    if line:
                        lines += ' ' + line
                dataset = dataset.append({'caption': lines, 'label': label}, ignore_index=True)
    return dataset,captionNames,classNames


def preProcess(s):
    # stop_words = set(stopwords.words('english'))
    s = re.sub("[^a-zA-Z]", " ", s)
    s = s.lower()
    porter = SnowballStemmer(language='english')
    output = porter.stem(s)
    return output

def load_images(base_dir,captions,classes):
    image_list = []
    for i in range(0,10):
        dir = base_dir + '\\'+classes[i] + '\\' + captions[i]
        dir = dir.replace("txt","jpg")
        img = Image.fromarray(cv2.imread(dir))
        new_width = 64
        new_height = 64
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img = asarray(img)
        img = np.float32(img)
        img = img.reshape((1, np.prod(img.shape[0:])))
        image_list.append(img)
    return image_list

def vectorize(train_list, vectorizer):
    X = vectorizer.transform(train_list)
    return X.toarray()
def clasify(train_vectors, test_vector):
    neigh = NearestNeighbors(n_neighbors=10)
    neigh.fit(train_vectors)
    pred = neigh.kneighbors(test_vector)
    return pred
def addlists(list1,list2):
    list3 = []
    for i in range(0,len(list2)):
        list3.append(list2[i] + list1[i])
    return list3
def mainGA(input_list,index_class):
    weights = []
    num_weights = 10
    sol_per_pop = 8
    pop_size = (sol_per_pop,num_weights)
    new_population = np.random.uniform(low=-4.0, high=4.0, size=pop_size)
    num_generations = 100
    num_parents_mating = 4
    for generation in range(num_generations):
        fitness = ga.cal_pop_fitness(input_list, new_population,index_class)
        print(fitness)
        parents = ga.select_mating_pool(new_population, fitness, num_parents_mating)
        offspring_crossover = ga.crossover(parents,offspring_size=(pop_size[0] - parents.shape[0], num_weights))
        offspring_mutation = ga.mutation(offspring_crossover)
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
    max_fitness_idx = np.where(fitness == np.max(fitness))
    weights = new_population[max_fitness_idx][0]
    return weights
def addlists(list1,list2):
    list3 = []
    for i in range(0,len(list2)):
        list3.append(list2[i] + list1[i])
    return list3
def main():
    test_dataframe,test_captions,testNames = dataFrame_creator("D:\projects\CNN\dataset\\test\sentences")
    train_dataframe,train_captions,trainNames = dataFrame_creator("D:\projects\CNN\dataset\\train\sentences")
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, max_features=200000, ngram_range=(1, 3),
                                 sublinear_tf=True, preprocessor=preProcess)
    all = test_dataframe['caption'].tolist() + train_dataframe['caption'].tolist()
    vectorizer.fit(all)
    all_vectors = vectorize(all, vectorizer)
    all_captions = test_captions + train_captions
    all_Names = testNames + trainNames
    train_vectors = vectorize(train_dataframe['caption'].tolist(), vectorizer)
    test_vectors = vectorize(test_dataframe['caption'].tolist(), vectorizer)
    print('enter the caption you want to find:')
    user_input = input()
    user_list = []
    user_list.append(user_input)
    user_vector = vectorize(user_list, vectorizer)
    temp = clasify(all_vectors, user_vector)
    list = temp[1][0]
    captions = []
    classes = []
    for item in list:
        captions.append(all_captions[item])
        classes.append(all_Names[item])
    print(captions)
    print(classes)
    image_list = load_images('D:\projects\CNN\GA_dataset',captions,classes)
    encoder = load_model('encoder.h5')
    decoder = load_model('decoder.h5')
    encode_list = []
    for img in image_list:
        comp = im2latent(encoder, img)
        encode_list.append(comp)
    weights = mainGA(encode_list,find_most_class(classes))
    final_list = []
    for i in range(0, 10):
        final_list.append(encode_list[i][0] * weights[i])
    output = addlists(final_list[0], final_list[1])
    for i in range(2, 10):
        output = addlists(output, final_list[i])
    output = np.array(output, dtype=np.float32)
    output = np.reshape(output, (1, 20))
    im2 = latent2im(decoder, output)
    print(type(im2))
    plt.imshow(im2)
    plt.show()
if __name__ == '__main__':
    main()
