import numpy
import tensorflow as tf
import numpy as np
from PIL import Image
from vae import latent2im,im2latent
from matplotlib import cm
import matplotlib.pyplot as plt
import tensorflow.keras.applications.efficientnet as efn
from tensorflow.keras.models import load_model


my_model = load_model('my_model')
decoder = load_model('decoder.h5')
base_model = efn.EfficientNetB4(input_shape=(380, 380, 3), include_top=False, weights='imagenet')
for layer in base_model.layers:
    layer.trainable = False
my_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
def addlists(list1,list2):
    list3 = []
    for i in range(0,len(list2)):
        list3.append(list2[i] + list1[i])
    return list3
def cal_pop_fitness(inputs, pop,index_class):
    fitness = []
    for genum in pop:
        final_list = []
        for i in range(0, 10):
            final_list.append(inputs[i][0] * genum[i])
        output = addlists(final_list[0], final_list[1])
        for i in range(2, 10):
            output = addlists(output, final_list[i])
        output = np.array(output, dtype=np.float32)
        output = np.reshape(output, (1, 20))
        im2 = latent2im(decoder, output)
        image = Image.fromarray(np.uint8(im2*255))
        newsize = (380, 380)
        image = image.resize(newsize)
        image = numpy.asarray(image)
        arr = np.zeros((1,380,380,3))
        arr[0] = image
        features = base_model.predict(arr)
        features = np.reshape(features, (1, 12 * 12 * 1792))
        predict_label = my_model.predict(features)
        fitness.append(predict_label[0][index_class])
    return np.array(fitness)

def select_mating_pool(pop, fitness, num_parents):
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover):
    for idx in range(offspring_crossover.shape[0]):
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
    return offspring_crossover