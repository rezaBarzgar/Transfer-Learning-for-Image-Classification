import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers

train_features =  np.load('train_features2.npy')
train_labels = np.load('train_labels2.npy')
test_features  = np.load('test_features2.npy')
test_labels = np.load('test_labels2.npy')

train_features = np.reshape(train_features, (532, 12 * 12 * 1792))
test_features = np.reshape(test_features, (380, 12 * 12 * 1792))

# train_features = np.concatenate((train_features,train_features),axis=0)
# train_labels = np.concatenate((train_labels,train_labels),axis=0)

callback = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy', min_delta=0, patience=4, verbose=0,
    mode='max', baseline=None, restore_best_weights=True
)

model = tf.keras.models.Sequential()
model.add(layers.Dense(1024, activation='sigmoid', input_dim=12 * 12 * 1792))
model.add(layers.Dropout(0.5))
# model.add(layers.Dense(256, activation='sigmoid', input_dim=10 * 10 * 1280))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='sigmoid', input_dim=12 * 12 * 1792))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(19, activation='softmax', input_dim=12 * 12 * 1792))

model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])

history = model.fit(train_features, train_labels,callbacks=[callback],epochs=20)

loss, acc = model.evaluate(test_features,test_labels)
model.save("my_model")
print("Accuracy", acc)