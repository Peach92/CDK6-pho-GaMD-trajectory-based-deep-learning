#!/bin/python3
from matplotlib import pyplot
import pickle

pyplot.style.use('seaborn-v0_8')

metrics_file = open('./metrics/metrics.pkl', 'rb')
history = pickle.load(metrics_file)
metrics_file.close()

pyplot.subplot(211)

#pyplot.title('Loss')
pyplot.plot(history['loss'], label='Train')
pyplot.plot(history['val_loss'], label='Validation')
print("loss------------")
print(history['val_loss'])
pyplot.xticks(fontsize=10, fontweight='bold')
pyplot.yticks(fontsize=10, fontweight='bold')
print(history['val_loss'])
#pyplot.xlabel('Epoch', fontsize=22, fontweight='bold')
pyplot.ylabel('Loss', fontsize=20, fontweight='bold')
pyplot.legend(fontsize=12)

pyplot.subplot(212)

#pyplot.title('Accuracy')
pyplot.plot(history['accuracy'], label='Train')
pyplot.plot(history['val_accuracy'], label='Validation')
pyplot.xticks(fontsize=10, fontweight='bold')
pyplot.yticks(fontsize=10, fontweight='bold')
pyplot.xlabel('Epoch', fontsize=18, fontweight='bold')
pyplot.ylabel('Accuracy', fontsize=20, fontweight='bold')
pyplot.legend(fontsize=12)
print("accuracy--------")
print(history['val_accuracy'])




pyplot.savefig('./metrics/metrics.jpg', dpi=600, bbox_inches='tight')