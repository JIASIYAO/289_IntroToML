from data_manager_pytorch import data_manager
from cnn_pytorch import CNN
from trainer_pytorch import Solver
from viz_features_pytorch import Viz_Feat
import random


import matplotlib.pyplot as plt

CLASS_LABELS = ['apple','banana','nectarine','plum','peach','watermelon','pear','mango','grape','orange','strawberry','pineapple', 
    'radish','carrot','potato','tomato','bellpepper','broccoli','cabbage','cauliflower','celery','eggplant','garlic','spinach','ginger']

LITTLE_CLASS_LABELS = ['apple','banana','eggplant']

image_size = 90

random.seed(0)

classes = CLASS_LABELS
dm = data_manager(classes, image_size)

cnn = CNN(classes,image_size)

solver = Solver(cnn,dm)

solver.optimize()

plt.plot(solver.test_accuracy,label = 'Validation')
plt.plot(solver.train_accuracy, label = 'Training')
plt.legend()
plt.xlabel('Iterations (in 200s)')
plt.ylabel('Accuracy')
plt.show()

val_data = dm.val_data
train_data = dm.train_data

cm = Viz_Feat(val_data,train_data,CLASS_LABELS)

cm.vizualize_features(cnn)




