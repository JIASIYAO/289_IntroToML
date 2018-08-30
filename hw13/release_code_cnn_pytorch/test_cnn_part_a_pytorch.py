from data_manager_pytorch import data_manager
from cnn_pytorch import CNN
from trainer_pytorch import Solver
import random

from confusion_mat_pytorch import Confusion_Matrix

random.seed(0)

CLASS_LABELS = ['apple','banana','nectarine','plum','peach','watermelon','pear','mango','grape','orange','strawberry','pineapple', 
    'radish','carrot','potato','tomato','bellpepper','broccoli','cabbage','cauliflower','celery','eggplant','garlic','spinach','ginger']

image_size = 90
classes = CLASS_LABELS
dm = data_manager(classes, image_size)

cnn = CNN(classes,image_size)

val_data = dm.val_data
train_data = dm.train_data


cm = Confusion_Matrix(val_data,train_data,CLASS_LABELS)

cm.test_net(cnn)

