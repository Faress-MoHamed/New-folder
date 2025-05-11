import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from train import train_model
from evaluate import evaluate_model
from predict import predict_image

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

flower_class_indices = [0, 1, 2, 3, 4]

train_filter = np.isin(y_train.flatten(), flower_class_indices)
x_train = x_train[train_filter]
y_train = y_train[train_filter]

test_filter = np.isin(y_test.flatten(), flower_class_indices)
x_test = x_test[test_filter]
y_test = y_test[test_filter]

y_train = np.array([flower_class_indices.index(label[0]) for label in y_train])
y_test = np.array([flower_class_indices.index(label[0]) for label in y_test])

y_train_cat = to_categorical(y_train, num_classes=5)
y_test_cat = to_categorical(y_test, num_classes=5)

x_train, x_val, y_train_cat, y_val_cat = train_test_split(x_train, y_train_cat, test_size=0.2)

input_shape = x_train.shape[1:]
num_classes = 5
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model_name = 'cnn2'
model, history = train_model(model_name, x_train, y_train_cat, x_val, y_val_cat, input_shape, num_classes)

evaluate_model(model, x_test, y_test_cat, class_names)

result = predict_image(model, "x.jpg", class_names, target_size=input_shape[:2])
print("Predicted class:", result)
