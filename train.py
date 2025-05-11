#train.py

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from cnn_model_1 import build_cnn_model_1
from cnn_model_2 import build_cnn_model_2
from vgg16_transfer import build_vgg16_transfer_model

def train_model(model_name, x_train, y_train, x_val, y_val, input_shape, num_classes):
    if model_name == 'cnn1':
        model = build_cnn_model_1(input_shape, num_classes)
    elif model_name == 'cnn2':
        model = build_cnn_model_2(input_shape, num_classes)
    elif model_name == 'vgg16':
        model = build_vgg16_transfer_model(input_shape, num_classes)
    else:
        raise ValueError("Unknown model name")

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(f"best_{model_name}.h5", save_best_only=True)
    ]

    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(x_val, y_val),
        callbacks=callbacks
    )
    return model, history
