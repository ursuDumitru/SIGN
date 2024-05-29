import matplotlib.pyplot as plt
import os

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from train_code.sign_language_model import Model


ATTEMPT = '_abc_3'

base_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sign_labels_file_path = base_dir + 'data/sign_labels/sign_labels_abc_3.csv'
data_set_file_path = base_dir + 'data/data_set/data_set_abc_3.csv'
model_save_path = base_dir + f'models/model{ATTEMPT}.h5'

model = Model(
    sign_labels_file_path = sign_labels_file_path,
    data_set_path = data_set_file_path,  # may need to change the path
    model_save_path = model_save_path,
    random_state=0
    )

X_train, X_test, y_train, y_test = model.load_data_set()

model.model.summary()

checkpoint = ModelCheckpoint(model.model_save_path, verbose=1, save_weights_only=False) # what is this ?
early_stopping = EarlyStopping(patience=20, verbose=1) # what is this ?
tensor_board = TensorBoard(log_dir='./logs', histogram_freq=1) # what is this ?

model.model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['categorical_accuracy']
)

results = model.model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=128, # may wanna try to change this too
    validation_data=(X_test, y_test),
    # callbacks=[tensor_board, checkpoint, early_stopping]
    callbacks=[tensor_board, checkpoint]
)

val_loss, val_acc = model.model.evaluate(X_test, y_test, batch_size=64)

model.save_model()

plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig(base_dir + f'images/models/model{ATTEMPT}.png')
# plt.show()