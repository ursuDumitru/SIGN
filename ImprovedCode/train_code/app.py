import matplotlib.pyplot as plt

from sign_language_model import Model

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


ATTEMPT = '19_abc_4'

model = Model(
    sign_labels_file_path = './data/sign_labels_abc_2.csv',
    data_set_path = './data/data_set_abc_2.csv', # may need to change the path
    model_save_path = f'./models/model{ATTEMPT}.h5',
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
plt.savefig(f'./images/model{ATTEMPT}.png')
# plt.show()