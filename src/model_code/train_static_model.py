import matplotlib.pyplot as plt
import os

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from SignLanguageModels import ModelStatic

TRY = "_5"
ATTEMPT = f"{TRY}_test_1"

base_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sign_labels_file_path = base_dir + f"data/static/sign_labels/sign_labels{TRY}.csv"
data_set_path = base_dir + f"data/static/data_set/data_set{TRY}.csv"
model_save_path = base_dir + f"models/static/model_static{ATTEMPT}.h5"
model_summary_save_path = base_dir + f"images/model_summary/model_static{ATTEMPT}.txt"

model = ModelStatic(sign_labels_file_path=sign_labels_file_path,
                    data_set_path=data_set_path,
                    model_save_path=model_save_path,
                    random_state=55)

x_train, x_test, y_train, y_test = model.load_data_set()

# model.model.summary()
with open(model_summary_save_path, 'w') as f:
    model.model.summary(print_fn=lambda x: f.write(x + '\n'))

exit(1)

checkpoint = ModelCheckpoint(model.model_save_path, verbose=1, save_weights_only=False)  # what is this ?
early_stopping = EarlyStopping(patience=20, verbose=1)  # what is this ?
tensor_board = TensorBoard(log_dir='./logs', histogram_freq=1)  # what is this ?

model.model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'])

results = model.model.fit(x_train,
                          y_train,
                          epochs=500,
                          #   batch_size=16,
                          validation_data=(x_test, y_test),
                          # callbacks=[tensor_board, checkpoint, early_stopping]
                          callbacks=[tensor_board, checkpoint])

val_loss, val_acc = model.model.evaluate(x_test, y_test, batch_size=16)

model.save_model()

plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig(base_dir + f'images/models/static/model{ATTEMPT}.png')
# plt.show()
