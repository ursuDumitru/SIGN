import matplotlib.pyplot as plt
import numpy as np
import os

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from BuildLanguageModels import ModelDynamic

TRY = "_3"
ATTEMPT = f"{TRY}_dropout_allData_earlyStopping_50Patience"

base_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sign_labels_file_path = base_dir + f"data/dynamic/sign_labels/sign_labels{TRY}.csv"
data_set_path = base_dir + f"data/dynamic/data_set/data_set{TRY}"
model_save_path = base_dir + f"models/dynamic/model_dynamic{ATTEMPT}.h5"
model_summary_save_path = base_dir + f"images/models/dynamic/summary/model_static{ATTEMPT}.txt"

model = ModelDynamic(sign_labels_file_path=sign_labels_file_path,
                     data_set_path=data_set_path,
                     model_save_path=model_save_path,
                     random_state=55)

x_train, x_test, y_train, y_test = model.load_data_set()

checkpoint = ModelCheckpoint(model.model_save_path, verbose=1,
                             save_weights_only=False)
early_stopping = EarlyStopping(patience=50, verbose=1)
tensor_board = TensorBoard(log_dir='./logs', histogram_freq=1)

model.model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'])

results = model.model.fit(x_train,
                          y_train,
                          epochs=1500,
                          validation_data=(x_test, y_test),
                          callbacks=[tensor_board, checkpoint,
                                     early_stopping])

val_loss, val_acc = model.model.evaluate(x_test, y_test,
                                         batch_size=16)

model.save_model()

# plot val and train loss
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Pierderea Modelului')
plt.ylabel('Pierdere')
plt.xlabel('Epocă')
plt.legend(['Antrenare', 'Validare'], loc='upper right')
plt.savefig(base_dir + f'images/models/dynamic/loss/loss{ATTEMPT}.png')

plt.close('all')

# plot val and train accuracy
plt.plot(results.history['categorical_accuracy'])
plt.plot(results.history['val_categorical_accuracy'])
plt.title('Acuratețea Modelului')
plt.ylabel('Acuratețe')
plt.xlabel('Epocă')
plt.legend(['Antrenare', 'Validare'], loc='upper left')
plt.savefig(base_dir + f'images/models/dynamic/accuracy/acc{ATTEMPT}.png')

with open(model_summary_save_path, 'w') as f:
    model.model.summary(print_fn=lambda x: f.write(x + '\n'))
