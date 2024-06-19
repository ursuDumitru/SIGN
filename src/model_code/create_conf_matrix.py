import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from UseLanguageModels import ModelStatic, ModelDynamic

MODE = "dynamic" # dynamic | static
TRY = "_5" if MODE == "static" else "_3"
ATTEMPT = f"{TRY}_dropout_allData_earlyStopping_50Patience"
# noDropout_30Data + 150 epoch = best

base_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sign_labels_file_path = base_dir + f"data/{MODE}/sign_labels/sign_labels{TRY}.csv"
model_weights_file_path = base_dir + f"models/{MODE}/model_{MODE}{ATTEMPT}.h5"
if MODE == "static":
    data_set_path = base_dir + f"data/{MODE}/data_set/data_set{TRY}.csv"
else:
    data_set_path = base_dir + f"data/{MODE}/data_set/data_set{TRY}"

if MODE == "static":
    model = ModelStatic(sign_labels_file_path, data_set_path, model_weights_file_path, 55)
else:
    model = ModelDynamic(sign_labels_file_path, data_set_path, model_weights_file_path, 55)

x_train, x_test, y_train, y_test = model.load_data_set()
# x_train, x_test, y_train, y_test = model.load_data_set_first_x(50)

y_pred = model.model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)


# Plot the confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(15, 15))

disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=model.sign_labels)
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')  # Rotate labels to vertical

ax.set_xlabel('Clasificare Prezisă', fontsize=20)
ax.set_ylabel('Clasificarea Adevărată', fontsize=20)
ax.xaxis.set_tick_params(rotation=90, labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)

plt.tight_layout()
plt.savefig(base_dir + f'images/models/{MODE}/conf_matrix/conf_matrix{ATTEMPT}.png')

plt.close('all')

# Plot the classification report
report = classification_report(y_true_classes, y_pred_classes, target_names=model.sign_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df = report_df.applymap(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)

fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=report_df.values, colLabels=report_df.columns, rowLabels=report_df.index, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.5, 1.5)
table.auto_set_column_width([0])
for i in range(1, len(report_df.columns) + 1):
    table.auto_set_column_width(i)

# plt.title('Classification Report')
plt.savefig(base_dir + f'images/models/{MODE}/class_report/class_report{ATTEMPT}.png')