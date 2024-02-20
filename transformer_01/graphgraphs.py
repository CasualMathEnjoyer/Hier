import pickle
import matplotlib.pyplot as plt

model_file_name = "model_to_delete"

with open(model_file_name + '_HistoryDict', "rb") as file_pi:
    history = pickle.load(file_pi)

print(list(history.keys()))
print(list(history))
print(history)

plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for precission and recall
plt.plot(history['precision'])
plt.plot(history['recall'])
plt.plot(history['val_precision'])
plt.plot(history['val_recall'])
plt.title('model precision/recall')
plt.ylabel('score')
plt.xlabel('epoch')
plt.legend(['precision', 'recall', 'val_precision', 'val_recall'], loc='upper left')
plt.show()


# summarize history for F1
plt.plot(history['F1_score'])
plt.plot(history['val_F1_score'])
plt.title('model F1')
plt.ylabel('F1 score')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()