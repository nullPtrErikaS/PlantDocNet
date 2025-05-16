import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

IMG_SIZE = (224, 224)

# Load model and class labels
model = tf.keras.models.load_model('model/plantdocnet.h5')
class_labels = list(model.class_names) if hasattr(model, 'class_names') else ["Class index " + str(i) for i in range(model.output_shape[1])]

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    
    preds = model.predict(x)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    print(f"Predicted Class: {class_labels[class_idx]} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict_image(sys.argv[1])
