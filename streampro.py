import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


def main():
    st.title("Robot to Predict CIFAR-10")
    st.write("Upload an image to classify it with the CIFAR-10 model")

    file = st.file_uploader("Please upload the image", type=['jpg', 'png'])

    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        # preprocess
        resized_image = image.resize((32, 32))
        img_array = np.array(resized_image) / 255.0
        img_array = img_array.reshape(1, 32, 32, 3)

        # load model
        model = tf.keras.models.load_model('cifar10_model.keras')

        # prediction
        predictions = model.predict(img_array)

        cifar10_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        # plot
        fig, ax = plt.subplots()
        y_pos = np.arange(len(cifar10_classes))
        ax.barh(y_pos, predictions[0])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cifar10_classes)
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        ax.set_title("CIFAR-10 Prediction")

        st.pyplot(fig)

    else:
        st.text("You have not uploaded an image")


if __name__ == '__main__':
    main()
