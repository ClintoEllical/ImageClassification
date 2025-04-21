import os
import shutil  
import numpy as np
from TestApp.path import logger
from TestApp.path import OUTPUT_DIR
from TestApp.path import MODELS_DIR, DATASET_DIR
from TestApp.services.dynamic_service import get_categories
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array



def classify_images(model_name, dataset_folder):
    try:
        logger.info("INFO : Function classify_service started")
        dataset_path = os.path.join(DATASET_DIR, dataset_folder)
        model_path   = os.path.join(MODELS_DIR, model_name)

        # Check if paths exist
        if not (os.path.exists(dataset_path) and os.path.exists(model_path)):
            raise FileNotFoundError(f"Either dataset path '{dataset_path}' or model path '{model_path}' does not exist.")
        
        # Load the saved model
        model = load_model(model_path)
        #print("Summary",model.summary())

        # Prepare results folder
        last_folder_name = os.path.basename(dataset_path)
        results_path     = os.path.join(OUTPUT_DIR, last_folder_name)
        class0, class1   = get_categories(model_name)
        # print("Categoreies obtained")

        # Create folders for classification results
        class0_folder = os.path.join(results_path, class0)
        class1_folder = os.path.join(results_path, class1)
        os.makedirs(class0_folder, exist_ok=True)
        os.makedirs(class1_folder, exist_ok=True)

        # Define image size expected by the model
        image_size = (224, 224)  # Update this to your model's input size

        # Loop through images in the dataset
        for image_file in os.listdir(dataset_path):
            image_path = os.path.join(dataset_path, image_file)
            if not os.path.isfile(image_path):
                continue  # Skip if it's not a file

            # Preprocess image
            img       = load_img(image_path, target_size=(150, 150))  # Resize to match Conv2D input
            img_array = img_to_array(img) / 255.0  # Normalize pixel values
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict class
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            prediction = model.predict(img_array)
            # print("Prediction output:", prediction)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_class = 0 if prediction[0][0] > 0.7 else 1
            # print("Predicted Class:", predicted_class,image_file)

            # Copy the image to the corresponding folder
            if predicted_class == 1:
                shutil.copy(image_path, os.path.join(class0_folder, image_file))
            elif predicted_class == 0:
                shutil.copy(image_path, os.path.join(class1_folder, image_file))

        logger.info("INFO : Function classify_service Ended")
        return results_path

    except Exception as e:
        raise e
