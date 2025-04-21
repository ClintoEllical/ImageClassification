# # IMPORT
# # DEFINING PATHS
# # DATA AUGMENTATION
# # SPLITING IMAGES
# # DATA PREPROCESSING 
# # TRAIN MODEL
# # RESULTS
# # SAVE MODEL


# #imports are mentioned here
# import os
# import random
# import shutil
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras import models, layers, regularizers
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten, Dense, Input,Conv2D, MaxPooling2D
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




# # Define source directory and output folders
# # classs 1: motion_blur
# motion_blur_source_dir = r'C:\Users\clint\OneDrive\coding\Django_backend\dataset_train\motion_blur'
# motion_blur_train_dir = r'C:\Users\clint\OneDrive\coding\Django_backend\dataset_train\motionblursharp_split\train\motion_blur_train'
# motion_blur_val_dir = r'C:\Users\clint\OneDrive\coding\Django_backend\dataset_train\motionblursharp_split\validate\motion_blur_validate'
# motion_blur_test_dir = r'C:\Users\clint\OneDrive\coding\Django_backend\dataset_train\motionblursharp_split\test\motion_blur_test'
# # class 2 : sharp images
# sharp_source_dir = r'C:\Users\clint\OneDrive\coding\Django_backend\dataset_train\sharp'
# sharp_train_dir = r'C:\Users\clint\OneDrive\coding\Django_backend\dataset_train\motionblursharp_split\train\sharp_train'
# sharp_val_dir = r'C:\Users\clint\OneDrive\coding\Django_backend\dataset_train\motionblursharp_split\validate\sharp_validate'
# sharp_test_dir = r'C:\Users\clint\OneDrive\coding\Django_backend\dataset_train\motionblursharp_split\test\sharp_test'
# # Create the directories if they don't exist
# os.makedirs(motion_blur_train_dir, exist_ok=True)
# os.makedirs(motion_blur_val_dir, exist_ok=True)
# os.makedirs(motion_blur_test_dir, exist_ok=True)
# os.makedirs(sharp_train_dir, exist_ok=True)
# os.makedirs(sharp_val_dir, exist_ok=True)
# os.makedirs(sharp_test_dir, exist_ok=True)



# # DATA AUGMENTATION
# # Paths to original dataset directories
# source_dir = r'C:\Users\clint\OneDrive\coding\Django_backend\dataset_train\testing'
# augmented_dir = r'C:\Users\clint\OneDrive\coding\Django_backend\augmented_images\sharp'
# # Create augmented directory if it doesn't exist
# os.makedirs(augmented_dir, exist_ok=True)

# # Initialize ImageDataGenerator with augmentation options
# augment_datagen = ImageDataGenerator(
#     rotation_range=20,      # Rotate images by up to 20 degrees
#     width_shift_range=0.2,  # Shift image horizontally by 20%
#     height_shift_range=0.2, # Shift image vertically by 20%
#     shear_range=0.2,        # Shear the image
#     zoom_range=0.2,         # Zoom in/out
#     horizontal_flip=True,   # Flip the image horizontally
#     fill_mode='nearest'     # Fill missing pixels after transformations
# )
# # Generate augmented images and save them to disk
# augment_generator = augment_datagen.flow_from_directory(
#     source_dir,
#     target_size=(150, 150),  # Resize images to 150x150
#     batch_size=1,            # Process one image at a time
#     save_to_dir=augmented_dir,  # Directory where augmented images are saved
#     save_prefix='aug',         # Prefix for saved images
#     save_format='jpeg'         # Format of saved images
# )
# # Save 500 augmented images
# for i in range(300):  # Change the number to your desired count
#     next(augment_generator)  # Generate and save augmented image

# print("Augmented images successfully created and saved!")





# # SPLITING IMAGES
# def split_data(source_dir,train_dir,val_dir,test_dir):
#     # Get all files in the source directory and shuffle them
#     split_ratio=(0.7, 0.2, 0.1)
#     file_names = os.listdir(source_dir)
#     random.shuffle(file_names)
#     # Compute split sizes
#     train_split = int(len(file_names) * split_ratio[0])
#     val_split = train_split + int(len(file_names) * split_ratio[1]) 
#     # Split the files
#     train_files = file_names[:train_split]
#     val_files = file_names[train_split:val_split]
#     test_files = file_names[val_split:]
#     # Copy files to the respective directories
#     for file in train_files:
#         shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))
#     for file in val_files:
#         shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, file))
#     for file in test_files:
#         shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, file))

# # Function to split the dataset 
# split_data(motion_blur_source_dir, motion_blur_train_dir, motion_blur_val_dir, motion_blur_test_dir)
# split_data(sharp_source_dir, sharp_train_dir, sharp_val_dir, sharp_test_dir)

# # After splitting the data, count the files in each folder
# print(f"Train - motion_blur: {len(os.listdir(motion_blur_train_dir))}")
# print(f"Validate - motion_blur: {len(os.listdir(motion_blur_val_dir))}")
# print(f"Test - motion_blur: {len(os.listdir(motion_blur_test_dir))}")

# print(f"Train - sharp: {len(os.listdir(sharp_train_dir))}")
# print(f"Validate - sharp: {len(os.listdir(sharp_val_dir))}")
# print(f"Test - sharp: {len(os.listdir(sharp_test_dir))}")




# # DATA PREPROCESSING 
# # Paths to dataset directories
# train_dir = r'C:\Users\clint\OneDrive\coding\Django_backend\dataset_train\motionblursharp_split\train'
# val_dir = r'C:\Users\clint\OneDrive\coding\Django_backend\dataset_train\motionblursharp_split\validate'
# test_dir = r'C:\Users\clint\OneDrive\coding\Django_backend\dataset_train\motionblursharp_split\test'

# # Data preprocessing

# # Data augmentation for training
# train_datagen = ImageDataGenerator(
#     rescale=1./255,  # Normalize pixel values
#     rotation_range=20,  # Rotate images by up to 20 degrees
#     width_shift_range=0.2,  # Shift image horizontally by 20%
#     height_shift_range=0.2,  # Shift image vertically by 20%
#     shear_range=0.2,  # Shear the image
#     zoom_range=0.2,  # Zoom in/out
#     horizontal_flip=True,  # Flip the image horizontally
#     fill_mode='nearest'  # Fill any missing pixels after transformation
# )
# val_test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(150, 150),
#     batch_size=4,
#     class_mode='binary'
# )
# val_generator = val_test_datagen.flow_from_directory(
#     val_dir,
#     target_size=(150, 150),
#     batch_size=4,
#     class_mode='binary'
# )
# test_generator = val_test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(150, 150),
#     batch_size=4,
#     class_mode='binary',
#     shuffle=False  # Important for evaluation
# )



# # TRAIN MODEL
# # Create a simple model
# model = Sequential([
#     Input(shape=(150, 150, 3)),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])
# # Compile the model
# model.compile(
#     optimizer=Adam(learning_rate=0.0001),
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )
# # Train the model
# history = model.fit(
#     train_generator,
#     steps_per_epoch=len(train_generator),
#     epochs=15,
#     validation_data=val_generator,  # Include validation data
#     validation_steps=1 # Adjust as needed
# )
# # Evaluate the model on the test set
# test_loss, test_accuracy = model.evaluate(test_generator)
# print(f"Test Accuracy: {test_accuracy:.2f}")




# # RESULTS
# # Step 6: Visualize Training Progress
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.legend()
# plt.show()
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.legend()
# plt.show()
# # Step 7: Test the Model
# test_loss, test_accuracy = model.evaluate(test_generator)
# print(f"Test Accuracy: {test_accuracy:.2f}")

# # Step 1: Predict labels for the test set
# y_true = test_generator.classes  # True labels from the test generator
# y_pred = model.predict(test_generator)  # Predicted probabilities
# y_pred = np.round(y_pred).astype(int)  # Convert probabilities to binary labels

# # Step 2: Generate confusion matrix
# cm = confusion_matrix(y_true, y_pred)

# # Step 3: Visualize the confusion matrix
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())
# disp.plot(cmap='Blues')  # Use 'Blues' colormap for better visualization




# # SAVE MODEL
# model.save(r'C:\Users\clint\OneDrive\coding\Django_backend\models\motion_blur_classification_model.h5')
