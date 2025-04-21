from TestApp.path import logger



def get_categories(model_name):
    try:      
        logger.info("INFO: Function - get_coatories started")
        # Check the model name and assign categories
        if model_name == "car_classification_model.h5":
            category1 = "car"
            category2 = "no_car"
        elif model_name == "blur_classification_model.h5":
            category1 = "blur"
            category2 = "no_blur"
        elif model_name == "motion_blur_sharp_classification_model.h5":
            category1 = "motionblur"
            category2 = "sharp"
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        logger.info("INFO: Function - get_coatories ended")
        return category1, category2
    except Exception as e:
        # Handle unexpected errors
        return str(e)