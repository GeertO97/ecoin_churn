"""
Script to deploy a trained model to production
"""
import joblib
import logging
import datetime
import os

# Initialize logger
logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.INFO)


def register_model(trained_model):
    """
    Save trained model object in the model_registry directory
    :param trained_model: trained_model object
    """

    # Get the current datetime
    now = datetime.datetime.now()

    # Format the datetime as a string
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Write model to joblib
    joblib.dump(trained_model, f"model_registry/trained_model_{date_string}.joblib")
    log.info("Model written to joblib")


def load_model():
    """
    Load the most recently trained model from the model registry
    :return: Trained model object
    """

    file_list = os.listdir("model_registry")
    most_recent = sorted(file_list, reverse=True)[0]
    model = joblib.load(f"model_registry/{most_recent}")
    return model


if __name__ == '__main__':
    load_model()

