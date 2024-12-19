import numpy as np
import os


def load_data_standardized(feature_choice):
    """
    Dynamically load data based on the feature choice.
    Options:
    - "2-body": Use 2-body features only
    - "3-body": Use 3-body features only
    - "4-body": Use 4-body features only
    - "2+3-body": Combine 2-body and 3-body features
    - "3+4-body": Combine 3-body and 4-body features
    - "2+3+4-body": Combine all features
    """
    current_dir = os.getcwd()

    data_paths = {
    "energies": os.path.join(current_dir, "data", "energies.npy"),
    "2-body": os.path.join(current_dir, "data", "features_2b.npy"),
    "3-body": os.path.join(current_dir, "data", "features_3b.npy"),
    "4-body": os.path.join(current_dir, "data", "features_4b.npy"), 
}
    energies = np.load(data_paths["energies"])

    if feature_choice == "2-body":
        features = np.load(data_paths["2-body"])
        features = (features - features.mean())/features.std()
    elif feature_choice == "3-body":
        features = np.load(data_paths["3-body"])
        features = (features - features.mean())/features.std()
    elif feature_choice == "4-body":
        features = np.load(data_paths["4-body"])
        features = (features - features.mean())/features.std()
    elif feature_choice == "2+3-body":
        features_2 = np.load(data_paths["2-body"])
        features_2 =(features_2 - features_2.mean())/features_2.std()
        features_3 = np.load(data_paths["3-body"])
        features_3 = (features_3 - features_3.mean())/features_3.std()
        features = np.hstack((features_2, features_3))
    elif feature_choice == "4+3-body":
        features_4 = np.load(data_paths["4-body"])
        features_4 = (features_4 - features_4.mean())/features_4.std()
        features_3 = np.load(data_paths["3-body"])
        features_3 = (features_3 - features_3.mean())/features_3.std()
        features = np.hstack((features_4, features_3))
    elif feature_choice == "2+3+4-body":
        features_2 = np.load(data_paths["2-body"])
        features_2 = (features_2 - features_2.mean())/features_2.std()
        features_3 = np.load(data_paths["3-body"])
        features_3 = (features_3 - features_3.mean())/features_3.std()
        features_4 = np.load(data_paths["4-body"])
        features_4 = (features_4 - features_4.mean())/features_4.std()
        features = np.hstack((features_2, features_3, features_4))
    else:
        raise ValueError("Invalid feature choice!")

    energies = (energies - energies.mean())/energies.std()
    
    return features, energies
