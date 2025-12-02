# Chest-X-Ray-Detection
Welcome to the GitHub repository for Chest X-Ray Detection, a powerful diagnostic tool developed from a dissertation project titled "COVID-19 Detection in Chest X-Ray Images using Explainable Boosting Algorithms". The dissertation addresses the urgent need for transparency in AI-powered diagnostic models. Combining the robustness of Convolutional Neural Networks (CNNs), the strength of eXtreme Gradient Boosting (XGBoost), and the interpretability of Gradient-weighted Class Activation Mapping (Grad-CAM), Chest X-Ray Detection delivers accurate and explainable COVID-19 predictions from chest X-ray images. Achieving an accuracy of 91.61% and an F1 score of 91.39%, this tool is designed to support healthcare professionals in the fight against the pandemic.

#Model Overview

Chest X-Ray Detection is a hybrid diagnostic tool that combines the image-processing strength of CNNs, the robust ensemble learning of XGBoost, and the interpretability of Grad-CAM. This integration results in a comprehensive, efficient, and transparent system for detecting six chest-related diseases.

**Convolutional Neural Networks (CNNs):** CNNs excel at processing image data. They automatically extract and learn features from raw images using filters and layered architectures, which are then fed into the classification pipeline.

XGBoost: XGBoost is a high-performance gradient boosting library that iteratively learns from previous errors. By combining multiple weak learners, it enhances prediction accuracy and generalization.

Grad-CAM: Gradient-weighted Class Activation Mapping (Grad-CAM) provides visual explanations for model predictions by generating heatmaps that highlight the critical regions in an image influencing the decision.

Methodology

Data Preprocessing: Normalizes and standardizes chest X-ray images to improve model learning.

Data Augmentation: Expands the training set through techniques such as rotation, flipping, brightness adjustment, and zooming to improve generalization.

Dataset Preparation: Uses ImageDataGenerator to organize and stream data for training, validation, and testing.

CNN Feature Extraction: Employs pre-trained CNN models (VGG16, ResNet50, InceptionV3) to extract deep features from images. These features are then passed to XGBoost for classification.

XGBoost Hyperparameter Tuning and Feature Selection: Uses Bayesian optimization with cross-validation to find optimal hyperparameters while selecting the most predictive features.

Hybrid Model Evaluation: Assesses performance using metrics such as accuracy, precision, recall, F1 score, specificity, and AUC-ROC.

Grad-CAM Visualization: Generates heatmaps to interpret the CNN model’s decision-making process.

Chest X-Ray Verification: Confirms that the uploaded image is a chest X-ray using a specially-trained VGG16 model before classification, preventing errors from incorrect inputs.

Together, these steps create a model that is accurate, efficient, and interpretable, providing insight into its diagnostic decisions.
