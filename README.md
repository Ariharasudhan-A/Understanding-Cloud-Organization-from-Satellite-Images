# Understanding-Cloud-Organization-from-Satellite-Images
![Understanding Clouds from Satellite Images](https://github.com/user-attachments/assets/4ad8b761-9d95-4e0b-9768-488cccd68267)

## Overview

The objective of this project is to build a Deep learning model to classify cloud organization patterns from satellite images, which is crucial for understanding and predicting climate changes.

The project involves:

- **Dataset**: https://www.kaggle.com/competitions/understanding_cloud_organization/overview
  
- **Data Preprocessing**: Dataset Splitting, Decoding EncodedPixels, Resizing & Normalization, Data Augmentation, Conversion to Tensor.
  
- **Modeling**: Implemented a **U-Net** architecture with **ResNet-34** backbone for **multi-class semantic segmentation** of cloud formation patterns from satellite images.
  
- **Evaluation and Visualization**: Model performance is evaluated using **Accuracy** and **Dice Coefficient**.

## Dataset

The images were downloaded from **NASA Worldview**. Three regions, spanning **21 degrees longitude** and **14 degrees latitude**, were chosen. The true-color images were taken from two polar-orbiting satellites, **TERRA and AQUA**, each of which passes a specific region once a day.

Each cloud formation label in an image is encoded using **Run-Length Encoding (RLE)**. If a cloud type is absent from an image, its `EncodedPixels` field is left blank.

The formations are categorized into the following labels:

- **Fish**
- **Flower**
- **Gravel**
- **Sugar**

## Data Preprocessing

To ensure high-quality input data for the model, the following preprocessing steps were performed:

- **Dataset Splitting**:
   - The dataset was divided into **Train (80%)**, **Validation (10%)**, and **Test (10%)** sets.

- **Decoding EncodedPixels**:
   - The dataset provides **EncodedPixels** for segmentation masks, which were decoded into **multi-class binary masks** for each cloud type.
   - If a particular cloud type was not present in an image, its corresponding mask was left blank.

- **Resizing & Normalization**:
   - All satellite images were resized to **(256x256)** for consistency.
   - Pixel values were normalized to **[0,1]** by dividing by 255 for better model convergence.

- **Data Augmentation** (Applied to enhance generalization):
   - **Horizontal & Vertical Flipping**
   - **Elastic Transformations** (to account for natural distortions in cloud formations)

- **Conversion to Tensor**:
   - Images and masks were converted to PyTorch tensors for seamless model training.

## Model Overview

This implementation uses a U-Net architecture for semantic segmentation with the following structure:

- **Encoder**: A ResNet-34 backbone, pre-trained on ImageNet.

- **Decoder**: A series of convolutional layers followed by upsampling to reconstruct the segmentation mask.

- **Output Layer**: A 1x1 convolution to predict pixel-wise class labels (5 classes: 1 for background + 4 cloud types).

- **Hyperparameters**:
  - Learning Rate: 1e-4 (Dynamic)
  - Optimizer: Adam
  - Loss Function: CrossEntropyLoss (for multi-class segmentation)

## Results

We Achieved an **Accuracy of 50.45% and a Dice score of 0.5917** in the segmentation process, highlighting the model's effectiveness.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any queries or contributions, feel free to reach out to:
- **Ariharasudhan A** - [Email](mailto:ariadaikalam1234@gmail.com)
- **Harish R** - [Email](mailto:harishsekar2004@gmail.com)
