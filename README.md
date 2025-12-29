# Dissertation_Water_Body

# Comparative Analysis of Deep Learning–Based Semantic Segmentation Models for Water Body Extraction
# Project Overview

This project focuses on the automatic extraction of surface water bodies from satellite remote sensing imagery using deep learning–based semantic segmentation techniques. Accurate water body detection is crucial for applications such as flood monitoring, drought assessment, hydrological analysis, and climate change studies. Traditional index-based approaches like the Normalised Difference Water Index (NDWI) often fail in complex environments due to spectral confusion, shadows, and seasonal variability (Xu, 2006; Chen et al., 2020). To address these limitations, this study evaluates advanced deep learning models capable of pixel-level classification.

# Objectives

The primary objective of this project is to design, implement, and compare multiple deep learning semantic segmentation models for surface water extraction. Specifically, the project aims to:

Extract water bodies at pixel level from satellite images.

Compare the performance of FCN, U-Net, DeepLabV3-Lite, and Attention-Enhanced Residual U-Net models.

Evaluate models using standard segmentation metrics such as Accuracy, IoU, Dice Coefficient, and F1-score.

Identify the most robust architecture for complex and heterogeneous remote sensing imagery.

# Models Implemented

Four deep learning architectures are implemented and evaluated under identical experimental conditions:

Fully Convolutional Network (FCN): An early semantic segmentation model enabling end-to-end pixel-wise prediction, but limited by coarse boundary outputs (Long et al., 2015).

U-Net: An encoder–decoder CNN with skip connections that preserves fine spatial details, making it effective for water body boundary delineation (Ronneberger et al., 2015).

DeepLabV3-Lite: A lightweight variant of DeepLabV3 using atrous convolutions and ASPP for multi-scale contextual feature extraction (Chen et al., 2018).

Attention-Enhanced Residual U-Net: An improved U-Net architecture integrating residual connections for stable deep training and attention mechanisms to focus on hydrologically relevant regions (Liu et al., 2022).

# Dataset Description

The experiments are conducted using the Satellite Images of Water Bodies dataset, which contains 2,841 image–mask pairs with pixel-level ground truth annotations. The dataset includes diverse water body types such as rivers, lakes, and reservoirs, captured under varying environmental and lighting conditions. This diversity makes the dataset suitable for training and evaluating deep learning segmentation models.

# Dataset Link:
🔗 https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies

# Methodology

The dataset is preprocessed and split into training, validation, and testing sets. Data augmentation techniques such as rotation, flipping, and scaling are applied to improve model generalisation. Each model is trained using the same hyperparameters and loss functions to ensure fair comparison. Performance is evaluated using pixel-level segmentation metrics, including Accuracy, Intersection-over-Union (IoU), Dice Coefficient, and F1-score, which are widely used in remote sensing segmentation studies (Zhao et al., 2021).

# Results Summary

Experimental results demonstrate that the Attention-Enhanced Residual U-Net achieves the best overall performance, with an F1-score close to 0.80, IoU around 0.67, Dice Coefficient near 0.80, and pixel accuracy above 0.88. The improved performance is attributed to the combination of residual learning, which enhances gradient flow, and attention mechanisms, which suppress background noise and emphasise water regions. While FCN and DeepLabV3-Lite provide competitive results, they show limitations in fine boundary delineation and mixed-pixel regions.

# Applications

The proposed deep learning framework can be applied to:

Flood and drought monitoring

Long-term surface water change analysis

Climate change impact studies

Sustainable water resource management

Environmental and hydrological decision support systems

# References

Chen, L. C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2018). DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. IEEE TPAMI.

Chen, Y., et al. (2020). Surface water extraction from satellite imagery: Challenges and advances. Remote Sensing.

Gorelick, N., et al. (2020). Google Earth Engine: Planetary-scale geospatial analysis. Remote Sensing of Environment.

Liu, Y., et al. (2022). Attention-based deep learning models for water body segmentation. IEEE Journal of Selected Topics in Applied Earth Observations.

Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. CVPR.

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. MICCAI.

Xu, H. (2006). Modification of NDWI to enhance open water features. International Journal of Remote Sensing.
