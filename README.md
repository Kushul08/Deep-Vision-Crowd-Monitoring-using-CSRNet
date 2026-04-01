# Deep-Vision-Crowd-Monitoring-using-CSRNet
**DeepVision Crowd Monitor
Overview**

DeepVision Crowd Monitor is a real-time, AI-driven system for crowd density estimation and safety analysis. The project was developed as part of the Infosys Springboard Virtual Internship 6.0 (AI) from September to November 2025. It is designed to process multiple input streams and provide timely alerts to help mitigate overcrowding risks in public spaces.

**Key Features**

Multi-modal input support: image, video, and live webcam.
 Real-time crowd density estimation and visualization.
 Automated visual and audio alerts for overcrowding detection.
 Modular, extensible pipeline for easy integration and scaling.

**Model Architecture**

The system is built on a deep learning architecture optimized for crowd counting tasks:

CSRNet for density estimation.
 VGG-16 front-end for feature extraction.
 Dilated convolutional layers for high-resolution density map generation.

**Performance**

Achieved a 47.3% reduction in Mean Absolute Error (MAE),
Benchmarked against CP-CNN models,
Evaluated on the ShanghaiTech Part B dataset.

**Technology Stack**

Programming Language: Python

**Frameworks and Libraries:**

 PyTorch,
 OpenCV,
 NumPy,
 Streamlit,
YOLOv8.

**System Workflow**

Input data (image, video, or live stream) is captured,
 Frames are preprocessed and passed to the trained model,
 CSRNet generates density maps for crowd estimation,
 Threshold-based logic identifies overcrowding conditions,
 Real-time alerts (visual and audio) are triggered

**Future Work**

Deployment on scalable cloud infrastructure,
 Optimization for edge devices and low-resource environments,
 Integration of advanced analytics and monitoring dashboards.

**Contact**

If you find this project useful or interesting, please consider starring the repository and connecting for further collaboration.
