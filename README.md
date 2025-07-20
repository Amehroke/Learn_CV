# Learn_CV
This repository is dedicated to learning and implementing computer vision


Absolutely. Here’s a structured list of **algorithms**, **metrics**, and **architectures** you should know to gain **fundamental to advanced knowledge in Computer Vision**, covering both **classical methods** and **deep learning** techniques.

---

## ✅ **1. Classical Computer Vision Algorithms**

### 🧠 Image Processing & Feature Extraction

* **Convolution**
* **Sobel, Scharr, Prewitt** filters (edge detection)
* **Gaussian Blur**
* **Canny Edge Detector**
* **Histogram Equalization**
* **Thresholding (Global, Adaptive, Otsu’s)**
* **Morphological Operations**: Erosion, Dilation, Opening, Closing
* **Hough Transform** (line and circle detection)
* **Fourier Transform** (frequency domain filtering)

### 🔍 Feature Detection & Matching

* **Harris Corner Detector**
* **Shi-Tomasi Corner Detector**
* **SIFT** (Scale-Invariant Feature Transform)
* **SURF** (Speeded Up Robust Features)
* **ORB** (Oriented FAST and Rotated BRIEF)
* **FAST** (Features from Accelerated Segment Test)
* **BRIEF** (Binary Robust Independent Elementary Features)
* **AKAZE, KAZE**

### 🧭 Object Tracking / Motion

* **Optical Flow (Lucas-Kanade, Farneback)**
* **Kalman Filter** (object tracking)
* **Mean-shift & Camshift Tracking**
* **Background Subtraction (MOG, KNN)**

### 🧊 Segmentation (Classical)

* **Watershed Algorithm**
* **Graph Cut**
* **GrabCut**
* **K-means on pixels**
* **Region Growing**

---

## ✅ **2. Deep Learning Architectures for Computer Vision**

### 🧠 Core CNN Architectures

* **LeNet-5** – First successful CNN (digits)
* **AlexNet** – Deep CNN on ImageNet
* **VGG16 / VGG19** – Deep, simple architecture with 3×3 convolutions
* **GoogLeNet (Inception)** – Multi-scale filters (1×1, 3×3, 5×5)
* **ResNet** – Introduces residual connections (ResNet18, ResNet50, ResNet101)
* **DenseNet** – Each layer connected to every other layer (dense connections)
* **MobileNet / EfficientNet / SqueezeNet** – Lightweight, mobile-friendly CNNs

### 🧱 Object Detection

* **R-CNN**
* **Fast R-CNN**
* **Faster R-CNN**
* **YOLO (v1 to v8)** – One-shot real-time object detection
* **SSD (Single Shot MultiBox Detector)**
* **RetinaNet** – Focal loss for class imbalance

### 🎨 Segmentation

* **FCN (Fully Convolutional Networks)**
* **U-Net** – Encoder-decoder with skip connections
* **SegNet**
* **DeepLab (v2-v3+)**
* **Mask R-CNN** – Instance segmentation

### 🔭 Vision Transformers (ViT)

* **Vision Transformer (ViT)**
* **Swin Transformer** – Hierarchical ViT
* **DETR (Detection Transformer)** – End-to-end object detection using transformers
* **Segmenter** – Transformer for segmentation

---

## ✅ **3. Key Performance Metrics**

### 📷 Image Classification

* **Accuracy**
* **Precision / Recall / F1-Score**
* **Confusion Matrix**
* **Top-1 / Top-5 Accuracy** (used in ImageNet)

### 🎯 Object Detection

* **IoU (Intersection over Union)**
* **mAP (mean Average Precision)**

  * `mAP@0.5` (IoU = 0.5 threshold)
  * `mAP@0.5:0.95` (average over IoU thresholds)
* **Precision-Recall Curve**
* **Focal Loss** (used in RetinaNet for class imbalance)

### 🖼️ Image Segmentation

* **Pixel Accuracy**
* **Mean IoU (mIoU)**
* **Dice Coefficient (F1 for pixels)**
* **Boundary IoU** (for fine detail)

### 📹 Super-Resolution / Image Quality

* **PSNR (Peak Signal-to-Noise Ratio)**
* **SSIM (Structural Similarity Index)**

---

## ✅ **4. Essential Libraries & Tools**

* **OpenCV** – Image processing & classic CV
* **scikit-image** – Image processing in Python
* **PyTorch / torchvision** – Deep learning
* **TensorFlow / Keras** – Deep learning
* **Detectron2** – Facebook's object detection library
* **MMDetection** – OpenMMLab’s detection framework
* **Albumentations** – Data augmentation
* **Hugging Face Transformers** – For ViT and DETR
* **LabelImg / CVAT / Roboflow** – Dataset labeling

---

Let me know if you want cheat sheets, visual summaries, or project ideas for each group!
