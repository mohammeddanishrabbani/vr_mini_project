#### VR_Project1- Anukriti Singh, Susmita Roy, Mohd. Danish Rabbani


## Face Mask Classification

### Overview
This project section compares the performance of different traditional classifiers, including neural networks and CNNs, to determine which method is most effective for face mask classification. The objective is to show that CNNs outperform traditional models.

### Dataset Structure
The dataset was initially stored in a folder, zipped for upload to Google Colab, and later unzipped for processing. It consists of two subfolders:

```
├─ with_mask      # Images of faces with masks
├─ without_mask   # Images of faces without masks
```
The images are in JPG and PNG formats.

### Methods Used
### Feature Extraction
1. **Plain Flattening**: Directly converts image data into a feature vector.
2. **Histogram of Oriented Gradients (HOG)**: Extracts gradient-based features, leading to better performance.
3. **Scale-Invariant Feature Transform (SIFT)**: Converts descriptors into a fixed-length 128-dimensional vector per image.

### Classification Models
1. **Random Forest (100 Decision Trees)**: Handles high-dimensional data effectively.
2. **Support Vector Machine (SVM)**: Shows strong performance for this task.
3. **Neural Network (3-Layer, 1 Hidden Layer)**: Outperforms traditional methods by learning hierarchical representations.
4. **Convolutional Neural Network (CNN)**: Achieves the best performance by leveraging spatial patterns in images.

         CNN Architecture
       
           ----------------------------------------------------------------
                    Layer (type)               Output Shape         Param #
            ================================================================
                        Conv2d-1         [-1, 32, 128, 128]             896
                     MaxPool2d-2           [-1, 32, 64, 64]               0
                        Conv2d-3           [-1, 64, 64, 64]          18,496
                     MaxPool2d-4           [-1, 64, 32, 32]               0
                        Conv2d-5          [-1, 128, 32, 32]          73,856
                     MaxPool2d-6          [-1, 128, 16, 16]               0
                        Linear-7                  [-1, 128]       4,194,432
                       Dropout-8                  [-1, 128]               0
                        Linear-9                    [-1, 1]             129
            ================================================================
            Total params: 4,287,809
            Trainable params: 4,287,809
            Non-trainable params: 0
            ----------------------------------------------------------------
            Input size (MB): 0.19
            Forward/backward pass size (MB): 8.75
            Params size (MB): 16.36
            Estimated Total Size (MB): 25.30
            ----------------------------------------------------------------
        
       

### Challenges Faced
- Using a **6-layer neural network** resulted in overfitting, with zero training loss but high test loss.
- **SIFT feature extraction** had variable-length descriptors, requiring transformation into a fixed-length 128-dimensional vector by averaging descriptors.


### Results
#### Results with Traditional Classification Techniques
![image](https://github.com/user-attachments/assets/3ad72204-e65d-4fde-909f-c1144aca243d)
![image](https://github.com/user-attachments/assets/c1935370-54ca-4e00-8fb2-e1358b3065fd)
![image](https://github.com/user-attachments/assets/5250eb97-6ebc-49fa-87cf-718c58172d14)
- **HOG feature extraction** improved traditional methods' accuracy.
- **SVM outperformed other traditional models**, but CNN still performed best.
- **A 5-layer neural network with SIFT showed lower performance than a 3-layer NN with HOG.**


#### Results with CNN for Classification
#### MODEL PARAMETERS
    Learning Rate: 0.01
    Batch Size: 32
    Optimizer: Adam
    Activation: Sigmoid

![image](https://github.com/user-attachments/assets/002e1ce6-630f-44fb-a3ef-00ceb7376e40)


#### MODEL PARAMETERS
    Learning Rate: 0.005
    Batch Size: 32
    Optimizer: Adam with Weight Decay 1e-4
    Activation: Sigmoid
![image](https://github.com/user-attachments/assets/547ed5ec-8e28-4f63-810f-cac1338c88d8)


#### MODEL PARAMETERS
    Learning Rate: 0.005
    Batch Size: 32
    Optimizer: RMSprop
    Activation: Sigmoid
    L2 Regularization: 1e-5
![image](https://github.com/user-attachments/assets/f1db84d4-6a1e-411f-8063-8635500cfcc6)


#### MODEL PARAMETERS
    Learning Rate: 0.001
    Batch Size: 32
    Optimizer: adam
    Activation: Sigmoid
![image](https://github.com/user-attachments/assets/1bbe7820-54f4-4d00-af5e-d0a8d6c87d02)





    
### Conclusion
- Traditional methods like SVM and Random Forest show varying performance, with SVM being superior.
- Neural networks outperform traditional models due to their hierarchical feature learning.
- **CNN achieves the highest accuracy**, proving its effectiveness over both traditional classifiers and simple neural networks.

### Usage
1. **Load dataset.zip** when prompted.
2. **Run the Python scripts in Google Colab** (Colab-specific implementation).
3. **Dataset is stored via Git LFS**, so cloning requires Git LFS installation.

```bash
# Clone repository with dataset
git lfs install
git clone <repository_url>
```

### References
- HOG Feature Extraction: [Dalal & Triggs, 2005](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
- CNN-based Classification: [LeCun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

## Face Mask Segmentation

### Overview
This project section explores face mask segmentation using both traditional image processing techniques and deep learning-based approaches. The goal is to compare different segmentation methods and evaluate their performance.

### Dataset Structure

The dataset consists of two main folders; however, only **Folder 1** was used for training and evaluation due to the lack of ground truth in **Folder 2**. The dataset was split into **80% training** and **20% testing**.

```
├─ /1
│　├─ face_crop              # Contains cropped face images
│　├─ face_crop_segmentation # Contains segmented face images
│　├─ img                    # Raw images
│　└─ dataset.csv            # CSV file containing dataset metadata
├─ /2 (Not Used)
│　└─ img                    # Unused image directory
```

### Methods Used
### Traditional Methods
1. **Region Growing Segmentation**: Expands regions based on pixel similarity to segment the mask area.
2. **Edge Contour Segmentation**: Detects mask boundaries using edge-based techniques.
3. **Thresholding Segmentation**: Segments the mask region based on intensity thresholds.

### Deep Learning Method
- **UNet**: A fully convolutional neural network designed for pixel-wise segmentation, trained to accurately segment face masks.

### Challenges Faced
- Traditional methods require fine-tuning of parameters for different lighting conditions.
- Region growing and edge detection methods may fail on complex backgrounds.
- UNet requires a large dataset and computational power for training.
- Data preprocessing and augmentation were necessary to improve generalization.

### Results
#### Results with traditional methods for segmentation
![image](https://github.com/user-attachments/assets/c82cad30-bd72-4ad6-b75e-a7dcf5aa1144)


#### Results with UNET for segmentation
![image](https://github.com/user-attachments/assets/99f08dc7-9b28-47cd-aad8-ccbe773c0a54)

#### Result Summary
| Method                  | Accuracy | Intersection over Union     | Dice Coefficient|
|-------------------------|--------- |---------------------------- |-----------------|
| Region Growing          | 0.6730   | 0.2755                      | 0.3652          |
| Edge Contour            | 0.6305   | 0.1563                      | 0.2552          |
| Thresholding Segment.   | 0.5291   | 0.2909                      | 0.4052          |
| UNet                    | 0.6119   | 0.8944                      | 0.9390          |

### Conclusion
- Traditional methods work well for simple cases but struggle with complex backgrounds.
- UNet outperforms traditional methods in terms of segmentation accuracy.
- Future improvements could include using more advanced architectures like DeepLabV3+ or Transformer-based segmentation models.

### Usage
1. Install dependencies:
    ```bash
    pip install torch torchvision numpy opencv-python matplotlib tqdm
    ```
2. Run the segmentation.ipynb notebook

### References
- UNet Paper: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- Image Processing Techniques: [OpenCV Documentation](https://docs.opencv.org/)
