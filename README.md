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

### Challenges Faced
- Using a **6-layer neural network** resulted in overfitting, with zero training loss but high test loss.
- **SIFT feature extraction** had variable-length descriptors, requiring transformation into a fixed-length 128-dimensional vector by averaging descriptors.

### Results
- **HOG feature extraction** improved traditional methods' accuracy.
- **SVM outperformed other traditional models**, but CNN still performed best.
- **A 5-layer neural network with SIFT showed lower performance than a 3-layer NN with HOG.**

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
