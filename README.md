# Deep-Learning-Based-Brain-Tumor-Classification-from-MRI-Scans

## Project Metadata
### Authors
- **Team:** Asrar Almogbil
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:**( IAU, KFUPM)

## Introduction
Cancer is the abnormal growth of cells in any part of the body. According to the World Health Organization, it is considered one of the main causes of death worldwide. One of the most common and aggressive types is brain tumors. 


Brain tumors are classified into gliomas, meningiomas, and pituitary tumors based on the size, shape, and location of the mass in the brain. The early detection of brain tumors plays an important role in treatment planning. As the manual detection and classification of brain tumors can be time-consuming and error-prone, the process needs to be automated.


Medical imaging via magnetic resonance imaging (MRI), which provides detailed information about parts of the human body, such as the internal structures and tissues, plays a crucial role in detecting and classifying brain tumors.


## Problem Statement
The accurate classification of brain tumors is critical for reducing diagnostic errors and treatment planning. A robust system with a solid baseline was achieved through pretraining fine-tuning EfficientNet model that  is used to classify brain tumors via MRI (Zulfiqar et al., 2023). Although the EfficientNet model has shown robust accuracy in classifying brain tumor types, it focuses on local feature extraction and does not capture the global relationships between tumors, which can lead to failures in the classification of complex cases. Applying transformer-based architecture would be computationally expensive and require a large data set, which might not be available, particularly in the medical imaging domain. Accordingly, in this paper, a transformer-enhanced EfficientNet model is proposed. The aim is to investigate the effects of extending EfficientNet with a transformer layer to combine (1) the system’s robust local feature extraction with (2) learning about important regions and global reasoning. 

## Application Area and Project Domain
The area of the proposed application is medical imaging to automate the classification of brain tumor types into gliomas, meningiomas, and pituitary tumors. The project domain is a deep learning framework to investigate the effects of extending EfficientNet with a transformer layer to capture rich feature maps.

## What is the paper trying to do, and what are you planning to do?
Zulfiqar et al. (2023) proposed a robust system that classified brain tumors into gliomas, meningiomas, and pituitary using transfer learning of pretrained fine-tuned EfficientNet models. The EfficientNet model was selected because it is considered lightweight and computationally inexpensive. Five EfficientNet models (i.e., EfficientNet B0–B4) were trained to classify tumors via MRI using the Figshare brain tumor dataset. To modify the architecture of the EfficientNet model, the pretrained ImageNet weights were loaded to the base model, then three top layers—Global Average Pooling (GAP), Dropout, and a fully connected layer—were added, as shown in Figure 1. To train the model, different hyperparameters were explicitly fine-tuned. 
 <img width="940" height="192" alt="image" src="https://github.com/user-attachments/assets/e9026d35-0c16-4d9c-9263-4d274cebc5c8" />

                                           Figure 1(Zulfiqar et al., 2023)


Figure 2 shows the pipeline of the proposed methodology, in which several preprocessing steps were applied to the data in the training set. Then, transfer learning of pretrained fine tuning EfficentNet models has been applied.
<img width="877" height="413" alt="image" src="https://github.com/user-attachments/assets/dfccc541-a497-476d-b9a4-7962bb32af40" />

 
                                           Figure 2 (Zulfiqar et al., 2023)

The performance of these models (i.e., EfficientNet B0–B4)  was tested nder the same set experiments. The EfficientNet B2 model outperformed the other models, as it achieved 98.70% and 91.35 %accuracy on the Figshare brain tumor and Kaggle MRI datasets, respectively. The researchers applied Grad-CAM to the results obtained from EfficientNet B2 for explainability. 

The EfficientNet model showed robust accuracy in classifying brain tumors. However, convolution neural network (CNN) -based models focus on local features, which can result in the failure to classify complex and confusing scenarios. Accordingly, by combining the power of CNN-based architecture to capture local features and transformer-based architectures to capture global features, I propose extending EfficientNet with transform layers and to investigate the effects. In addition, a variation of cross-entropy loss function, which is label smoothing loss, will be tested. 








Zulfiqar, F., Bajwa, U.I. and Mehmood, Y., 2023. Multi-class classification of brain tumor types from MR images using EfficientNets. Biomedical Signal Processing and Control, 84, p.104777.



# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](/Project Presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

### Reference Dataset
- [LAION-5B Dataset](https://laion.ai/blog/laion-5b/)


## Project Technicalities

### Terminologies
-**EfficientNet**: A CNN architecture optimized for parameter efficiency and high performance, used as the backbone for feature extraction.  
-**Transformer Encoder**: An attention-based module that captures global dependencies across the image by learning relationships between tokens.  
-**Positional Embedding**: A method for encoding spatial location information, ensuring that the transformer is aware of each token’s position in the original image.  
-**Patch Tokenization**: reshape feature maps into sequences (tokens), enabling sequential processing by the transformer.  
-**1×1 Convolution**: a reducing dimensionality layer that compresses deep feature channels into a smaller embedding size before feeding into the transformer.  
-**Label Smoothing**: A regularization technique that softens target labels to prevent overconfidence and reduce overfitting.  
-**CBAM (Convolutional Block Attention Module)**: An attention block that enhances feature discrimination by applying channel attention followed by spatial attention.  
-**Thresholding**: An image processing method that transforms  a grayscale image into a black and white  image by setting pixel values above a chosen threshold to white and below it to black.  
-**Erosion**: A morphological operation that shrinks white regions in a binary image to remove noise and small unwanted artifacts.  
-**Dilation**: A morphological operation that expands or thickens white regions in a binary image, useful for closing gaps and strengthening detected regions.  
-**Macro F1 Score**: A performance metric that computes the F1-score for each class and averages them, giving equal importance to all classes regardless of frequency.

### Problem Statements
- **Problem 1:** CNN architectures have a limited ability to capture global contextual information within MRI scans. CNN architectures focus solely on local features and may overlook long-range spatial relationships within the brain.
- **Problem 2:** Difficulty in distinguishing tumors with overlapping visual characteristics. For example, Glioma and meningioma often share similar intensity patterns and shapes on MRI, leading to confusion for models.
- **Problem 3:** Uncertainty regarding the effect of attention and regularization techniques on model performance. Techniques such as label smoothing and attention modules (e.g., CBAM) can either improve or degrade performance, and their specific impact in the context of transformer-enhanced EfficientNet architectures remains unclear.

### Loopholes or Research Areas
-Dataset Dependency: The model was trained and tested on a specific dataset of  MRI scans. Performance may vary if real clinical MRI data is used.  
-Limited Exploration of Model Variants: The EfficientNet model and the hybrid EfficientNet model with transformer have been tested. This leaves room for other studies to explore transformer types, varying numbers of transformer layers, alternative backbone models, and additional attention mechanisms.  
-The current system only classifies the tumor type but does not perform tumor segmentation or localization within the MRI.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. Introduce a transformer-based attention head after EfficientNet to enable global context learning and relational reasoning across spatial regions. To enhance the ability to differentiate similarly appearing tumor types, provide positional awareness through embeddings, and capture long-range dependencies.
2.  Reshape feature maps into token sequences and apply positional embeddings to allow the model to understand where features originate in the brain, improve  tumor boundary interpretation, and enhance classification for ambiguous cases.
3.  Systematically evaluate transformer vs. no-transformer, label smoothing vs. no label smoothing, and CBAM vs. no CBAM to provide evidence-based architectural design, avoid unnecessary components, and select optimal model configuration. 

### Proposed Solution: Code-Based Implementation
-**Hybrid CNN–Transformer Architecture**: EfficientNet is used as the base feature extractor, followed by a 1×1 convolution, a reshape layer, and a transformer encoder with positional embeddings to capture global contextual information from MRI images.  
-**Standard Categorical Cross-Entropy Loss**: Used as the main loss function for the classification task, with additional experiments including label smoothing to study its regularization effect.  
-**Optimized Training Procedure**: The model is trained with the Adam optimizer, an appropriate batch size, and a learning rate scheduler to ensure steady learning. Preprocessing has been applied, such as cropping, noise removal, and shuffling. Training data is augmented to have an adequate number of samples.   
### Key Components
- **`Split_folders.ipynb`**: Handles dataset separation into training and testing.  
- **`Crop_Brain_Contours.ipynb`**:Performs brain region extraction.  
- **`Data_Augmentation.ipynb`**:Applies augmentation techniquesto increase dataset size and improve generalization.  
- **`Enhancedhybridmodel.ipynb`**:Implements the proposed EfficientNet + Transformer hybrid architecture for tumor classification.  
- **`ExpermentLabelSmthing.ipynb`**:Tests the effect of applying label smoothing to reduce overconfidence and study its impact on performance metrics.  
- **`ExpermentCBAMLayer.ipynb`**:Incorporates the CBAM attention module to examine whether it improves classification accuracy.  


## Model Workflow
The workflow of the proposed EfficientNet–Transformer classification model is designed as follows:

1. **Input:**
      Brain MRI images are passed to the model. These images have been resized to match the model's required size. Preprocessing has been applied, such as cropping, noise removal, and shuffling. Training data is augmented to have an adequate number of samples.   

2. **Feature extraction :**
     MRI images are  fed to the EfficientNet model to extract low and high-level local features. This results in a feature map of the MRI.   

3. **Transformer encoder satges:**
    -**1×1 Conv with 256 filters:** reduce the number of channels.  
    -**reshape layer:** Feature maps are reshaped into a sequence of tokens representing spatial patches of the MRI.  
    -**Positional Embedding:** Each token is assigned positional information so the transformer knows the location of each patch in the input.  
    -**Attention Processing:** The transformer encoder analyzes relationships between distant regions in the image, enabling global pattern recognition and contextual tumor understanding.  

4. **Classification Output:**
    -**Global Average Pooling:** to reduce the dimensionality.  
    -**Softmax Output:** Produces probabilities for the three classes, which are: Glioma, Meningioma, and Pituitary.  
    -**Final Prediction:** The model outputs the tumor class with the highest probability.  
## How to Run the Code
1.	Crop brain contour
2.	Use https://github.com/jfilter/split-folders to split dataset into train and test of 80:20
3.	Apply Data augmentation on dataset in train set only
4.	Train models and evaluate.

## Acknowledgments
-Special thanks to Dr. Dr. Muzammil Behzad for its continuous guidance, insightful discussions, and encouragement throughout the development of this work.  
-The MRI dataset used in this research was  available on Kaggle, a platform whose contributions to open scientific data are invaluable.  
-I acknowledge the developers and contributors of TensorFlow, Keras, NumPy, Matplotlib, and other open-source tools that enabled the implementation and testing of the model.  

