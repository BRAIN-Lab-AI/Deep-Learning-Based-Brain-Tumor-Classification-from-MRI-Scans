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
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

### Reference Dataset
- [LAION-5B Dataset](https://laion.ai/blog/laion-5b/)


## Project Technicalities

### Terminologies
- **Diffusion Model:** A generative model that progressively transforms random noise into coherent data.
- **Latent Space:** A compressed, abstract representation of data where complex features are captured.
- **UNet Architecture:** A neural network with an encoder-decoder structure featuring skip connections for better feature preservation.
- **Text Encoder:** A model that converts text into numerical embeddings for downstream tasks.
- **Perceptual Loss:** A loss function that measures high-level differences between images, emphasizing perceptual similarity.
- **Tokenization:** The process of breaking down text into smaller units (tokens) for processing.
- **Noise Vector:** A randomly generated vector used to initialize the diffusion process in generative models.
- **Decoder:** A network component that transforms latent representations back into image space.
- **Iterative Refinement:** The process of gradually improving the quality of generated data through multiple steps.
- **Conditional Generation:** The process where outputs are generated based on auxiliary inputs, such as textual descriptions.

### Problem Statements
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
