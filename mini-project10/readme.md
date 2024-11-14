# CS5330_F24_Group8_Mini_Project_10

## project members

Anning Tian, Pingyi Xu, Qinhao Zhang, Xinmeng Wu

## setup instructions

Download the zip file or use GitHub Desktop to clone the file folder     
Download the model and training dataset

### Model/Dataset Download Link:
https://drive.google.com/drive/folders/1sIOYjd31C8-2HazLxVR1i6RRx8K1Mt1_?usp=sharing

---
## Project Overview

This project implements a segmentation model for identifying horse masks in images using an Attention U-Net architecture. The code covers data preparation, model definition, training, and evaluation.

## Usage Guidance

- **Training** (`train.py`): This script is used to train the model. To run it:
  - Update `image_path` and `mask_path` variables to point to the location of your images and masks.
  - Modify `epochs` and `batch size` as needed based on your training requirements.
  - Adjust the `learning rate`, and specify the desired model save path and filename.
  - Once configured, you can start training by running `python train.py` or `python3 train.py`.

- **Testing** (`test.py`): This script allows testing the model on any image to evaluate its segmentation performance.
  - Update the test image path as well as the model path to specify the desired model for testing.
  - After making these changes, you can test the model by running `python test.py` or `python3 test.py`.

---

### Dataset Details
- **Source**: The dataset used in this project is publicly available online.
- **Link**: [Weizmann Horse Dataset](https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database)
- **Structure**: The dataset is structured in folders as follows:
  - `horse/`: Contains the images of horses in `.png` format.
  - `mask/`: Contains the binary masks for each horse image in `.png` format.
- **Total Images**: 327 horse images.
- **Total Masks**: 327 corresponding mask images.

---

### Data Loading and Preparation

- **Image and Mask Paths**:  
  The `image_path` and `mask_path` variables define the file paths for loading input images and corresponding masks.

- **Preview Function** (`preview_image_and_mask`):
  - This function randomly selects an image and its corresponding mask, resizes them to the target size (256x256 by default), and displays them side-by-side.
  - The image is loaded in RGB, normalized (scaled to [0, 1]), and converted to grayscale to add an additional channel.
  - The mask is converted to grayscale and binarized to distinguish horse and background pixels.
  
- **Data Loading Function** (`load_and_prepare_data`):
  - Loads and processes all images and masks, storing them in lists and converting them to arrays for use in model training.
  - Each image undergoes resizing, normalization, and grayscale conversion, stacking RGB and grayscale as channels.
  - Masks are loaded, resized, normalized, and binarized.

---

### Model Architecture: Attention U-Net

- **Overview**:
  - The model is based on an Attention U-Net architecture, an extension of U-Net with attention gates to enhance feature localization by focusing on salient regions of the input.
  - The model expects an input shape of `(256, 256, 4)`, where the four channels represent RGB and grayscale stacked.

- **Layers and Activation Functions**:
  - **Convolutional Layers** (`conv_block`):  
    Each convolutional block has two convolutional layers with 3x3 filters and ReLU activation, used throughout the encoder and decoder to learn spatial features.
  
  - **Attention Mechanism** (`attention_block`):
    - Attention blocks refine feature maps by modulating skip connections between the encoder and decoder.
    - In each attention block:
      - Two 1x1 convolutions are applied to the skip and gating inputs.
      - These results are summed, passed through a ReLU, and then a sigmoid activation to generate attention coefficients.
      - The attention coefficients are element-wise multiplied with the skip input to retain only relevant features.

    The attention mechanism can be expressed as:

    $\text{Attention}(x, g) = x \times \sigma(\text{ReLU}(\theta(x) + \phi(g)))$

    where $\theta$ and $\phi$ are 1x1 convolution layers applied to the skip connection and gating signal, respectively.

- **Model Structure**:
  - **Encoder**:
    - A sequence of convolutional blocks with max-pooling layers between each block.
    - The number of filters doubles with each pooling operation, starting from 32 to 512 filters.
  
  - **Decoder**:
    - A sequence of up-sampling operations to restore spatial resolution, each followed by attention-modulated skip connections from corresponding encoder layers.
    - Each up-sampling layer is followed by two convolutional layers with ReLU activation.
  
  - **Output Layer**:
    - A final convolutional layer with a 1x1 filter and sigmoid activation produces the binary mask for segmentation.

---

### Training and Callbacks

- **Batch Size**: 16
- **Epochs**: 100

- **Optimizer**:
  - The model uses the **Adam optimizer** with a learning rate of `0.001`.

- **Loss Function**:
  - **Combined Dice and Binary Cross-Entropy (BCE) Loss** (`combined_dice_bce_loss`):
    - The Dice loss measures the overlap between predicted and true masks, encouraging smooth and accurate segmentations. Defined as `1 - dice_coef`, it balances precision and recall.
    - The BCE loss penalizes pixel-wise errors, helping the model learn detailed segmentation boundaries.
    - The combined loss is the sum of Dice and BCE losses, promoting both boundary precision and overall segmentation accuracy.
    
    **Dice Coefficient**:

    $\text{Dice Coefficient} = \frac{2 \cdot |y_{\text{true}} \cap y_{\text{pred}}| + \text{smooth}}{|y_{\text{true}}| + |y_{\text{pred}}| + \text{smooth}}$,

    where $y_{\text{true}}$ and $y_{\text{pred}}$ are the flattened ground truth and predicted masks, respectively.

    **Binary Cross-Entropy (BCE) Loss**:

    $\text{BCE Loss} = - \frac{1}{N} \sum_{i=1}^{N} \left( y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right)$,

    where $y_i$ is the true label and $p_i$ is the predicted probability.

    **Combined Loss**:

    $\text{Combined Loss} = \text{Dice Loss} + \text{BCE Loss}$,

- **Dynamic Learning Rate Adjustment** (`ReduceLROnPlateau`):
  - The learning rate is reduced by half if the validation loss plateaus for three epochs, with a minimum learning rate threshold of $1 \times 10^{-6}$.

- **Metrics**:
  - **Accuracy**: Measures pixel-wise accuracy of segmentation.
  - **Dice Coefficient**: Evaluates overlap between predicted and true masks, providing a primary gauge of segmentation quality.
  - **Mean IoU (Intersection over Union)**: Calculates the average overlap between predicted and true masks, averaged across two classes (background and horse).

- **Callbacks for Model Saving**:
  - **ModelCheckpoint**:
    - Two models are saved during training:
      - `unet_horse_best_v2.h5`: Saves the model with the lowest validation loss.
      - `unet_horse_final_v2.h5`: Saves the last model state at the end of training.

- **Early Stopping**:
  - Stops training if validation loss does not improve for 30 epochs, reverting to the best model weights.

---

### Evaluation and Prediction Display

- **IoU Calculation** (`calculate_iou`):
  - Intersection over Union (IoU) is calculated by comparing the overlap of predicted and true mask pixels. IoU is a key metric for segmentation performance, with values closer to 1 indicating better overlap.

  The formula for IoU is:

  $\text{IoU} = \frac{|y_{\text{true}} \cap y_{\text{pred}}|}{|y_{\text{true}} \cup y_{\text{pred}}|}$

- **Prediction Display** (`display_predictions`):
  - This function randomly selects three test samples, generates predictions, and displays the results.
  - For each sample:
    - Original test image, ground truth mask, and predicted mask are displayed side-by-side.
    - Each predicted mask’s IoU score is calculated and displayed.

--- 

### Challenges and Potential Improvements

**1. Challenge**: We implemented data augmentation to improve model generalization, but it did not significantly enhance performance in this regard. For instance, when testing with horses in complex poses, the augmented data failed to improve the model's ability to generalize to these variations. 

**Potential Improvement**: Experimenting with more diverse or targeted augmentation techniques, such as pose-specific transformations or synthetic data generation, could better capture complex variations and improve generalization.

**2. Challenge:** We used dynamic learning rate adjustment (ReduceLROnPlateau) and early stopping to optimize the training process. However, these methods conflicted, as early stopping halted training when the model reached convergence, while dynamic learning rate adjustments aimed to continue fine-tuning. Ultimately, we prioritized dynamic learning rate adjustment.
    
**3. Potential Improvement**: Consider tuning early stopping parameters to trigger after a longer period of convergence, allowing the learning rate adjustments to take effect more fully without prematurely ending training. Alternatively, adaptive stopping criteria that account for learning rate changes could be explored.
