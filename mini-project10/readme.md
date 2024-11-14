# Horse Mask Segmentation using Attention U-Net

This project implements a segmentation model for identifying horse masks in images using an Attention U-Net architecture. The code covers data preparation, model definition, training, and evaluation.

## Contents

1. [Dataset Details](#dataset-details)
2. [Data Loading and Preparation](#data-loading-and-preparation)
3. [Model Architecture: Attention U-Net](#model-architecture-attention-u-net)
4. [Training and Callbacks](#training-and-callbacks)
5. [Evaluation and Prediction Display](#evaluation-and-prediction-display)

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

    \[
    \text{Attention}(x, g) = x \times \sigma(\text{ReLU}(\theta(x) + \phi(g)))
    \]

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

    `$$
    \text{Dice Coefficient} = \frac{2 \cdot |y_{\text{true}} \cap y_{\text{pred}}| + \text{smooth}}{|y_{\text{true}}| + |y_{\text{pred}}| + \text{smooth}}
    $$`

    where $y_{\text{true}}$ and $y_{\text{pred}}$ are the flattened ground truth and predicted masks, respectively.

    **Binary Cross-Entropy (BCE) Loss**:

    `$$
    \text{BCE Loss} = - \frac{1}{N} \sum_{i=1}^{N} \left( y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right)
    $$`

    where $y_i$ is the true label and $p_i$ is the predicted probability.

    **Combined Loss**:

    `$$
    \text{Combined Loss} = \text{Dice Loss} + \text{BCE Loss}
    $$`

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

  `$$
  \text{IoU} = \frac{|y_{\text{true}} \cap y_{\text{pred}}|}{|y_{\text{true}} \cup y_{\text{pred}}|}
  $$`

- **Prediction Display** (`display_predictions`):
  - This function randomly selects three test samples, generates predictions, and displays the results.
  - For each sample:
    - Original test image, ground truth mask, and predicted mask are displayed side-by-side.
    - Each predicted mask’s IoU score is calculated and displayed.

--- 

This README provides a concise breakdown of the code structure, model architecture, training process, and evaluation methods used in this segmentation model. Let me know if further details are needed!
