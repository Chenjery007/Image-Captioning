##Image Captioning System using CNN + LSTM (Flickr8k)
This project implements an image captioning system that generates natural language descriptions for images using a CNN encoder and an LSTM decoder. The model is trained on the Flickr8k dataset with TensorFlow/Keras.

##Project Overview
The system takes an image as input, extracts visual features using a pre-trained CNN, and feeds them into an LSTM-based decoder to generate a sequence of words forming a meaningful caption.

##Key Highlights
Dataset Handling via API: Kaggle API was used to automate dataset downloading and preprocessing.
CNN Model: InceptionV3 (pre-trained on ImageNet) used as encoder.
LSTM Decoder: Generates captions using embedded image features and previous words.
Training Pipeline: Custom DataGenerator for memory-efficient caption training.
Inference: Accepts random/custom images and generates captions.
Model Persistence: Saved tokenizer, model (.keras), and image features as .pkl.

##Objective
Build an end-to-end image captioning model from scratch.
Automate dataset download via Kaggle API.
Handle image preprocessing and feature extraction.
Train and evaluate a CNN-LSTM pipeline.
Save artifacts and build a prediction interface.

##Directory Structure
image-captioning/
├── Image Captioning.ipynb         # Final notebook: preprocessing, training, saving
├── caption_model.keras            # Trained model (LSTM decoder)
├── image_features.pkl             # Precomputed image features from InceptionV3
├── tokenizer.pkl                  # Vocabulary tokenizer
├── flickr8k_data/                 # Image & text dataset
├── README.md                      # Project documentation
├── .gitignore                     # Ignored heavy files: model, tokenizer, features

 ##Dataset
Source: Flickr8k Dataset
Images: 8,000
Captions: 5 per image
Download via: Kaggle API

##Tools & Technologies
Language: Python
Deep Learning: TensorFlow, Keras
Data Tools: Pandas, Numpy, Matplotlib
Text Preprocessing: NLTK
Evaluation: BLEU score
IDE: Jupyter Notebook
Version Control: Git + GitHub

##Model Architecture
1. CNN Encoder: InceptionV3
Input: Image
Output: 2048-dimensional feature vector
Layer used: Last pooling layer
2. LSTM Decoder
Input: Embedded caption sequences + image features
Output: Next word prediction
Layers: Embedding → LSTM → Dense
3. Combined Model Flow
Image → CNN → Features → LSTM → Caption Output

##Problems Faced & Solutions

| Problem                          | Solution                                                                 |
|----------------------------------|--------------------------------------------------------------------------|
| Memory overload during training  | Used `DataGenerator` class to stream data in batches                     |
| Shape mismatches in model input  | Inspected tensor shapes, adjusted LSTM input configuration               |
| Dataset file structure mismatch  | Cleaned and standardized captions, aligned filenames with images         |
| Git large file warning           | Added `.gitignore` to exclude `.keras`, `.pkl`, and other large files    |
| Random image caption generation  | Wrote a custom function using `random.choice()` and `encode_image()`     |


##Model Artifacts
File	Description
caption_model.keras	Final trained model for inference
tokenizer.pkl	Vocabulary used during training
image_features.pkl	Image embeddings (2048-D vectors)

 ##Author
Manoj Kumar Chenjery
GitHub: Chenjery007

##License
This project is intended for academic and research purposes only.
