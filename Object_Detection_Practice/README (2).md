# YOLOv8 Object Detection Project


This project utilizes the YOLOv8 model for object detection. The provided Jupyter Notebook (yolov8 (1).ipynb) guides you through the process of training a YOLOv8 model on a custom dataset and evaluating its performance.

## Description

YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system. The yolov8 (1).ipynb notebook in this project demonstrates:

Setting up the environment for YOLOv8
Training the YOLOv8 model on a custom dataset
Evaluating the trained model's performance
Making predictions on new images

## Running the Notebook

1.Set up the dataset: Ensure your dataset is structured correctly and update the dataset path in the notebook.

2.Train the model: Execute the cells for training the YOLOv8 model. Adjust parameters like epochs, batch size, and image size as needed.
3.Evaluate the model: Run the evaluation cells to see the model's performance on the validation set.

4.Make predictions: Use the trained model to make predictions on new images.

## Installation
### Prerequisites
Ensure you have the following installed:

Python 3.8 or later
Jupyter Notebook or Jupyter Lab
Setting Up the Environment

1.Clone the repository (if this is part of a larger repository) or download the notebook.

2.Install required libraries. It's recommended to use a virtual environment
```bash
python -m venv yolov8-env
source yolov8-env/bin/activate  # On Windows, use `yolov8-env\Scripts\activate`
pip install -r requirements.txt
```
Note: If you don't have a requirements.txt file, here are some common dependencies you might need:

```bash
pip install torch torchvision numpy pandas matplotlib opencv-python ultralytics
```
3.Launch Jupyter Notebook:

```bash
jupyter notebook
```

## Usage

Open the yolov8 (1).ipynb notebook in Jupyter.

Follow the steps in the notebook to:

- Set up the dataset paths

- Train the YOLOv8 model

- Evaluate the model

- Make predictions on new images


## Project Structure

```bash
 /path/to/project
│
├── datasets
│   └── Detect-Traffic-Sign-6
│       ├── images
│       │   ├── train
│       │   └── val
│       └── labels
│           ├── train
│           └── val
│
├── yolov8 (1).ipynb
└── README.md
```

### Dataset
The dataset should be structured as follows:

```bash
/datasets/Detect-Traffic-Sign-6
    ├── images
    │   ├── train
    │   └── val
    └── labels
        ├── train
        └── val
```
#### data.yaml Example
Ensure you have a data.yaml file that specifies the dataset configuration. Here is an example:

```bash
train: /content/datasets/Detect-Traffic-Sign-6/images/train
val: /content/datasets/Detect-Traffic-Sign-6/images/val

nc: 1
names: ['traffic_sign']
```
#### Training
To train the model, ensure the training paths are correctly set in the notebook, then run the training cells. Example command:

```bash
!yolo train data=/content/datasets/Detect-Traffic-Sign-6/data.yaml model=yolov8s.yaml epochs=10 batch=16 imgsz=600
```
#### Evaluation
After training, run the evaluation cells in the notebook to assess model performance.

#### Conclusion
By following this guide and running the provided notebook, you should be able to train and evaluate a YOLOv8 model on your custom dataset. Adjust parameters and experiment with different settings to achieve the best results.
