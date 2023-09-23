# DeepL-CNN with FER Dataset

## Project Overview

Emotion recognition has gathered significant attention in recent years owing to its extensive applications in various fields such as human-computer interaction, robotics, advertising, healthcare, among others. The primary objective of emotion recognition is to decipher the emotional state of a person into one of the basic emotions, which include anger, happiness, surprise, sadness, fear, disgust, or neutral. These emotions are typically recognized based on distinct signals such as facial expressions, body posture, tone of voice, etc.

This study focuses exclusively on recognizing emotions from facial expressions using Convolutional Neural Networks (CNNs) implemented in TensorFlow. The project is trained and tested on the FER2013 dataset, consisting of 34k images, which is suitable for CPU training.

## Methodology

### Step 1: Selection of Deep Learning Framework
- We selected TensorFlow as our deep learning framework for performing experiments due to its extensive features and ease of use. [TensorFlow Tutorial](https://www.tensorflow.org/tutorials)

### Step 2: Dataset Selection and Preparation
- We chose the [FER2013 dataset](https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge) from Kaggle for our experiments.
- The dataset was read, parsed, and loaded using TensorFlow’s data API.
- Various transformations and augmentation techniques were applied to improve image representation.

### Step 3: Building the CNN Model
- Multiple architectures and configurations were experimented with.
- A Sequential model was used, incorporating batch normalization and ReLU activation functions after each convolutional layer.
- Cross entropy was selected as the loss function, and ADAM was used along with a 0.001 learning rate.

### Step 4: Training and Testing
- The model was trained and tested using the standard pipeline.
- The decisions made and the rationale behind them will be explained in the report.
- Weights & Biases framework was considered for managing models and hyper-parameters.

### Step 5: Model Evaluation
- The model's accuracy, precision, and recall were checked on the test data and reported using several measures.

### Step 6: Visualization and Real-life Testing
- The learned filters on different layers were visualized to understand the internal representation of the CNN model.
- The model was tested on three real-life videos captured with a webcam to assess its performance and generalization ability.


## Running the Code in Google Colaboratory

This guide provides instructions on how to run the code in the Google Colaboratory environment.

### Prerequisites

- A Google account to access Google Colab and Google Drive.

### Steps

1. **Access the Google Drive Folder**

    The code and related files are stored in a Google Drive folder which can be accessed at the following URL:

    ```plaintext
    https://drive.google.com/drive/folders/18ZFc_B3AOvVNZ1deMIQzOI2_vHZAriQb?usp=sharing
    ```

2. **Open Google Colab**

    Visit [Google Colab](https://colab.research.google.com/) and sign in with your Google account.

3. **Open the Notebook**

    In Google Colab, navigate to `File > Open Notebook`. Then, switch to the `Google Drive` tab in the dialog box that appears. Navigate to the drive folder shared above and open the notebook.

4. **Mount Google Drive**

    To access the necessary files, you need to mount your Google Drive in the notebook environment. Run the cell containing the following code:

    ```python
    from google.colab import drive
    drive.mount("/content/drive")
    ```

    Follow the link that appears, sign in to your Google account, and allow Google Drive File Stream to access your Google Account. Copy the generated code, return to your notebook, paste it into the text box, and press Enter.

5. **Set the Directory Paths**

    Adjust the main directory path according to your Google Drive folder structure. The current setup assumes the following structure:

    ```python
    main_directory_path = "./drive/MyDrive/Assignment_CV/2/Group_4/" 
    ```

    You may need to adjust this to match the location of the files in your Google Drive. For example:

    ```python
    main_directory_path = "./drive/MyDrive/Group_4/"
    ```

    The rest of the directory paths are set relative to the main directory path, pointing to the dataset, images, models, loss & accuracy data, prediction results, videos, and hidden layers directories:

    ```python
    dataset_path = main_directory_path + "Dataset/"
    images_path = main_directory_path + "Data/Images/"
    models_path = main_directory_path + "Data/Models/"
    loss_accuracy_path = main_directory_path + "Data/Loss_Accuracy/"
    prediction_results_path = main_directory_path + "Data/Prediction_Results/"
    videos_path = main_directory_path + "Data/Videos/"
    layers_path = main_directory_path + "Data/Layers_Hidden/"
    ```

6. **Run the Notebook**

    You should now be able to run all the cells in the notebook. Make sure to use the "Runtime > Run all" option in Google Colab to run all the cells sequentially.

That's it! You have successfully set up and run the code in Google Colaboratory. If you encounter any issues or errors, make sure to check the directory paths and confirm that the necessary files are present in the specified locations.

Feel free to reach out for any further assistance. Happy coding!
