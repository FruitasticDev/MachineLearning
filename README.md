# Welcome to the Machine Learning part of Fruitastic

Fruitastic is an Android-based scanning application designed to help users quickly and accurately assess fruit freshness. Utilizing image processing technology and machine learning algorithms, this application allows users to scan their desired fruits, detecting visual indicators of freshness, such as color, texture, and blemishes. The model will predict wether the fruit is ripe, mild, and rotten. This model only trained with 8 type of fruit: Avocado, Banana, Cucumber, Grapefruit, Kaki, Papaya, Peach, Tomato


## Dataset
The fruit dataset we use is downloaded from here [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7224690.svg)](https://doi.org/10.5281/zenodo.7224690)


## Cleaned dataset
We already cleaned the dataset, and can be downloaded from [this drive](https://drive.google.com/file/d/1-09sgSnfWQk6C4-m5nxVD7dRrT2IjnZQ/view?usp=sharing)


## Model
We achieve 97% accuracy with the fine tuned model
![image](https://github.com/user-attachments/assets/cd9d5dfe-b9d1-4b55-a748-28262c65bf9f)

test the model with the real banana taken from refrigerator
![image](https://github.com/user-attachments/assets/e7f3e83c-75ca-43e4-9637-172bfb9bc97b)


## Deployed model
the main model is located in Fine_tune folder 
[Link to model archive](https://drive.google.com/drive/folders/1QpfyFaeajMxR9JHjwu7br1nDOidR7QYy?usp=sharing)

Model class index are: ['AvocadoQ_Fresh', 'AvocadoQ_Mild', 'AvocadoQ_Rotten', 'BananaDB_Fresh', 'BananaDB_Mild', 'BananaDB_Rotten', 'CucumberQ_Fresh', 'CucumberQ_Mild', 'CucumberQ_Rotten', 'GrapefruitQ_Fresh', 'GrapefruitQ_Mild', 'GrapefruitQ_Rotten', 'KakiQ_Fresh', 'KakiQ_Mild', 'KakiQ_Rotten', 'PapayaQ_Fresh', 'PapayaQ_Mild', 'PapayaQ_Rotten', 'PeachQ_Fresh', 'PeachQ_Mild', 'PeachQ_Rotten', 'tomatoQ_Fresh', 'tomatoQ_Mild', 'tomatoQ_Rotten']


## Installation
The model developed using Python 3.10.12, and the libraries used are:

- NumPy version: 1.26.4
- Matplotlib version: 3.8.0
- TensorFlow version: 2.17.1
- Pillow (PIL) version: 11.0.0
- Seaborn version: 0.13.2
- Scikit-learn version: 1.5.2

## Usage

### Directly use the model
To predict the fruit, just run this code:
```
import tensorflow as tf


def load_and_preprocess_image(image_path, img_size=(224, 224)):

    """

    Load and preprocess an image for model prediction.

    """

    img = tf.keras.preprocessing.image.load_img(

        image_path,

        target_size=img_size

    )

    img_array = tf.keras.preprocessing.image.img_to_array(img)

    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)  # Use ResNet preprocessing

    return img_array



def predict_image(image_path, model, classes):

    """

    Predict the fruit type and its quality based on the given image.



    Args:

        image_path (str): Path to the input image.

        model (tf.keras.Model): Trained model for fruit classification.

        classes (list): List of fruit class names.

    Returns:

        dict: Prediction results including fruit type, quality, and confidence scores.

    """

    # Preprocess image

    img = load_and_preprocess_image(image_path)



    # Predict using the models

    pred = model.predict(img)



    # Get class predictions and confidence scores

    index = np.argmax(pred)



    return {

        'fruit': classes[index],

        'confidence': float(np.max(pred)),

    }

model = tf.keras.models.load_model('your_model')
classes_fruit = ['AvocadoQ_Fresh', 'AvocadoQ_Mild', 'AvocadoQ_Rotten', 'BananaDB_Fresh', 'BananaDB_Mild', 'BananaDB_Rotten', 'CucumberQ_Fresh', 'CucumberQ_Mild', 'CucumberQ_Rotten', 'GrapefruitQ_Fresh', 'GrapefruitQ_Mild', 'GrapefruitQ_Rotten', 'KakiQ_Fresh', 'KakiQ_Mild', 'KakiQ_Rotten', 'PapayaQ_Fresh', 'PapayaQ_Mild', 'PapayaQ_Rotten', 'PeachQ_Fresh', 'PeachQ_Mild', 'PeachQ_Rotten', 'tomatoQ_Fresh', 'tomatoQ_Mild', 'tomatoQ_Rotten']


prediction = predict_image(

    image_path='/content/drive/MyDrive/Kepston_bangkit/banana_kulkas_gwej_1.jpg',  # specify ur own path

    model=model, # specify ur model

    classes=classes_fruit

)



print(prediction)

```
### Using the notebook
If you want to run the notebook, use the Final_model.ipynb and make sure the dataset is loaded properly
