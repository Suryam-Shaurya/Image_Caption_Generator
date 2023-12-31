# Image Caption Generator
## About Project
This project uses a Deep Learning model to generate two captions for each uploaded image using two independently trained models having the same architecture on two different datasets.
## Datasets
[Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset "Flickr30k")<br>
[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k "Flickr8k")
## Architecture/Technology
* Numpy
* Pandas
* Tensorflow
* Keras
* VGG16
* LSTM
* Streamlit

# Examples

[![model_working](https://github.com/Suryam-Shaurya/Image_Caption_Generator/blob/model_1/Snapshots/working.gif?raw=true "model_working")](https://github.com/Suryam-Shaurya/Image_Caption_Generator/blob/model_1/Snapshots/working.gif?raw=true "model_working")
<figure>
<img src="https://github.com/Suryam-Shaurya/Image_Caption_Generator/blob/model_1/Snapshots/pred_1.png?raw=true" alt="Model_Flickr8k" >
</figure>
<figure>
<img src="https://github.com/Suryam-Shaurya/Image_Caption_Generator/blob/model_1/Snapshots/pred_2.png?raw=true" alt="Model_Flickr8k" >
</figure>

## Model
* This project showcases a deep-learning model designed to generate captions for uploaded images. The model employs a combination of the VGG16 CNN architecture and LSTM (RNN) to accomplish this task.

* The VGG16 CNN model is utilized to extract relevant features from the input image.

<figure>
<figcaption align = "center"><b>VGG16</b></figcaption>
<img src="https://github.com/kennethleungty/Neural-Network-Architecture-Diagrams/blob/main/vgg16_image.png?raw=true" alt="Logo" >
</figure>

* Simultaneously, a parallel branch of LSTM takes advantage of its sequential nature to generate context-aware and linguistically accurate captions.

<figure>
<figcaption align = "center"><b>LSTM</b></figcaption>
<img src="https://github.com/Suryam-Shaurya/Image_Caption_Generator/blob/model_1/img_gif/LSTM.png?raw=true" alt="Logo" >
</figure>

* By merging the image features extracted by the VGG16 CNN model and the LSTM-generated captions, the model produces descriptive and meaningful captions. But since the data is minimal, sometimes it needs to be corrected.

<figure>
<figcaption align = "center"><b>Model_Flickr30k</b></figcaption>
<img src="https://github.com/Suryam-Shaurya/Image_Caption_Generator/blob/model_1/img_gif/final_model_2030.png?raw=true" alt="Model_Flickr30k" >
</figure>
<figure>
<figcaption align = "center"><b>Model_Flickr8k</b></figcaption>
<img src="https://github.com/Suryam-Shaurya/Image_Caption_Generator/blob/model_1/img_gif/best_model.png?raw=true" alt="Model_Flickr8k" >
</figure>

## Results

### With Flickr30k

After training for **30 epochs** on the Flickr30k dataset, which consists of 31,783 images with 5 captions each, using 90% of the data for training, the model achieved a **categorical cross-entropy loss of 2.3299.** Additionally, the model obtained a **BLEU score of 0.5924** for the first caption of each image.

### With Flickr8k

The model underwent **20 epochs** of training on the Flickr8k dataset, which comprised 8,091 images, each accompanied by 5 captions. 90% of the dataset was utilized for training purposes. Notably, the model attained a **categorical cross-entropy loss of 1.9289.** Furthermore, the model demonstrated a commendable **BLEU score of 0.5876** for the initial caption of each image.

## Run Command

Clone this repository, open `app.py` and run `streamlit run app.py` to use the model locally.

## References

* https://medium.com/analytics-vidhya/cnn-lstm-architecture-and-image-captioning-2351fc18e8d7
* https://www.tensorflow.org/api_docs/python/tf
* https://docs.streamlit.io/
