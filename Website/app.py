from cv2 import batchDistance
from flask import Flask, flash, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import model_autoencoder





UPLOAD_FOLDER = './pred/a/'
ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'secret poketmon'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
@app.route('/pktype', methods = ['GET','POST'])
def pktype():
    if request.method == 'POST':
        upload = request.files['file']
        if upload.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if upload and allowed_file(upload.filename):
            upload_name = secure_filename(upload.filename)
            upload.save(os.path.join(app.config['UPLOAD_FOLDER'], upload_name))
            #print('upload_image filename: ' + filename)
            
            DATAGEN_TEST = ImageDataGenerator(
            rescale=1./255,
            featurewise_center=True,
            featurewise_std_normalization=True,
            data_format="channels_last")

            test_generator = DATAGEN_TEST.flow_from_directory(
            directory="./pred",
            batch_size=16,
            seed=42,
            shuffle=False,
            class_mode="categorical",
            target_size=(224,224)
            )
            model = tf.keras.models.load_model('../010-0.1035-0.9666-1.8962-0.5618.hdf5')

            type_pred = np.argmax(model.predict(test_generator))

            def types(type_pred):
                dic = {
                    0 : 'electric',
                    1 : 'psychic',
                    2 : 'rock',
                    3 : 'water',
                    4 : 'grass',
                    5 : 'normal',
                    6 : 'poison',
                    7 : 'fire'
                }
                return dic[type_pred]
            
            def delete():
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], upload_name))

            type_pred = types(type_pred)
            img = upload_name
            
            return render_template('pkm.html', output=type_pred) , delete()
        else:
            flash('Allowed image types are -> png, jpg, jpeg')
            return redirect(request.url)
    else:
            return render_template('pkm.html')


@app.route('/')
@app.route('/encode', methods = ['GET','POST'])
def autoencoder():
    if request.method == 'POST':
        origin = request.files['file']
        if origin.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if origin and allowed_file(origin.filename):
            origin_name = secure_filename(origin.filename)
            origin.save(os.path.join('./static/', origin_name))
            #print('upload_image filename: ' + filename)

            # image -> sketch
            img = cv2.imread(os.path.join('./static/', origin_name))
            gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            invert = cv2.bitwise_not(gray)
            blur = cv2.GaussianBlur(invert, (21, 21), 0)
            invertedblur = cv2.bitwise_not(blur)
            sketch = cv2.divide(gray, invertedblur, scale=256.0)
            sketch = cv2.cvtColor(sketch,cv2.COLOR_BGR2RGB)
            sketch = cv2.resize(sketch, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            # sketch = sketch.astype('float32')/255.
            # cv2.imwrite(os.path.join('./static/','sketch_'+origin_name),sketch)
            plt.imsave(os.path.join('./static/','sketch_'+origin_name),sketch)
            sketch_name = 'sketch_'+origin_name
            return render_template('pkm.html', output2=sketch_name)
        else:
            flash('Allowed image types are -> png, jpg, jpeg')
            return redirect(request.url)
    else:
            return render_template('pkm.html')


@app.route('/')
@app.route('/decode', methods = ['GET','POST'])
def decoder():
    if request.method == 'POST':
        origin = request.files['file']
        if origin.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if origin and allowed_file(origin.filename):
            origin_name = secure_filename(origin.filename)
            origin.save(os.path.join('./static/', origin_name))
            # origin = cv2.imread('./static/images_13.jpeg')
            # origin = cv2.resize(origin, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            # sketch = cv2.imread('./static/sketch_images_13.jpeg')
            # sketch = tf.expand_dims(sketch, axis=0)
            # sketch = np.array([sketch])
            # origin = np.array([origin])

            sketch = cv2.imread(os.path.join('./static/', origin_name))
            sketch = cv2.cvtColor(sketch,cv2.COLOR_BGR2RGB)
            sketch = sketch.astype('float32')/255.
            sketch = tf.expand_dims(sketch, axis=0)
            model = tf.keras.models.load_model('../auto_encoder.hdf5')
            result = model.predict(sketch)
            # autoencoder = model_autoencoder.model_auto()
            # history , autoencoder = model_autoencoder.model_fit(autoencoder, origin, sketch)
            # result = model_autoencoder.model_pred(autoencoder, sketch)
            result = result[0]
            plt.imsave(os.path.join('./static/', 'recover_'+origin_name),result)
            # cv2.imwrite(os.path.join('./static/', 'recover_'+origin_name),result)
            recover_name = 'recover_'+origin_name
            sketch_name = origin_name
            return render_template('pkm.html', output2 = sketch_name, output3=recover_name)
        else:
            flash('Allowed image types are -> png, jpg, jpeg')
            return redirect(request.url)
    else:
            return render_template('pkm.html')






app.run(host='0.0.0.0', port=5001)