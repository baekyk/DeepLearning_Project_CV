from flask import Flask, flash, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np


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
            # 함수 변경
            # string = transfer.img_to_s(os.path.join(app.config[UPLOAD_FOLDER], upload_name))
            return render_template('pkm.html', output=type_pred) , delete()
        else:
            flash('Allowed image types are -> png, jpg')
            return redirect(request.url)
    else:
            return render_template('pkm.html')


@app.route('/')
@app.route('/auto', methods = ['GET','POST'])
def auto():
    pass

app.run(host='0.0.0.0', port=5001)