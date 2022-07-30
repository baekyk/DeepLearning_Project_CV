from flask import Flask, flash, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os


UPLOAD_FOLDER = './static/'
ALLOWED_EXTENSIONS = {'png','jpg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
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
            upload.save(os.path.join(app.config[UPLOAD_FOLDER], upload_name))
            #print('upload_image filename: ' + filename)

            # 함수 변경
            # string = transfer.img_to_s(os.path.join(app.config[UPLOAD_FOLDER], upload_name))
            return render_template('pkm.html', output='string')
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