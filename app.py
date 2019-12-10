from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
# from HMM import HMM
import time,json,re,os, requests

UPLOAD_FOLDER = 'input/tempData/'
ALLOWED_EXTENSIONS = set(['wav'])

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def main():
    return jsonify({'code': 200, 'message': 'Assalamualaykum, Welcome to Tajwida :)'}), 200

@app.route("/labels", methods=["GET"])
def label():
    with open('db/index.json') as f:
        data = json.load(f)
    return jsonify({'code': 200, 'payload': data['labels'] }), 200


@app.route("/labels/<id>", methods=["GET"])
def detail_label(id):
    with open('db/index.json') as f:
        data = json.load(f)
    
    for item in data['labels']:
        if str(item['id']) == id:
            datas = item
            break
        elif item['label'] == id:
            datas = item
            break
        else:
            datas = []
    return jsonify({'code': 200, 'payload': datas}), 200

@app.route("/predict", methods=["GET","POST"])
def predict():

    if 'audio' in request.files:
        filewav = request.files["audio"]
        
        if filewav and allowed_file(filewav.filename):
            filename = secure_filename(filewav.filename)
            # filewav.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filewav.save(os.path.join(
                app.config['UPLOAD_FOLDER'], 'data.wav'))
           
        else:
            return jsonify({'code': 401, 'message': 'Invalid Format File'}), 401

    else:
        return jsonify({'code': 401, 'message': 'File can not be null'}), 401

    detail = requests.get('http://localhost:8080/labels/' +
            request.form.get('label'))
    if len(detail.json()['payload']) == 0:
        return jsonify({'code': 401, 'message': 'label not found'}), 401
    data = {
        'label':request.form.get('label'),
        'detail': detail.json()['payload'],
        'status_prediction':False
    }
    return jsonify({'code': 200, 'payload': data }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
