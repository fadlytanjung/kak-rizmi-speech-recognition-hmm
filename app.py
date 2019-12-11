from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from HMM import HMM
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

    label = request.form.get('label')

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
    host = int(os.environ.get("PORT", 8080))
    detail = requests.get(host+'/labels/' +label)
    if len(detail.json()['payload']) == 0:
        return jsonify({'code': 401, 'message': 'label not found'}), 401

    CATEGORY = ['Q_3', 'D2_4', 'D1_1', 'D1_2', 'D1_4', 'D2_2', 'D2_3',
                'D3_3', 'D4_3','Q_4', 'D3_2', 'Q_1', 'D4_1', 'D1_3', 
                'D2_1', 'D4_4', 'D4_2', 'D3_1', 'Q_2', 'D3_4']
    obj = HMM(CATEGORY=CATEGORY)
    predict_data = obj.predict(label, 'input/tempData/data.wav')

    if predict_data == label:
        prediction = True
    else:
        prediction = False

    data = {
        'label_expected':label,
        'detail': detail.json()['payload'],
        'status_prediction':prediction,
        'label_prediction':predict_data

    }
    return jsonify({'code': 200, 'payload': data }), 200


@app.route("/predict_model", methods=["GET", "POST"])
def predict_model():

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

    CATEGORY = ['idgham', 'iqlab']
    obj = HMM(CATEGORY=CATEGORY)
    predict = obj.single_test(CATEGORY, 'input/tempData/data.wav')

    return jsonify({'code': 200, 'payload': predict}), 200

@app.route("/train_model", methods=["POST"])
def train_model():
    if request.json['app_name'] != 'tajwida':
        return jsonify({'code': 401, 'message': 'Incorrect App name'}), 401
    CATEGORY = ['idgham', 'iqlab']
    obj = HMM(CATEGORY=CATEGORY)
    wavdict, labeldict = obj.gen_wavlist('data/training')
    testdict, testlabel = obj.gen_wavlist('data/testing')

    obj.train(wavdict=wavdict, labeldict=labeldict)
    result = obj.test(wavdict=testdict, labeldict=testlabel)
    
    x = 0
    for key,val in result[1].items():
        x+=int(val)
    callback = {
        'correct_total':x,
        'detail_correct':result[1],
        'incorrect_total':len(result[0]),
        'detail_incorrect':result[0],
        'accurate':result[2]
    }
    return jsonify({'code': 200, 'payload': callback}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port) #env for heroku
    # app.run(port=port, debug=True)
