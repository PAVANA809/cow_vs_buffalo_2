from flask import request,render_template
from flask import jsonify
from flask import Flask
import base64
from PIL import Image
import io
import re
import keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle


app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

app.config["IMAGE_UPLOADS"] = "static/images/uploads"

@app.route('/upload_image',methods=['POST','GET'])
def upload_image():
    print("before post")
    if request.method == "POST":
        print("after post")
        message = request.get_json(force=True)
        encoded = message['image']
        image_data = re.sub('^data:image/.+;base64,', '', encoded)
        decoded = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(decoded))
        print("image decoded")
        # web_img = request.files["file"]
        # image.save(os.path.join(app.config["IMAGE_UPLOADS"], Image.filename))
        # print("Image is saved")
        # path = "static/images/uploads/"+image.filename
        # img = image.load_img(path)
        processed_image = preprocess_image(image, target_size=(224, 224))
        prediction = predict(processed_image)
        print(prediction)
        response = {
                'cow': prediction[0][0],
                'buffalo': prediction[0][1]
            }
        print(response)
        return jsonify(response)
    return



def get_model():
    global model
    model = load_model('mobile_net_after_training_cowvsbuffalo.h5')
    # pickl_in = open("mobile_net_after_training_cowvsbuffalo.h5", "rb")
    # model = pickle.load(pickl_in)
    print(" * Model loaded!")

print("loading model")
get_model()

def preprocess_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict(img):
    global pre
    pre = model.predict(img).tolist()
    return pre

if __name__ == '__main__':
    app.run(debug=True)