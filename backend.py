import base64
from flask import Flask, flash, jsonify, render_template, request, redirect, url_for, send_file
from flask_cors import CORS
import os
from deep_translator import GoogleTranslator
from gtts import gTTS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import os
import nltk

# Load the fine-tuned model and processor
model_id = "Salesforce/blip-image-captioning-base"
model_path = 'caption'
finetuned_model = AutoModelForVision2Seq.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_id)

def generate_caption(file_path: str) -> str:
    raw_image = Image.open(file_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")

    out = finetuned_model.generate(**inputs)
    generated_caption = processor.decode(out[0], skip_special_tokens=True)

    return generated_caption

app = Flask(__name__)
CORS(app)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = 'static'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def translate(eng): 
    thai = GoogleTranslator(source='en', target='th').translate(eng)
    return thai
    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    speech_file_paths = []
    if 'file_0' not in request.files:
        print("Not found")
        flash('No file part')
        return jsonify({'error' : 'No file part'})

    # filesKey = request.files.keys()
    files = request.files.to_dict()
    lang = request.form.to_dict()['language']
    #lang = request.body

    # print(files['file_0'])
    allCaption = []

    for key, file in files.items():
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        print("Get DIr: ", dir(file))
        print(file.content_type)
        print(file.filename)
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return render_template('index.html', filename=filename)
            
            caption = generate_caption(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # cap_slice = caption.split()
            # cap_slice.remove("startseq")
            # cap_slice.remove("endseq")
            # caption = ' '.join(cap_slice)

            print(lang)

            image_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(caption)

            if(lang == "Thai"):
                caption = translate(caption)
                speech_file_path = os.path.splitext(image_file_path)[0] + "_speech_th.mp3"
                tts = gTTS(text=caption, lang='th')
                tts.save(speech_file_path)
                speech_file_paths.append(speech_file_path)
            elif("," in lang):
                speech_file_path = os.path.splitext(image_file_path)[0] + "_speech_en.mp3"
                tts = gTTS(text=caption, lang='en')
                tts.save(speech_file_path)

                tran = translate(caption)

                caption += "\n" + tran

                speech_file_path_th = os.path.splitext(image_file_path)[0] + "_speech_th.mp3"
                tts2 = gTTS(text=tran, lang='th')
                tts2.save(speech_file_path_th)

                speech_file_paths.append(speech_file_path)
            else:
                speech_file_path = os.path.splitext(image_file_path)[0] + "_speech_en.mp3"
                tts = gTTS(text=caption, lang='en')
                tts.save(speech_file_path)
                speech_file_paths.append(speech_file_path)
            
            allCaption.append(caption)
            
            print(speech_file_paths)            
            
    return jsonify({
        "captions": allCaption,
        "tts": [encode_mp3(x) for x in speech_file_paths]
    })

def encode_mp3(filename):
    # Read the MP3 file in binary mode
    with open(filename, 'rb') as mp3_file:
        mp3_data = mp3_file.read()

    # Encode the binary data to Base64
    encoded_mp3 = base64.b64encode(mp3_data).decode('utf-8')
    print(encoded_mp3)
    
    return encoded_mp3

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)