from flask import Flask
import os

UPLOAD_FOLDER = os.getcwd() + '/uploads'

#UPLOAD_FOLDER = "C:\\Users\\Reddy\\Desktop\\proj_gui\\uploads"

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
