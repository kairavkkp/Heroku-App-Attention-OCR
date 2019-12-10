import os
import numpy as np
import glob
import sys,argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util
import flask
from flask import render_template, send_from_directory, request
from flask import redirect,url_for
from flask import jsonify
import base64
from io import StringIO
from werkzeug.utils import secure_filename



app = flask.Flask(__name__)
UPLOAD_FOLDER='static'


def load_graph(frozen_graph_filename):
    #with tf.gfile.GFile(frozen_graph_filename,"rb") as f:
    #    restored_graph_def = tf.GraphDef()
    #    restored_graph_def.ParseFromString(f.read())

    #with tf.Graph().as_default() as graph:
    #    tf.import_graph_def(restored_graph_def,
    #                       input_map=None,
    #                       return_elements=None,
    #                       name='')
    graph = tf.Graph()
    graphDef = tf.GraphDef()
    with open(frozen_graph_filename,"rb") as graphFile:
         graphDef.ParseFromString(graphFile.read())
    with graph.as_default():
	 tf.import_graph_def(graphDef)
    return graph


def getImage(path):
    with open(path, 'rb') as img_file:
        img = img_file.read()
    return img

graph = load_graph('manually_frozen_graph_1.14.pb')

@app.route('/',methods=['POST','GET'])
def demo():
    if request.method == 'POST':
        upload_file = request.files['file']
        filename = secure_filename(upload_file.filename)
        upload_file.save(os.path.join(UPLOAD_FOLDER,filename))

        image = getImage(os.path.join(UPLOAD_FOLDER,filename))

        x =  graph.get_tensor_by_name('input_image_as_bytes:0')
        probabilities = graph.get_tensor_by_name("probability:0")
        y = graph.get_tensor_by_name("prediction:0")

        with tf.Session(graph=graph) as sess:
            (y_out,probs_output) = sess.run([y,probabilities],feed_dict={x:[image]})


        result = {"PREDICTIONS": [{"ocr":str(y_out)}]}

        return jsonify(result)
    return '''
     <!doctype html>
    <html lang="en">
    <head>
      <title>IITB Assignment</title>
    </head>
    <body>
    <div class="site-wrapper">
        <div class="cover-container">

          <div class="inner cover">
          </div>
          <div class="mastfoot">
          <hr />
            <div class="container">
              <div style="margin-top:5%">
		            <h1 style="color:black">Attention OCR Custom Model-IITB Assignment</h1>
		            <h4 style="color:black">Upload new Image (Image should have height<=32px and width<=256) </h4>
		            <form method=post enctype=multipart/form-data>
	                 <p><input type=file name=file>
        	        <input type=submit style="color:black;" value=Upload>
		            </form>
	            </div>
            </div>
        	</div>
     </div>
   </div>
</body>
</html>


    '''


if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
