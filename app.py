import os
import io
import uuid
import shutil
import glob
import warnings
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
from utils import load_model, inference

warnings.filterwarnings("ignore")

app = Flask(__name__)
device = "cpu"
model = load_model("./assets/model", "model_ckpt.pth", device=device)


@app.route('/', methods=['GET', 'POST'])
def index():
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        
        if file and allowed_file(file.filename):
            image = Image.open(file.stream)
            box_image = inference(model, image, score_thresh=0.75, device=device)

            # Generate a unique filename and save the image temporarily
            temp_filename = f"image_{uuid.uuid4()}.jpg"
            box_image.save(os.path.join('static', temp_filename))
            
            # Pass the filename to the template
            return render_template('inference.html', image_path=temp_filename)

    elif request.method == 'GET':
        # remove temporary images
        temp_images = glob.glob(os.path.join('static', 'image_*.jpg'))
        for image in temp_images:
            os.remove(image)
            
        # Get all images from the db folder
        db_images = glob.glob(os.path.join('static', 'db', '*.jpg'))
        return render_template('index.html', images=db_images)
    
    return render_template('index.html')



@app.route('/save_image/<filename>', methods=['POST']) 
def save_image(filename):

    src = os.path.join('static', filename)
    dest = os.path.join('static', 'db', filename)

    try:
        shutil.move(src, dest)
        print(f"Saving image {filename} to db")
        return redirect(url_for('index'))
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return "Error saving image", 500
    

@app.route('/remove_image/<filename>', methods=['DELETE'])
def remove_image(filename):
    file_path = os.path.join('static', 'db', filename)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return {"message": f"{filename} deleted successfully"}, 200
        else:
            return {"message": f"{filename} not found"}, 404
    except Exception as e:
        return {"message": str(e)}, 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
