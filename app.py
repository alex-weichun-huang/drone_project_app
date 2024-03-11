import os
import uuid
import glob
import torch
import boto3
from PIL import Image
from utils import (
    inference,
    load_model, 
    create_presigned_url,
    list_files_in_bucket,
)
from flask import (
    Flask, 
    request, 
    render_template, 
    redirect, 
    url_for
)
app = Flask(__name__)


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
            return render_template('inference.html', image_path=temp_filename)

    elif request.method == 'GET':
        # remove temporary images
        temp_images = glob.glob(os.path.join('static', 'image_*.jpg'))
        for image in temp_images:
            os.remove(image)
            
        # Get all the stored images from the S3 bucket
        try:
            db_images = list_files_in_bucket("drone-project-app")
            return render_template('index.html', images=db_images)
        except Exception as e:
            return {"message": str(e)}, 500
      
    return render_template('index.html')


@app.route('/save_image/<filename>', methods=['POST']) 
def save_image(filename):
    try:
        s3 = boto3.resource('s3')
        s3.Bucket("drone-project-app").upload_file(os.path.join('static', filename), filename)
        return redirect(url_for('index'))
    except Exception as e:
        return {"message": str(e)}, 500
   
      
@app.route('/remove_image/<filename>', methods=['DELETE'])
def remove_image(filename):
    s3 = boto3.client('s3')
    try:
        s3.delete_object(Bucket="drone-project-app", Key=filename)
        return {"message": f"File {filename} removed successfully."}, 200
    except Exception as e:
        return {"message": str(e)}, 500
   
   
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("./assets/model", "model_ckpt.pth", device=device)
    app.run(debug=True, port=5001)
