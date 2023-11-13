# Run the APP with Docker

1. Build the container

```
docker build . -t drone_app
```

2. Run the App

```
docker run -p 8501:8501 drone_app
```

3. Follow the URL link to view the app in browser


#  Run the APP without Docker

1. Install conda for environment management (The following commands works for Linux system):

```sh
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

2. Set up the environment:

```sh
conda create --name drone_app python=3.9
conda activate drone_app
python -m pip install -r requirements.txt
python -m pip install -e detectron2
```

3. Run the App

``` 
streamlit run --server.address 0.0.0.0 app.py
```
