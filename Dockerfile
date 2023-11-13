FROM ubuntu:23.10
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y unzip python3 python3-pip
RUN python3 -m pip install -r /app/requirements.txt --break-system-packages
RUN python3 -m pip install -e /app/detectron2 --break-system-packages
EXPOSE 8501
CMD ["streamlit", "run", "--server.address", "0.0.0.0", "/app/app.py"]