## Start docker container
docker pull kaggle/python:latest

docker run -it -v ${pwd}/KaggleCompetitions/cdiscount-image-classification/:/home -p 8888:8888 kaggle/python:latest

### Running Jupyter Notebook within Docker Container
Within docker container:
jupyter notebook --ip=0.0.0.0 --allow-root

Access through browser: http://192.168.99.100:8888
