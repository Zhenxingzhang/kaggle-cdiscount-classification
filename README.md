A repo for [Kaggle cdiscount image classification challenge](https://www.kaggle.com/c/cdiscount-image-classification-challenge)

## Python enviourment for running jupyter notebook

you can run the docker image for kaggle python(https://hub.docker.com/r/kaggle/python/) in case that you don't want to bother with the setup python enviourment in your machine(install dependencies and so on).

### Pull down the docker image
```
docker pull kaggle/python:latest
```

### Run the docker image
```
docker run -it -v ${pwd}/:/notebook -p 8888:8888 kaggle/python:latest
```

### Running Jupyter Notebook within Docker Container
Within docker container bash: 
```
jupyter notebook --ip=0.0.0.0 --allow-root /notebook 
```

Access through browser: http://192.168.99.100:8888
