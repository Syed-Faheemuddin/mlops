FROM tensorflow/tensorflow

MAINTAINER Syed Faheemuddin <syedfaheemuddin456987@gmail.com>

RUN apt-get install net-tools -y
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install numpy
RUN pip3 install pillow
RUN pip3 install pandas
RUN pip3 install sklearn
RUN pip3 install tensorflow 
RUN pip3 install keras
RUN pip3 install matplotlib
RUN pip3 install seaborn

WORKDIR /root/mlops-project/
CMD ["python3","train.py"]
