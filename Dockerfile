FROM gcr.io/tensorflow/tensorflow:1.4.0-gpu

WORKDIR /data

COPY run_tools.sh /
RUN chmod +x /run_tools.sh

WORKDIR /app
CMD ["/run_tools.sh", "--allow-root"]