# Dockerfile - Prometheus (Monitoring)
# Menjalankan Prometheus untuk memonitor metrik dari sistem model serving
# (TensorFlow Serving) yang telah di-deploy ke cloud.
#
# Build & run lokal:
#   cd monitoring
#   docker build -t yusriiii-prometheus .
#   docker run -p 9090:9090 yusriiii-prometheus
#
# Setelah berjalan, dashboard Prometheus dapat diakses melalui:
#   http://localhost:9090

FROM prom/prometheus:latest

COPY prometheus.yml /etc/prometheus/prometheus.yml

EXPOSE 9090

ENTRYPOINT ["/bin/prometheus", "--config.file=/etc/prometheus/prometheus.yml"]
