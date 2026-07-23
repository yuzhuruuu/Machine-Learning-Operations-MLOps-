# Dockerfile - Model Serving (TensorFlow Serving) untuk Prediksi Risiko Diabetes
# Dipakai untuk deployment ke platform cloud (Railway / Heroku).
#
# Build & run lokal (untuk pengujian sebelum deploy):
#   docker build -t yusriiii-diabetes-serving .
#   docker run -p 8501:8501 -e PORT=8501 yusriiii-diabetes-serving
#
# Setelah berjalan, model dapat diakses melalui:
#   http://localhost:8501/v1/models/diabetes-model/metadata
#   http://localhost:8501/monitoring/prometheus/metrics   (endpoint metrik Prometheus)

FROM tensorflow/serving:latest

# Copy model hasil Pusher (format SavedModel dengan direktori versi) ke
# direktori model TensorFlow Serving.
COPY ./serving_model_deploy/diabetes-model /models/diabetes-model

# Copy konfigurasi monitoring Prometheus bawaan TF Serving.
COPY ./config /model_config

ENV MODEL_NAME=diabetes-model
ENV MODEL_BASE_PATH=/models

# Railway (maupun Heroku) menyediakan variabel environment $PORT secara
# otomatis; TF Serving REST API akan mendengarkan pada port tersebut.
RUN echo '#!/bin/bash \n\n\
env \n\
tensorflow_model_server \
--port=8500 \
--rest_api_port=${PORT} \
--model_name=${MODEL_NAME} \
--model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
--monitoring_config_file=/model_config/prometheus.config \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
