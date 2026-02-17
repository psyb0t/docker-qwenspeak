FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS cuda-runtime

FROM psyb0t/lockbox:v2.1.1

ENV LOCKBOX_USER=tts
ENV TTS_LOG_RETENTION=7d
ENV PROCESSING_UNIT=cpu
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# CUDA runtime + cuDNN libs
COPY --from=cuda-runtime /usr/local/cuda/lib64/ /usr/local/cuda/lib64/
COPY --from=cuda-runtime /usr/lib/x86_64-linux-gnu/libcudnn* /usr/lib/x86_64-linux-gnu/
COPY --from=cuda-runtime /usr/lib/x86_64-linux-gnu/libnccl* /usr/lib/x86_64-linux-gnu/
RUN echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && ldconfig

# System deps for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        libsndfile1 \
        libsox-fmt-all \
        sox \
        ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Log directory (777 because UID is set at runtime via LOCKBOX_UID)
RUN mkdir -p /var/log/tts && chmod 777 /var/log/tts
VOLUME /var/log/tts

# Jobs directory (ephemeral, not a volume)
RUN mkdir -p /jobs && chmod 777 /jobs

# App
COPY tts.py /app/tts.py
RUN cat <<'EOF' > /usr/local/bin/tts && chmod +x /usr/local/bin/tts
#!/usr/bin/python3
import os, sys
os.chdir("/work")
sys.path.insert(0, "/app")
from tts import main
main()
EOF

# Lockbox config
COPY allowed.json /etc/lockbox/allowed.json
