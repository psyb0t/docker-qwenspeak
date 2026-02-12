FROM psyb0t/lockbox

ENV LOCKBOX_USER=tts

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
