FROM python@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

WORKDIR /app
COPY fault_matrix/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY action_dynamics/application_telemetry.py /app/application_telemetry.py
COPY action_dynamics/interventions.py /app/interventions.py
COPY action_dynamics/application.py /app/application.py
COPY action_dynamics/run_capture.py /app/run_capture.py

CMD ["python", "/app/application.py", "api"]
