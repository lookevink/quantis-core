FROM python@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

WORKDIR /app
COPY fault_matrix/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY fault_matrix/application_logging.py /app/application_logging.py
COPY fault_matrix/service.py /app/fault_matrix_service.py
COPY fault_matrix/run_experiment.py /app/fault_matrix_run_experiment.py
COPY graph_jepa/operational_state.py /app/operational_state.py
COPY graph_jepa/service.py graph_jepa/run_experiment.py /app/

CMD ["python", "/app/service.py", "api"]
