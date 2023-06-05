FROM python:3.8.12
WORKDIR /app
COPY . /app
RUN pip install -r /app/requirements.txt
RUN pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pytest test_model.py
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "3", "--threads", "2", "wsgi:app"]

