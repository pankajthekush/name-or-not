FROM python:3.8.12
WORKDIR /app
COPY . /app
RUN pip install -r /app/requirements.txt
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "3", "--threads", "2", "wsgi:app"]

