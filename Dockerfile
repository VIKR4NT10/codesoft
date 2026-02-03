FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*
# -------------------------------------------------
WORKDIR /app
# -------------------------------------------------
COPY flask_app/ /app/

RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords wordnet 
# -------------------------------------------------
EXPOSE 5000
# -------------------------------------------------

#local 

# CMD ["python", "app.py"]

# production

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
