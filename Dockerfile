FROM python:3.13
WORKDIR /zeek_ml
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python","prepare-data.py","data/input","data/output"]

