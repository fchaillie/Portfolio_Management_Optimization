FROM python:3.11-slim

WORKDIR /app

# install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy your app code
COPY . .

# expose local port (Render will override with $PORT)
EXPOSE 8501

# run Streamlit (use $PORT if defined, else 8501 for local)
CMD ["bash", "-c", "streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port ${PORT:-8501}"]

