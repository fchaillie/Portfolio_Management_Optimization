FROM python:3.11-slim


# Environment sanity
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy your app code
COPY . .

# expose local port
EXPOSE 8501

# run Streamlit
CMD ["bash", "-c", "streamlit run app/main.py --server.headless true --server.address 0.0.0.0 --server.port ${PORT:-8501}"]

