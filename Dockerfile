# Usamos Python 3.10 para evitar incompatibilidades
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Paquetes del sistema necesarios para numpy / scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libatlas-base-dev && \
    rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Puerto (Render lo inyecta, pero dejamos por defecto)
ENV PORT=10000

# Comando de arranque: Gunicorn sirviendo el objeto `server` de app.py
CMD gunicorn app:server --bind 0.0.0.0:${PORT} --timeout 120
