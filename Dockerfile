# Dockerfile para o projeto LSTM Stock Prediction

FROM python:3.12-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos de dependências
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY src/ ./src/
COPY api/ ./api/

# Expor porta da API
EXPOSE 5000

# Comando para executar a API
CMD ["python", "api/app.py"]

