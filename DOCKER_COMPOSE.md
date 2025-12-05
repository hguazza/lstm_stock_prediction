# Docker Compose - Guia de Uso

Este guia explica como usar o `docker-compose.yml` para executar a stack completa do projeto LSTM Stock Prediction.

## Estrutura dos Serviços

### 1. API Service (`api`)
- **Porta**: 8000
- **Função**: Serviço de inferência que expõe a API FastAPI
- **Health Check**: `/health` endpoint
- **Dependências**: Redis (deve estar saudável antes de iniciar)

### 2. Training Service (`training`)
- **Função**: Executa o pipeline de treinamento do modelo
- **Volumes**: Compartilha `artifacts/` com a API
- **Profile**: Executa apenas quando o profile `training` é ativado

### 3. Redis Service (`redis`)
- **Porta**: 6379
- **Função**: Cache e fila de mensagens (preparado para uso futuro)
- **Persistência**: Volume nomeado `redis_data`

## Comandos Principais

### Iniciar apenas a API e Redis
```bash
docker-compose up -d
```

### Iniciar com serviço de treinamento
```bash
docker-compose --profile training up -d
```

### Executar treinamento uma vez
```bash
docker-compose --profile training run --rm training python train_pipeline.py
```

### Ver logs
```bash
# Todos os serviços
docker-compose logs -f

# Apenas API
docker-compose logs -f api

# Apenas Training
docker-compose logs -f training
```

### Parar serviços
```bash
docker-compose down
```

### Parar e remover volumes
```bash
docker-compose down -v
```

### Rebuild das imagens
```bash
docker-compose build --no-cache
```

## Volumes

### Artifacts
- **Bind Mount**: `./artifacts:/app/artifacts` (desenvolvimento)
- **Volume Nomeado**: `artifacts_data` (persistência em produção)

Os modelos treinados são salvos em `artifacts/` e compartilhados entre API e Training.

### Redis Data
- **Volume Nomeado**: `redis_data` (persistência de dados do Redis)

## Variáveis de Ambiente

Copie `env.example` para `.env` e ajuste conforme necessário:

```bash
cp env.example .env
```

Principais variáveis:
- `ARTIFACTS_DIR`: Diretório dos artefatos
- `MODEL_WEIGHTS_PATH`: Caminho dos pesos do modelo
- `CONFIG_PATH`: Caminho do arquivo de configuração
- `SCALER_PATH`: Caminho do scaler

## Fluxo de Trabalho

### 1. Treinar Modelo
```bash
# Opção 1: Executar treinamento como job
docker-compose --profile training run --rm training

# Opção 2: Executar treinamento localmente
python train_pipeline.py
```

### 2. Iniciar API
```bash
docker-compose up -d api
```

### 3. Testar API
```bash
# Health check
curl http://localhost:8000/health

# Predição (exemplo)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": [[100.0, 105.0, 99.0, 103.0, 1000000.0], ...],
    "steps": 5
  }'
```

## Troubleshooting

### API não inicia
- Verifique se os artefatos existem em `artifacts/`
- Verifique os logs: `docker-compose logs api`
- Certifique-se de que o modelo foi treinado primeiro

### Training falha
- Verifique se há espaço em disco suficiente
- Verifique os logs: `docker-compose logs training`
- Certifique-se de que o diretório `data/` existe

### Redis não conecta
- Verifique se o serviço está rodando: `docker-compose ps redis`
- Verifique os logs: `docker-compose logs redis`

## Desenvolvimento

Para desenvolvimento local sem Docker:

```bash
# Treinar modelo
python train_pipeline.py

# Iniciar API
uvicorn src.api.main:app --reload
```

## Produção

Para produção, considere:
- Usar variáveis de ambiente via `.env`
- Configurar reverse proxy (nginx)
- Usar volumes nomeados para persistência
- Configurar backups dos volumes
- Monitoramento e logging adequados

