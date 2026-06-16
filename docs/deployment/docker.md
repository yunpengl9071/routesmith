# Docker Deployment

## Build the Image

```bash
docker build -t routesmith:v0.2.0 .
```

## Run the Container

```bash
docker run --rm -p 8000:8000 \
  -v ./routesmith.yaml:/app/routesmith.yaml \
  routesmith:v0.2.0
```

## With docker-compose

```bash
docker compose up -d
# Routesmith on :8000
```

## Health Checks

```bash
curl http://localhost:8000/health
curl http://localhost:8000/live
curl http://localhost:8000/ready
```

## Monitoring

```bash
curl http://localhost:8000/metrics | grep routesmith
```