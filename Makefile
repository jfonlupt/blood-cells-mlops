export GIT_COMMIT := $(shell git rev-parse HEAD)

PORT ?= 8000
BASE_URL ?= http://127.0.0.1:$(PORT)

all:
	make mlflow
	make airflow
	make server

.PHONY: mlflow
mlflow:
	docker compose up -d mlflow

.PHONY: grafana
grafana:
	docker compose up -d grafana

.PHONY: airflow
airflow:
	docker compose up -d airflow-webserver airflow-scheduler airflow-worker airflow-dag-processor flower
	docker compose exec airflow-webserver cat /opt/airflow/simple_auth_manager_passwords.json.generated

.PHONY: build-server
build-server:
	uv run bentoml build
	uv run bentoml containerize blood_cells_service:latest -t blood_cells_service:latest

.PHONY: server
server:
	docker compose up -d --wait blood_cells_server

.PHONY: stop
stop:
	docker compose down

.PHONY: clean
clean:
	docker compose down -v

.PHONY: build-pipeline
build-pipeline:
	docker compose build blood_cells_training

.PHONY: pipeline
pipeline:
	docker compose up blood_cells_training

.PHONY: test-predict
test-predict:
	@token=$$(curl -s -X POST "$(BASE_URL)/login" \
		-H "Content-Type: application/json" \
		-d '{"credentials":{"username":"user","password":"password"}}' | jq -r '.token'); \
	curl -s -X POST "$(BASE_URL)/predict" \
		-H "Authorization: Bearer $$token" \
		-F "model=baseline-type" \
		-F "images=@data/test_data/lymphocyte_0.jpg" \
		-F "images=@data/test_data/promyelocyte_0.jpeg"

.PHONY: test-evaluate
test-evaluate:
	@token=$$(curl -s -X POST "$(BASE_URL)/login" \
		-H "Content-Type: application/json" \
		-d '{"credentials":{"username":"user","password":"password"}}' | jq -r '.token'); \
	curl -s -X POST "$(BASE_URL)/evaluate" \
		-H "Authorization: Bearer $$token" \
		-F "model=baseline-type" \
		-F 'labels=["Lymphocyte", "Promyelocyte", "Neutrophil", "Promyelocite"]' \
		-F "images=@data/test_data/lymphocyte_0.jpg" \
		-F "images=@data/test_data/promyelocyte_0.jpeg" \
		-F "images=@data/test_data/neutrophil_0.jpg" \
		-F "images=@data/test_data/promyelocyte_Angel_Wings_001.jpg"
