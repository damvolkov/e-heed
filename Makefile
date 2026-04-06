WAKEWORD      ?= $(word 2,$(MAKECMDGOALS))
N_SAMPLES     ?= 5000
N_SAMPLES_VAL ?= 1000
STEPS         ?= 50000
LANGS         ?= en
TTS           ?= kokoro

LEGACY := trainer/legacy

.PHONY: install sync lint type test test-slow check infra down deb clean-dist help

install: ## Install all dependencies + legacy artifacts
	uv sync --group dev --group train --python 3.12
	@mkdir -p $(LEGACY)
	@echo ">>> Cloning piper-sample-generator..."
	@test -d $(LEGACY)/piper-sample-generator || \
		git clone https://github.com/rhasspy/piper-sample-generator $(LEGACY)/piper-sample-generator
	@cd $(LEGACY)/piper-sample-generator && git checkout 213d4d5 2>/dev/null || true
	@mkdir -p $(LEGACY)/piper-sample-generator/models
	@test -f $(LEGACY)/piper-sample-generator/models/en_US-libritts_r-medium.pt || \
		wget -q --show-progress -O $(LEGACY)/piper-sample-generator/models/en_US-libritts_r-medium.pt \
		'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt'
	@echo ">>> Cloning openwakeword..."
	@test -d $(LEGACY)/openwakeword || \
		git clone https://github.com/dscripka/openwakeword $(LEGACY)/openwakeword
	@mkdir -p $(LEGACY)/openwakeword/openwakeword/resources/models
	@test -f $(LEGACY)/openwakeword/openwakeword/resources/models/embedding_model.onnx || \
		wget -q --show-progress -O $(LEGACY)/openwakeword/openwakeword/resources/models/embedding_model.onnx \
		https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx
	@test -f $(LEGACY)/openwakeword/openwakeword/resources/models/embedding_model.tflite || \
		wget -q --show-progress -O $(LEGACY)/openwakeword/openwakeword/resources/models/embedding_model.tflite \
		https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite
	@test -f $(LEGACY)/openwakeword/openwakeword/resources/models/melspectrogram.onnx || \
		wget -q --show-progress -O $(LEGACY)/openwakeword/openwakeword/resources/models/melspectrogram.onnx \
		https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx
	@test -f $(LEGACY)/openwakeword/openwakeword/resources/models/melspectrogram.tflite || \
		wget -q --show-progress -O $(LEGACY)/openwakeword/openwakeword/resources/models/melspectrogram.tflite \
		https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite
	@echo ">>> Downloading MIT RIRs..."
	@test -d $(LEGACY)/MIT_environmental_impulse_responses || \
		(command -v git-lfs >/dev/null 2>&1 && git lfs install && \
		git clone https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses $(LEGACY)/MIT_environmental_impulse_responses || \
		echo "⚠ Skipped MIT RIRs (git-lfs not installed: sudo apt install git-lfs)")
	@echo ">>> Downloading audioset sample..."
	@test -d $(LEGACY)/audioset || ( \
		mkdir -p $(LEGACY)/audioset && \
		wget -q --show-progress -O $(LEGACY)/audioset/bal_train09.tar \
		https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train09.tar && \
		cd $(LEGACY)/audioset && tar -xf bal_train09.tar \
	)
	@echo ">>> Downloading precomputed features..."
	@test -f $(LEGACY)/openwakeword_features_ACAV100M_2000_hrs_16bit.npy || \
		wget -q --show-progress -O $(LEGACY)/openwakeword_features_ACAV100M_2000_hrs_16bit.npy \
		https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
	@test -f $(LEGACY)/validation_set_features.npy || \
		wget -q --show-progress -O $(LEGACY)/validation_set_features.npy \
		https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy
	@echo ">>> Install complete. All artifacts in $(LEGACY)/"

sync: ## Sync dev dependencies
	uv sync --group dev --python 3.12

lint: ## Lint and format
	uv run ruff check --fix .
	uv run ruff format .

type: ## Type check
	uv run ty check src/

test: ## Run unit tests
	uv run pytest -v

test-slow: ## Run integration tests (requires running e-voice)
	uv run pytest -m slow -v

check: lint type test ## Full check: lint + type + test

infra: ## Start e-voice infrastructure
	docker compose -f compose.infra.yml up -d

down: ## Stop infrastructure
	docker compose -f compose.infra.yml down

deb: ## Build .deb package
	@./packaging/build.sh

clean-dist: ## Clean build artifacts
	@rm -rf dist/

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
