#!/usr/bin/env bash
# BATATA — inicia a aplicação web
# Uso: ./run.sh
# Opcional: BATATA_DATA_PATH=/caminho/para/potato_data ./run.sh

cd "$(dirname "$0")"
python app.py
