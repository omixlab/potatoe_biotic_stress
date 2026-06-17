#!/usr/bin/env bash
# BiSPoLP — inicia a aplicação web
# Uso: ./run.sh
# Opcional: BISPOLP_DATA_PATH=/caminho/para/potato_data ./run.sh

cd "$(dirname "$0")"
python app.py
