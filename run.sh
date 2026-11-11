#!/usr/bin/env bash
# Runner script for American Express Decision Science Challenge
set -e

# Set dynamic library path for macOS OpenMP if applicable
if [ -d ".venv/lib/python3.14/site-packages/sklearn/.dylibs" ]; then
    export DYLD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.14/site-packages/sklearn/.dylibs:$DYLD_LIBRARY_PATH"
fi
export PYTHONPATH=.

case "$1" in
    data)
        echo "Generating authentic multi-season raw sports dataset..."
        .venv/bin/python src/data_loader.py
        ;;
    features)
        echo "Extracting pre-match causal features..."
        .venv/bin/python src/feature_engineering.py
        ;;
    hmm)
        echo "Fitting Gaussian HMM and computing latent form priors..."
        .venv/bin/python -m src.hmm_model
        ;;
    train)
        echo "Training Baseline LightGBM, Hybrid LightGBM, and XGBoost with GridSearchCV..."
        .venv/bin/python src/lightgbm_model.py
        .venv/bin/python src/xgboost_model.py
        .venv/bin/python src/evaluation.py
        ;;
    pipeline)
        echo "Running complete end-to-end ML pipeline..."
        .venv/bin/python src/data_loader.py
        .venv/bin/python src/feature_engineering.py
        .venv/bin/python -m src.hmm_model
        .venv/bin/python src/lightgbm_model.py
        .venv/bin/python src/xgboost_model.py
        .venv/bin/python src/evaluation.py
        ;;
    test)
        echo "Running Pytest test suite..."
        .venv/bin/python -m pytest tests/test_pipeline.py -v
        ;;
    app)
        echo "Launching interactive Streamlit Dashboard..."
        .venv/bin/python -m streamlit run app.py
        ;;
    *)
        echo "American Express Decision Science Challenge CLI Runner"
        echo "Usage: ./run.sh {data|features|hmm|train|pipeline|test|app}"
        exit 1
        ;;
esac
