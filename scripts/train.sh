python scripts/train.py
python scripts/train.py train.task=mlm
python scripts/train.py data.dataset.name="20231101.uk" experiment_name="gemma-3-4b--mntp--no-quant--ukr"
python scripts/train.py train.task=mlm data.dataset.name="20231101.uk" experiment_name="gemma-3-4b--mlm--no-quant--ukr"