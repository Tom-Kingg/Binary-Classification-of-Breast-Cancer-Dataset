@echo off
echo Installing required packages...
pip install -r requirements.txt

echo Training the model pipeline...
python train_pipeline.py

echo Running prediction on sample input...
python predict.py sample_input.json

pause
