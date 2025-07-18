# Change to source code directory
cd src

# Run the training pipeline with default arguments
python run.py \
    --data us \
    --seed 0 \
    --guid 0.25 \
    --window 8 \
    --statwindow 256 \
    --epochs 500 \
    --units 256 \
    --rve 64 \
    --lr 0.001 \
    --batch-size 8 \
    --patience 35 \
    --dropout 0.5 \
    --invest \
    --save \
    --fm lstm
