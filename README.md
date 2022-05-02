# Seq2Seq - Sorting
Over-engineering a solution to a classic computer science problem by training a stupidly big neural network to do it more slowly and less accurately 👍

## Problem
Given a string containing lowercase letters, sort the string. The input strings:
- contain only lowercase letters (`abcdefghijklmnopqrstuvwxyz`)
- may contain duplicates
- are between 1 and 26 characters long (inclusive)

Here are some examples:
```text
Input -> Desired Output
-----------------------
dcba -> abcd
alcjhq -> achjlq
az -> az
bbabbaba -> aaabbbbb
```

## Results
Trained for 64 epochs and taking the best model (by val loss), the model achieves an accuracy of 95% (an output is considered correct if it matches the desired output exactly). Here are 4 examples of the model getting it right, and 4 of the model getting it wrong:
```text
---- Example 0 ----
src = yvdazaaqwjqlhfa
trg = aaaadfhjlqqvwyz
prd = aaaadfhjlqqvwyz
score = 1

---- Example 1 ----
src = krgecwhpdygwnvnl
trg = cdegghklnnprvwwy
prd = cdegghklnnprvwwy
score = 1

---- Example 2 ----
src = kzxxxoytfi
trg = fikotxxxyz
prd = fikotxxxyz
score = 1

---- Example 3 ----
src = elfbcbajynchten
trg = abbcceefhjlnnty
prd = abbcceefhjlnnty
score = 1

...

---- Example 99597 ----
src = qlcuqnebjrb
trg = bbcejlnqqru
prd = bbbcejlnqqru
score = 0

---- Example 99655 ----
src = wirauayekicybi
trg = aabceiiikruwyy
prd = aaabceiiikruwyy
score = 0

---- Example 99737 ----
src = urcuwuvttpjuzbkt
trg = bcjkprtttuuuuvwz
prd = bcjkprtttuuuuuvwz
score = 0

---- Example 99908 ----
src = yxcbynnoxlpzzzxhphtpuu
trg = bchhlnnoppptuuxxxyyzzz
prd = bchhlnnopptuuxxxyyzzz
score = 0
```
Seems the model struggles with duplicates. Note, the model took 20 minutes to train on an RTX 3090.

## Usage
**Install Dependencies:**
```bash
poetry install
poetry run pip install torch==1.11.0+cu113 --no-cache -f https://download.pytorch.org/whl/torch_stable.html  # must be run anytime poetry.lock is changed
```

**Train Model:**
```bash
poetry run python train.py \
    models/run1 \
    --gpus 1 \
    --gradient_clip_val 1 \
    --max_epochs 64 \
    --val_check_interval 0.2
```

**Evaluate Model:**
```bash
poetry run python evaluate.py models/run1
```

**Run Predictions:**
```bash
poetry run python predict.py models/run1 lakjoiuc dcba acyqtb
```

**Run Tests:**
```bash
poetry run pytest tests.py
```

**Run API:**
```bash
# run api
poetry run uvicorn app:app

# test api endpoint
curl -X POST 'http://localhost:8000/api/predict' \
    -H 'Content-Type: application/json' \
    -H "X-API-Key: ${API_KEY}" \
    -d '{"input_sequence":"bcdaefg"}'
```
Docs found at `https://localhost:8000/redoc`.

**Docker Image for API:**
```bash
# build image
sudo docker build -t ${SERVICE_NAME} .

# run image
docker run -it -p 8000:8000 --rm -e API_KEY=${API_KEY} ${SERVICE_NAME}

# test api endpoint
curl -X POST 'http://localhost:8000/api/predict' \
    -H 'Content-Type: application/json' \
    -H "X-API-Key: ${API_KEY}" \
    -d '{"input_sequence":"bcdaefg"}'
```
