#!/usr/bin/env bash

M1="../models/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
M2="../models/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"

echo
echo "JavaScript test"
python scale.py -m "$M1" -c ../../uni-remote/hat/webapp/static/editor.js -o temp/editor.js -v

echo
echo "C test"
python scale.py -m "$M1" -c ../../RISC\ OS/C_BASIC/c/workspace.c -o temp/workspace.c -v

echo
echo "Python test"
python scale.py -c /d/Programming/uni-remote/hat/webapp/app.py -o temp/app.py -v --n-ctx 12288 --n-batch=256
python scale.py -c scale.py -o temp/scale.py -v --n-ctx 12288 --n-batch=256
python scale.py -c scale_c.py -o temp/scale_c.py -v --n-ctx 12288 --n-batch=256
python scale.py -c scale_python.py -o temp/scale_python.py -v --n-ctx 12288 --n-batch=256
python scale.py -c scale_javascript.py -o temp/scale_javascript.py -v --n-ctx 12288 --n-batch=256
python scale.py -c scale_log.py -o temp/scale_log.py -v --n-ctx 12288 --n-batch=256
python scale.py -c scale_llm.py -o temp/scale_llm.py -v --n-ctx 12288 --n-batch=256
