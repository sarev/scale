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
python scale.py -c /d/Programming/uni-remote/hat/webapp/app.py -o temp/app.py -v --n-ctx 64000 --n-batch=192
