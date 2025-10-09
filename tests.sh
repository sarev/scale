#!/usr/bin/env bash
#
# Copyright 2025 7th software Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

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
