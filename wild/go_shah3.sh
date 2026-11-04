#!/usr/bin/env bash
# Paper 4.3 reproduction, user's go 2026-11-03. Openings: all 2,750 (275 roles x 10 questions), opening reply 512
# tokens; turn 2: Shah-style questions; temp 0.7 / top-p 0.9 / 512 tokens (repo defaults).
#   main:   2,750 openings x 42 Kimi-K2-written questions (writer not the judge's family)
#   checks: 550 openings (first 2 questions) x 42 DeepSeek-written and x 17 GPT-4-written questions
#   baselines: each question alone x 10 samples (the paper's baseline); 10 no-role openings x all three sets
# Then Qwen3Guard-8B on ToxicChat. Judging runs on the laptop.
set -uo pipefail
for v in RUNPOD_POD_ID RUNPOD_API_KEY; do export $v="$(tr "\0" "\n" < /proc/1/environ | grep "^$v=" | cut -d= -f2-)"; done
export HF_HOME=/dev/shm/hf PYTHONPATH=/workspace/repo PATH=/workspace/venv/bin:$PATH
PY=/workspace/venv/bin/python
R=/workspace/wild/roles_shah; B=/workspace/wild/noroles_shah; Q=data/dl/chqa/data/catqa_english.json; D=data/shah
mkdir -p $R $B; cd /workspace/repo/wild
exec >> $R/run.log 2>&1
trap 'echo "== $(date -u +%FT%TZ) end"; touch /workspace/.shah_done; for i in $(seq 90); do [ -f /workspace/.shah_fetched ] && break; sleep 20; done; /usr/bin/runpodctl stop pod "$RUNPOD_POD_ID"' EXIT
step() { echo "== $(date -u +%FT%TZ) $*"; }
while pgrep -f "scripts/setup.sh" >/dev/null; do sleep 20; done
[ -x $PY ] && $PY -c "import vllm" 2>/dev/null || { step setup; (cd /workspace/repo && bash sanity_check/scripts/setup.sh); }
step weights; $PY -c "from huggingface_hub import snapshot_download as s; s('unsloth/Llama-3.3-70B-Instruct', allow_patterns=['*.json', '*.safetensors', 'tokenizer*'])"
[ -f $R/selected.jsonl ] || { step build; $PY roles_build.py $R; }
[ -f $B/selected.jsonl ] || { step build-noroles; $PY roles_build.py $B --no-role; }
for d in $R $B; do [ -f $d/drift.jsonl ] || { step pool $d; $PY harm_wild.py $d $Q pool --first-max-tokens 512; }; done
for d in $R $B; do step activations $d; $PY activations.py $d; done
[ -f $R/harm_wild.jsonl ] || { step harm main; $PY harm_wild.py $R $Q harm --per-band 100000 --questions-jsonl $D/questions_kimi_42.jsonl; }
[ -f $R/harm_wild_ds42.jsonl ] || { step harm ds42; $PY harm_wild.py $R $Q harm --per-band 100000 --hash-regex='-q(216|98)$' --questions-jsonl $D/questions_42.jsonl --tag _ds42; }
[ -f $R/harm_wild_gpt4.jsonl ] || { step harm gpt4; $PY harm_wild.py $R $Q harm --per-band 100000 --hash-regex='-q(216|98)$' --questions-jsonl $D/questions_gpt4_17.jsonl --tag _gpt4; }
for t in kimi_42:_k2 42:_ds42 gpt4_17:_gpt4; do f=${t%%:*}; g=${t##*:}; [ -f $B/harm_wild$g.jsonl ] || { step noroles $g; $PY harm_wild.py $B $Q harm --per-band 100000 --questions-jsonl $D/questions_$f.jsonl --tag $g; }; done
[ -f $B/alone.jsonl ] || { step alone; $PY alone.py $B/alone.jsonl $D/questions_kimi_42.jsonl $D/questions_42.jsonl $D/questions_gpt4_17.jsonl --samples 10; }
[ -f data/cls/toxicchat_test.cls_qwen3guard.jsonl ] || { step qwen3guard; $PY classify.py data/cls/toxicchat_test.jsonl qwen3guard --batch 64; }
step done
