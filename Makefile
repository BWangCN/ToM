SHELL := /bin/bash
PY := .venv/bin/python
export PYTHONPATH := $(CURDIR)/src
export HF_HOME := $(HOME)/hf_cache
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — it is
# broken under WSL2 (CUDA "device not ready" inside optimizer foreach ops).

.PHONY: env data test canary_pre_a stage_a canary_post_a stage_b canary_post_b \
        resume_check stage_b_thinkonly perturb ablation report smoke clean_runs

env:
	mkdir -p reports
	nvidia-smi | tee reports/nvidia_smi.txt
	$(PY) -c "import json; from dvla.common import env_report; \
	  info = env_report(); print(json.dumps(info, indent=1)); \
	  open('reports/env.json','w').write(json.dumps(info, indent=1)); \
	  assert info['cuda_available'], 'torch does not see CUDA'"

data:
	mkdir -p reports
	$(PY) scripts/make_toy_data.py --check-determinism | tee reports/data_check.txt
	$(PY) scripts/validate_jsonl.py | tee -a reports/data_check.txt

test:
	mkdir -p reports
	$(PY) -m pytest -q tests 2>&1 | tee reports/pytest.txt

canary_pre_a:
	$(PY) scripts/run_canary.py --init fresh --tag canary_pre_a

# Two legs: if the first 300-step run ends with criteria unmet (exit 3), resume
# for up to 300 more steps in a SECOND run — each leg stays under the 30-min
# hard rule; ">= 300 steps" is satisfied by leg 1.
stage_a:
	$(PY) scripts/train_sft.py --stage A --out-dir runs/stage_a || \
	$(PY) scripts/train_sft.py --stage A --out-dir runs/stage_a \
	  --resume runs/stage_a/ckpt_final --max-steps 600 --resume-extra-steps 300

canary_post_a:
	$(PY) scripts/run_canary.py --init runs/stage_a/ckpt_final --tag canary_post_a

stage_b:
	$(PY) scripts/train_sft.py --stage B --out-dir runs/stage_b \
	  --init-from runs/stage_a/ckpt_final --save-mid 20 || \
	$(PY) scripts/train_sft.py --stage B --out-dir runs/stage_b \
	  --resume runs/stage_b/ckpt_final --max-steps 600 --resume-extra-steps 300

canary_post_b:
	$(PY) scripts/run_canary.py --init runs/stage_b/ckpt_final --tag canary_post_b

resume_check:
	$(PY) scripts/train_sft.py --stage B --out-dir runs/stage_b_resume \
	  --resume runs/stage_b/ckpt_mid --resume-extra-steps 5 --tag stage_b_resume \
	  --no-criteria-gate

stage_b_thinkonly:
	$(PY) scripts/train_sft.py --stage B --decoder-condition think_only \
	  --out-dir runs/stage_b_thinkonly --init-from runs/stage_a/ckpt_final \
	  --tag stage_b_thinkonly || \
	$(PY) scripts/train_sft.py --stage B --decoder-condition think_only \
	  --out-dir runs/stage_b_thinkonly --resume runs/stage_b_thinkonly/ckpt_final \
	  --max-steps 600 --resume-extra-steps 300 --tag stage_b_thinkonly

perturb:
	$(PY) scripts/perturb_test.py --checkpoint runs/stage_b/ckpt_final

ablation:
	$(PY) scripts/ablation_eval.py --default-ckpt runs/stage_b/ckpt_final \
	  --thinkonly-ckpt runs/stage_b_thinkonly/ckpt_final

report:
	$(PY) scripts/eval_smoke.py --checkpoint runs/stage_b/ckpt_final

smoke:
	@mkdir -p reports && date +%s > reports/.smoke_t0
	$(MAKE) env
	$(MAKE) data
	$(MAKE) test
	$(MAKE) canary_pre_a
	$(MAKE) stage_a
	$(MAKE) canary_post_a
	$(MAKE) stage_b
	$(MAKE) canary_post_b
	$(MAKE) resume_check
	$(MAKE) stage_b_thinkonly
	$(MAKE) perturb
	$(MAKE) ablation
	$(MAKE) report
	@t0=$$(cat reports/.smoke_t0); t1=$$(date +%s); \
	  echo "SMOKE TOTAL WALL: $$(( (t1-t0)/60 )) min $$(( (t1-t0)%60 )) s"; \
	  $(PY) -c "import sys; sys.path.insert(0,'src'); from dvla.common import record_timing; record_timing('smoke_total', $$t1-$$t0)"
	@echo "SMOKE DONE -> reports/smoke_report.md"

clean_runs:
	rm -rf runs reports/*.json reports/*.md reports/*.png reports/*.txt reports/.smoke_t0
