SHELL := /usr/bin/env bash
ISAACLAB_ROOT ?= .deps/IsaacLab
GR00T_ROOT ?= .deps/Isaac-GR00T
SONIC_ROOT ?= .deps/GR00T-WholeBodyControl
ISAAC_PYTHON := $(ISAACLAB_ROOT)/.venv/bin/python
GR00T_PYTHON := bash scripts/groot/run_with_groot_env.sh $(GR00T_ROOT)/.venv/bin/python
TASK_FLAT := Unitree-G1-Velocity-Flat-Robust
TASK_ROUGH := Unitree-G1-Velocity-Rough-Robust
SMOKE_STEPS ?= 256
FLAT_NUM_ENVS ?= 8192
FLAT_MINI_BATCHES ?= 8
ROUGH_NUM_ENVS ?= 8192
ROUGH_MINI_BATCHES ?= 8
TRAIN_EXTRA_ARGS ?=
INSTRUCTION ?= Walk forward at a moderate speed.
MULTITASK_INSTRUCTION ?= Pick up the red apple and place it on the plate

.PHONY: setup setup-isaaclab setup-groot setup-locomanip setup-sonic setup-sonic-sim setup-sonic-teleop setup-sonic-data setup-applepnp-wbc \
	setup-sonic-inference setup-sonic-deploy download-sonic-model sonic-check doctor install list smoke-flat smoke-rough train-flat train-rough \
	train-flat-resume train-rough-resume status-rough benchmark-training performance-check play play-rough evaluate evaluate-rough collect convert audit-raw \
	validate-dataset finetune-groot serve-groot hierarchical demo-navigation-expert evaluate-navigation vision-ablation \
	generate-fetch-curriculum collect-fetch process-fetch audit-fetch finetune-fetch serve-fetch \
	download-fetch-dataset download-applepnp-model download-locomanip-model download-locomanip-dataset validate-fetch-dataset demo-fetch-sim validate-fetch-scenes fetch-preflight \
	smoke-applepnp serve-applepnp demo-applepnp \
	download-multitask-dataset audit-multitask-dataset validate-multitask-dataset smoke-multitask-scene \
	smoke-multitask-wbc-scene train-multitask serve-multitask demo-multitask lab-multitask demo-multitask-wbc evaluate-multitask \
	serve-locomanip demo-commanded-fetch smoke-commanded-fetch \
	fetch-collection-preflight smoke-fetch-stack test lint

setup: setup-isaaclab setup-groot

setup-isaaclab:
	bash scripts/setup/bootstrap_isaaclab.sh

setup-groot:
	bash scripts/setup/bootstrap_groot.sh

setup-locomanip:
	bash scripts/setup/bootstrap_locomanipulation.sh

setup-sonic:
	bash scripts/setup/bootstrap_sonic.sh

setup-sonic-sim: setup-sonic
	cd $(SONIC_ROOT) && bash install_scripts/install_mujoco_sim.sh

setup-sonic-teleop: setup-sonic
	cd $(SONIC_ROOT) && bash install_scripts/install_pico.sh

setup-sonic-data: setup-sonic
	cd $(SONIC_ROOT) && bash install_scripts/install_data_collection.sh

setup-applepnp-wbc:
	bash scripts/setup/bootstrap_applepnp_wbc.sh

setup-sonic-inference: setup-sonic
	bash scripts/setup/bootstrap_sonic_inference.sh

setup-sonic-deploy: setup-sonic
	bash scripts/setup/bootstrap_sonic_deploy.sh

download-sonic-model: setup-sonic
	test -x $(SONIC_ROOT)/.venv_sim/bin/python || { echo "Run 'make setup-sonic-sim' first" >&2; exit 1; }
	uv pip install --python $(SONIC_ROOT)/.venv_sim/bin/python huggingface_hub
	cd $(SONIC_ROOT) && .venv_sim/bin/python download_from_hf.py --sonic-v1-1

sonic-check:
	$(ISAAC_PYTHON) scripts/sonic/preflight.py --require source --strict

doctor:
	$(ISAAC_PYTHON) scripts/doctor.py --strict

install:
	uv pip install --python $(ISAAC_PYTHON) -e 'source/unitree_rl_groot[groot-client]'

list:
	$(ISAAC_PYTHON) scripts/list_envs.py

smoke-flat:
	$(ISAAC_PYTHON) scripts/random_agent.py --task $(TASK_FLAT) --num_envs 8 --max_steps $(SMOKE_STEPS) --viz none

smoke-rough:
	$(ISAAC_PYTHON) scripts/random_agent.py --task $(TASK_ROUGH) --num_envs 8 --max_steps $(SMOKE_STEPS) --viz none

train-flat:
	$(ISAAC_PYTHON) scripts/train.py --rl_library rsl_rl --task $(TASK_FLAT) \
		--num_envs $(FLAT_NUM_ENVS) --viz none \
		agent.algorithm.num_mini_batches=$(FLAT_MINI_BATCHES) $(TRAIN_EXTRA_ARGS)

train-flat-resume:
	$(ISAAC_PYTHON) scripts/train.py --rl_library rsl_rl --task $(TASK_FLAT) \
		--resume --checkpoint latest --num_envs $(FLAT_NUM_ENVS) --viz none \
		agent.algorithm.num_mini_batches=$(FLAT_MINI_BATCHES) $(TRAIN_EXTRA_ARGS)

train-rough:
	$(ISAAC_PYTHON) scripts/train.py --rl_library rsl_rl --task $(TASK_ROUGH) \
		--num_envs $(ROUGH_NUM_ENVS) --viz none \
		agent.algorithm.num_mini_batches=$(ROUGH_MINI_BATCHES) $(TRAIN_EXTRA_ARGS)

train-rough-resume:
	$(ISAAC_PYTHON) scripts/train.py --rl_library rsl_rl --task $(TASK_ROUGH) \
		--resume --checkpoint latest --num_envs $(ROUGH_NUM_ENVS) --viz none \
		agent.algorithm.num_mini_batches=$(ROUGH_MINI_BATCHES) $(TRAIN_EXTRA_ARGS)

status-rough:
	$(ISAAC_PYTHON) scripts/training_status.py

benchmark-training:
	$(ISAAC_PYTHON) scripts/benchmark_training.py

performance-check:
	$(ISAAC_PYTHON) scripts/performance_check.py

play:
	$(ISAAC_PYTHON) scripts/play.py --rl_library rsl_rl --task $(TASK_FLAT) \
		--checkpoint latest --num_envs 1 --viz kit

play-rough:
	$(ISAAC_PYTHON) scripts/play.py --rl_library rsl_rl --task $(TASK_ROUGH) \
		--checkpoint latest --num_envs 1 --viz kit

evaluate:
	$(ISAAC_PYTHON) scripts/evaluate.py --task $(TASK_FLAT) --checkpoint latest --viz none

evaluate-rough:
	$(ISAAC_PYTHON) scripts/evaluate.py --task $(TASK_ROUGH) --checkpoint latest --viz none

collect:
	$(ISAAC_PYTHON) scripts/groot/collect_navigation.py --checkpoint latest

audit-raw:
	$(ISAAC_PYTHON) scripts/groot/audit_raw_navigation.py --raw-dir datasets/raw/g1_navigation --strict

convert:
	bash scripts/groot/convert_dataset.sh

validate-dataset:
	$(GR00T_PYTHON) scripts/groot/validate_dataset.py --dataset datasets/lerobot/g1_navigation

finetune-groot:
	bash scripts/groot/finetune_navigation.sh

serve-groot:
	bash scripts/groot/serve_navigation.sh

hierarchical:
	$(ISAAC_PYTHON) scripts/groot/run_hierarchical.py --checkpoint latest --instruction "$(INSTRUCTION)"

demo-navigation-expert:
	$(ISAAC_PYTHON) scripts/groot/collect_navigation.py --checkpoint latest \
		--output-dir outputs/navigation_demo_raw --episodes 1 --max-attempts 10 \
		--episode-seconds 18 --goal-hold-seconds 1.6 --dataset-fps 10 \
		--obstacle-count 6 --viz kit

evaluate-navigation:
	$(ISAAC_PYTHON) scripts/groot/evaluate_navigation.py --checkpoint latest --viz none

vision-ablation:
	$(ISAAC_PYTHON) scripts/groot/evaluate_vision_ablation.py \
		--raw-dir datasets/raw/g1_navigation --strict

generate-fetch-curriculum:
	$(ISAAC_PYTHON) scripts/sonic/generate_fetch_curriculum.py

download-fetch-dataset:
	bash scripts/setup/download_fetch_dataset.sh

download-applepnp-model:
	bash scripts/setup/download_applepnp_model.sh

download-locomanip-model:
	bash scripts/setup/download_locomanipulation.sh

download-locomanip-dataset:
	bash scripts/setup/download_locomanipulation_dataset.sh

smoke-applepnp:
	$(GR00T_PYTHON) scripts/applepnp/smoke_applepnp.py

serve-applepnp:
	$(GR00T_PYTHON) scripts/applepnp/serve_applepnp.py

demo-applepnp:
	bash scripts/applepnp/demo_applepnp.sh

download-multitask-dataset:
	bash scripts/setup/download_multitask_dataset.sh

audit-multitask-dataset:
	$(ISAAC_PYTHON) scripts/multitask/audit_dataset.py \
		--root datasets/g1_fruits_multitask_hf --strict \
		--json-output outputs/g1_multitask_dataset_audit.json

validate-multitask-dataset:
	$(GR00T_PYTHON) scripts/multitask/validate_dataset.py \
		--root datasets/g1_fruits_multitask_hf

smoke-multitask-scene:
	export OMNI_KIT_ACCEPT_EULA=1; $(ISAAC_PYTHON) scripts/multitask/run_isaac_g1_fruits.py \
		--instruction "$(MULTITASK_INSTRUCTION)" --scene-smoke --viz none

smoke-multitask-wbc-scene:
	MUJOCO_GL=egl $(SONIC_ROOT)/.venv_sim/bin/python scripts/multitask/smoke_scene.py

train-multitask:
	bash scripts/multitask/finetune_g1_fruits.sh

serve-multitask:
	bash scripts/multitask/serve_g1_fruits.sh

demo-multitask:
	INSTRUCTION="$(MULTITASK_INSTRUCTION)" bash scripts/multitask/demo_isaac_g1_fruits.sh

lab-multitask:
	MULTITASK_INTERACTIVE=1 INSTRUCTION="" bash scripts/multitask/demo_isaac_g1_fruits.sh

demo-multitask-wbc:
	INSTRUCTION="$(MULTITASK_INSTRUCTION)" bash scripts/multitask/demo_g1_fruits.sh

evaluate-multitask:
	bash scripts/multitask/evaluate_g1_fruits.sh

serve-locomanip:
	.deps/Isaac-GR00T-N1.5/.venv/bin/python scripts/locomanip/serve_policy.py

demo-commanded-fetch:
	bash scripts/locomanip/demo_commanded_fetch.sh

smoke-commanded-fetch:
	bash scripts/locomanip/demo_commanded_fetch.sh --viz none --max-seconds 3 --smoke

collect-fetch:
	bash scripts/sonic/collect_fetch.sh

process-fetch:
	bash scripts/sonic/process_fetch_dataset.sh

audit-fetch:
	$(ISAAC_PYTHON) scripts/sonic/audit_fetch_dataset.py \
		--dataset datasets/sonic/g1_fetch_clean --strict

validate-fetch-dataset:
	$(GR00T_PYTHON) scripts/sonic/validate_fetch_dataset.py \
		--dataset datasets/sonic/g1_fetch_clean

finetune-fetch:
	bash scripts/sonic/finetune_fetch.sh

serve-fetch:
	bash scripts/sonic/serve_fetch.sh

demo-fetch-sim:
	bash scripts/sonic/demo_fetch_sim.sh

validate-fetch-scenes:
	$(SONIC_ROOT)/.venv_sim/bin/python scripts/sonic/validate_fetch_scenes.py

fetch-preflight:
	$(ISAAC_PYTHON) scripts/sonic/preflight.py --require demo-sim --strict

fetch-collection-preflight:
	$(ISAAC_PYTHON) scripts/sonic/preflight.py --require collection-sim --strict

smoke-fetch-stack:
	bash scripts/sonic/smoke_fetch_stack.sh

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(ISAAC_PYTHON) -m pytest

lint:
	$(ISAAC_PYTHON) -m ruff check .
	$(ISAAC_PYTHON) -m ruff format --check .
