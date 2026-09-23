# timsim-necro — set up the lean simulator. Everything is ingested from published/git sources; no rustims.
PREFIX ?= $(HOME)/.local

.PHONY: predict-deps rust-bins setup
predict-deps:   ## the lean prediction + orchestration stack (necroflow + timsim-predict -> pepdl -> mscorepy)
	pip install -r requirements.txt

rust-bins:      ## the timsim-cli protocol/render binaries — installed from git, crates.io deps only
	cargo install --git https://github.com/MS-Simulation/timsim-cli --rev 8e0816bc1da84896b6e13be739eec77ad1a0a64a --locked --features tdf,thermo,sciex --root $(PREFIX)

setup: predict-deps rust-bins   ## everything the simulator needs — no rustims, no imspy
