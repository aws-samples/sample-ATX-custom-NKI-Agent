SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

.PHONY: help prep lint security update-deps deploy destroy smoke-live check-network-drift clean clean-deep

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[1m%-14s\033[0m %s\n", $$1, $$2}'

prep: ## Install/sync all dependencies and run the test suite
	bash cicd/prep.sh

lint: ## Lint all Python projects with ruff check
	bash cicd/lint.sh

security: ## Scan dependencies for known vulnerabilities (pip-audit fatal, npm audit + ASH advisory)
	bash cicd/security-check.sh

update-deps: ## Upgrade dependencies (uv + npm) across the repo, then re-scan
	bash cicd/update-deps.sh

deploy: ## Deploy AgentCore + reward server to AWS (high-risk, confirmation-gated)
	bash cicd/deploy.sh

destroy: ## Tear down everything deploy created (destructive, confirmation-gated)
	bash cicd/destroy.sh

smoke-live: ## Exercise the real compile-verify-profile loop against the live deployment: real Bedrock + Trainium calls, no infra changes (confirmation-gated; SKIP_MIGRATION=1 for the fast smoke only)
	bash cicd/smoke-live.sh $(if $(SKIP_MIGRATION),--skip-migration,)

check-network-drift: ## Detect drift between the AgentCore runtime's live network config and what reward_server_cdk expects (read-only, no infra changes)
	bash cicd/check-agentcore-network-drift.sh

clean: ## Remove local caches and build artifacts (non-destructive)
	bash cicd/clean.sh

clean-deep: ## Also remove .venv/node_modules (re-run `make prep` afterward)
	bash cicd/clean.sh --deep
