.PHONY: data clean benchmark

NUM_SEED?=10
PROB?=0.05
N_TIMESTEP?=100

GRAPHS = social comm collab random scale_free
ALGORITHMS = celf genetic degree

define benchmark_template
$(1): main.py data
	@echo "benchmarking $(1) algorithm..."
	uv run main.py -k $(NUM_SEED) -N $(N_TIMESTEP) -p $(PROB) --graph_type social --seed_alg $(1)
	uv run main.py -k $(NUM_SEED) -N $(N_TIMESTEP) -p $(PROB) --graph_type comm --seed_alg $(1)
	uv run main.py -k $(NUM_SEED) -N $(N_TIMESTEP) -p $(PROB) --graph_type collab --seed_alg $(1)
	uv run main.py -k $(NUM_SEED) -N $(N_TIMESTEP) -p $(PROB) --graph_type random --seed_alg $(1)
	uv run main.py -k $(NUM_SEED) -N $(N_TIMESTEP) -p $(PROB) --graph_type scale_free --seed_alg $(1)
endef

benchmark: $(ALGORITHMS)

$(foreach alg,$(ALGORITHMS),$(eval $(call benchmark_template,$(alg))))

data:
	$(MAKE) -C data all

clean:
	$(MAKE) -C data clean
