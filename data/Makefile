SNAP_URL = https://snap.stanford.edu/data

SOCIAL_DATA_FILE = twitter_combined.txt.gz
COLLAB_DATA_FILE = ca-GrQc.txt.gz
COMM_DATA_FILE = email-Enron.txt.gz

SOCIAL_DATA = social.txt
COLLAB_DATA = collab.txt
COMM_DATA = comm.txt
RANDOM_DATA = random.txt
SCALE_FREE_DATA = scale_free.txt

ALL_DATA = $(SOCIAL_DATA) $(COLLAB_DATA) $(COMM_DATA) $(RANDOM_DATA) $(SCALE_FREE_DATA)

define data_template
$(basename $(1)) $(1): remap.awk
	@echo "generating $(1)..."
	@curl -L '$(SNAP_URL)/$(2)' | gunzip | awk -f remap.awk > $(1)
endef

help: ## Prints help for targets with comments
	@cat $(MAKEFILE_LIST) | grep -E '^[a-zA-Z_-]+:.*?## .*$$' | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

all: $(ALL_DATA) ## Generate all of the data files

$(eval $(call data_template,$(SOCIAL_DATA),$(SOCIAL_DATA_FILE)))
$(eval $(call data_template,$(COLLAB_DATA),$(COLLAB_DATA_FILE)))
$(eval $(call data_template,$(COMM_DATA),$(COMM_DATA_FILE)))

$(RANDOM_DATA) $(SCALE_FREE_DATA): ./random_graphs.py
	uv run python random_graphs.py
	# Don't need to remap
	# gawk -i inplace -f remap.awk $(RANDOM_DATA) $(SCALE_FREE_DATA)

clean:
	rm -rf $(ALL_DATA)
