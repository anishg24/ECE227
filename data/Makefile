SNAP_URL = https://snap.stanford.edu/data

SOCIAL_DATA_FILE = twitter_combined.txt.gz
COLLAB_DATA_FILE = ca-GrQc.txt.gz
COMM_DATA_FILE = email-Enron.txt.gz

SOCIAL_DATA = social.txt
COLLAB_DATA = collab.txt
COMM_DATA = comm.txt

ALL_DATA = $(SOCIAL_DATA) $(COLLAB_DATA) $(COMM_DATA)

define data_template
$(basename $(1)) $(1):
	@echo "generating $(1)..."
	@curl -L '$(SNAP_URL)/$(2)' | gunzip > $(1)
endef

help: ## Prints help for targets with comments
	@cat $(MAKEFILE_LIST) | grep -E '^[a-zA-Z_-]+:.*?## .*$$' | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

all: $(ALL_DATA) ## Generate all of the data files

$(eval $(call data_template,$(SOCIAL_DATA),$(SOCIAL_DATA_FILE)))
$(eval $(call data_template,$(COLLAB_DATA),$(COLLAB_DATA_FILE)))
$(eval $(call data_template,$(COMM_DATA),$(COMM_DATA_FILE)))

clean:
	rm -rf $(ALL_DATA)
