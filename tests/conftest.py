import os

# Specs name local checkpoints as $VLALAGBENCH_CHECKPOINTS/<name>; tests that
# resolve them without a real checkpoint need the variable set.
os.environ.setdefault("VLALAGBENCH_CHECKPOINTS", "/nonexistent/checkpoints")
