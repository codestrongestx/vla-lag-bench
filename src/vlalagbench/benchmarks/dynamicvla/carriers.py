"""Invisible carriers that move a DOM object without acting on it directly.

An environment lists its carriers under ``dynabench_carriers``. A carrier is a
collider-only rigid body (no rendered geometry) driven by the planar driver;
the free object rests on it and moves only through contact. Carriers ignore
gravity so they press on the table with just the carried object's weight:
table friction then cannot stall the driver's velocity writes, while the
carrier's own high-friction material still grips what it carries.
"""
from functools import wraps

# Keep the serialized key used by checksum-pinned benchmark scenes.
METADATA_KEY = "dynabench_carriers"


def install_carriers(upstream):
    """Honor carrier metadata without changing other environments."""
    original = upstream._get_env_cfg
    if getattr(original, "_vlalagbench_carriers", False):
        return

    @wraps(original)
    def configured(source, *args, **kwargs):
        cfg = original(source, *args, **kwargs)
        for name in source.get(METADATA_KEY, ()):
            asset = getattr(cfg.scene, name, None)
            if asset is None or name == "object":
                raise ValueError(f"carrier {name!r} must be a non-target scene rigid object")
            asset.spawn.rigid_props.disable_gravity = True
        return cfg

    configured._vlalagbench_carriers = True
    upstream._get_env_cfg = configured
