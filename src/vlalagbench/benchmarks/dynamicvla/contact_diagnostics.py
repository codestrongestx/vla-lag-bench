"""Opt-in, filtered box contact measurements for instrumented DOM scenes."""
from functools import wraps

CONTACT_LINKS = [*(f'panda_link{i}' for i in range(1,8)), 'panda_hand', 'panda_leftfinger', 'panda_rightfinger']
# Preserve the scene and trace identifiers used by recorded benchmark runs.
SENSOR_NAME = 'dynabench_box_contacts'


def install_contact_diagnostics(upstream):
    """Honor scene metadata without changing uninstrumented environments."""
    original=upstream._get_env_cfg
    if getattr(original,'_vlalagbench_contacts',False):
        return

    @wraps(original)
    def configured(source,*args,**kwargs):
        cfg=original(source,*args,**kwargs)
        if source.get('dynabench_contact_diagnostics') is True:
            from isaaclab.sensors import ContactSensorCfg
            if cfg.decimation != 1:
                raise ValueError('Contact diagnostics require one physics step per recorded control step')
            cfg.scene.container.spawn.activate_contact_sensors=True
            setattr(cfg.scene,SENSOR_NAME,ContactSensorCfg(
                prim_path='{ENV_REGEX_NS}/Container',update_period=0.,history_length=1,
                filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/'+link for link in CONTACT_LINKS]))
        return cfg
    configured._vlalagbench_contacts=True
    upstream._get_env_cfg=configured
