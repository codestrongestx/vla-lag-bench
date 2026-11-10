import mujoco
import pytest

from vlalagbench.simulators.mujoco.dynamics import add_slide_joints, slide_joint_names

XML = """<mujoco><worldbody>
<body name="cabinet_main" pos="0.1 -0.2 0.9" quat="0 0 0 1"><geom type="box" size="0.1 0.1 0.1" mass="5"/>
  <body name="drawer"><joint name="drawer_slide" type="slide" axis="0 1 0"/><geom type="box" size="0.05 0.05 0.05" mass="1"/></body>
</body>
<body name="bowl_main" pos="0 0 1.2"><freejoint/><geom type="sphere" size="0.03" mass="0.1"/></body>
</worldbody></mujoco>"""


def test_adds_planar_slides_once_before_child_joints():
    xml = add_slide_joints(XML, ["cabinet_main"])
    assert add_slide_joints(xml, ["cabinet_main"]) == xml
    model = mujoco.MjModel.from_xml_string(xml)
    names = [model.joint(index).name for index in range(model.njnt)]
    assert names[:3] == [*slide_joint_names("cabinet_main"), "drawer_slide"]
    assert model.nq == 3 + 7


def test_rejects_bodies_with_other_joints():
    with pytest.raises(ValueError, match="must be jointless"):
        add_slide_joints(XML, ["bowl_main"])
