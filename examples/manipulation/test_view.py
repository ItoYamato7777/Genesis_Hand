import genesis as gs
gs.init(backend=gs.cuda)

scene = gs.Scene(
    show_viewer=True,
)
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(
        pos = (0.0, 0.0, 0.5),
        file="xml/shadow_hand/left_hand.xml",
    ),
    # vis_mode="collision",
)

scene.build()

for i in range(1000):
    scene.step()