# Lane strip editor

Attach **Lane Editor** (Add Component > Digital Twin > Lane Strip Editor) to an empty GameObject. Each strip represents **one traffic lane**, with its own width, direction, and sublanes. Create several adjacent lane strips to describe a road with several lanes.

Assign **Surface Root** to the existing road/model, such as `asphalt` or `yenphu`, and include its layer in **Surface Layers**. Click **Prepare source mesh colliders** if collision is missing. Drawing and snapped movement use collider raycasts. Missed clicks leave the graph unchanged; generated lane meshes are excluded from source picking.

## Line mode: traffic flow

Choose **Line** in the Inspector or runtime panel to display traffic centerlines and direction arrows. In edit-time Line mode, generated strip and junction meshes/colliders are not created, so their selection wireframes and collider gizmos cannot clutter the flow paths. Drawing still raycasts against the source road's colliders. Strip mode rebuilds the lane geometry; Play mode also builds the requested lane collision, including in Line mode.

**Node Pick Radius** sets the node click target size (26 by default, in Scene view GUI points / Game view pixels). It applies to selecting, dragging, and connecting nodes while keeping the visible markers small. Closely spaced nodes use the nearest target.

**Node Marker Size**, under **Lane display**, controls the visible cyan/yellow node spheres and runtime markers. It is a multiplier: `1` is the original size, `2` doubles it, and `0.5` halves it. Changing it leaves **Node Pick Radius** unchanged.

All node markers belonging to the selected lane turn yellow, including shared junctions. Other nodes remain cyan.

1. Click **Add lane strip**, then draw/append points on the source surface. Place at least two nodes. Point order defines Forward travel; Reverse travels against it; Both permits either direction. Escape finishes drawing, and Alt keeps Scene view navigation available.
2. Select an existing node to move it with Scene view handles. Exact coordinates are available through **Node position (world)**. Moving a shared junction updates every connected lane.
3. Select a node and choose **Branch lane from selected node** in the Inspector to start another lane there. An interior branch splits the original lane, preserving its settings. Draw toward the branch's destination.
4. Use **Pick merge destination in Scene view**, then click an existing node or another lane's centerline. A segment connection inserts a shared node and splits the receiving lane so routing can follow the connection. Target lane/segment controls also allow an exact merge position.

Use shared nodes for turns, branches, and merges. Lane paths that cross visually without a shared node do not connect, which also permits bridges and underpasses. Direction-compatible connections are available to the traffic controller; turn restrictions, priorities, signals, and vehicle conflict handling require controller logic.

## Strip mode: lane geometry

Choose **Strip** to display each lane as a ribbon of quads. Select a node/section, then choose **Move**, **Rotate**, or **Resize** in the Scene view editing tools.

- **Move** adjusts the node position. **Snap moved nodes to surface** makes handle movement follow source collision. Disable it for free elevation edits. Numeric world positions are applied directly.
- **Rotate** changes the selected section's pitch, yaw, and roll relative to its automatic tangent frame. Use it to bank a lane or adjust a cross-section at a bend. The Inspector also accepts rotation values.
- **Resize** changes the selected section's width. The lane's base width multiplied by the section's width scale gives its width at that node. Adjacent sections interpolate, supporting tapered approaches and merges.

Positions belong to shared nodes. Rotation and width belong to each lane's section, so branches retain independent geometry. Add sections around curves and slope changes. Vertices are not individually draped onto the source mesh. Avoid near-180-degree bends within a strip. Scene handle changes support Undo.

**Strip Material** and **Strip Color** control the filled mesh; **Lane Material** controls markings. Assign material assets for player builds so their shaders are included. **Generate Lane Colliders** controls generated collision.

**Draw Above Surface** is enabled by default. Strips and markings use an unlit overlay so uneven asphalt cannot cover them; markings draw after the strip fill. Overlay copies the assigned material's base color, albedo texture, and texture scale/offset without changing the material asset. It also shows through other scene geometry, including vehicles. Disable this setting to use the original materials and normal depth testing. Node positions and generated collision are unchanged. The overlay shader is included through `Resources` for player builds.

## Sublanes and runtime data

Set **Sublane count** on each lane. A 3.2 m lane with four sublanes has nominal 0.8 m occupancy slots. A vehicle can occupy multiple slots using its width and continuous lateral position, allowing narrower motorcycles to share the lane alongside larger vehicles. All slots inherit the strip's direction. Occupancy checks cover one lane strip; vehicle behavior must coordinate movement between adjacent lanes.

- `LaneStrips`, `Nodes`, and `FindNode(id)` expose the graph. Node positions are stored in the editor object's local space.
- `GetSublanePaths(lane)` returns world-space slot paths, widths, and direction; points retain authoring order.
- `GetConnections()` returns directed junction connections using lane IDs, sublane indices, and sampled routing curves.
- `GetOccupiedSublanes(lane, section, coordinate, width)` returns slots touched by the footprint, or an empty result when it cannot fit. Coordinate `0` is the first slot center; `0.5` lies between the first two centers.
- `CanOccupy(..., forward)` also checks travel direction. It does not check other vehicles.
- `MoveNode(id, worldPosition)` moves shared nodes; `SetEditMode(mode)` changes display/editing mode. Call `RebuildVisuals()` after directly changing data.

Vehicle spawning, steering, reservations, gap selection, and Python service integration are not supplied by this editor. Connection curves guide routing; they do not guarantee vehicle clearance.

## Play mode and saving

The runtime panel includes Line/Strip switching, drawing, branching, and merging. Strip mode provides lane width/sublane sliders and selected-node width/pitch/yaw/roll sliders. Exact numeric position, rotation, and width fields are in the Inspector. In Game view, select a node, choose **Move selected node: click destination**, and click the source surface. Escape cancels. Use **Merge lane end: click target lane** to connect an endpoint. **Show lanes** or **L** toggles gameplay rendering; generated collision remains active.

Save the scene after edit-mode changes. **Save JSON before stopping play** to retain runtime edits, then load it after stopping and save the scene. Files use `Application.persistentDataPath`; use distinct filenames for separate editor objects.

Version 3 stores the lane graph, per-section width/rotation, and edit mode. Legacy v1 paths migrate to lane strips. Version 2 road strips migrate into separate parallel lane strips, preserving widths and directions. Shared junction nodes are paired by physical proximity, with relative lateral order breaking ties. Review migrated junctions and save the scene. Generated children are rebuildable display/collision caches; use authoring controls instead.

## Traffic light stop lines

Choose **Traffic lights** in the Inspector, Scene overlay, or runtime panel. Click
**Draw traffic light line**, then click two points on the source surface to place
a red line. Use **Prepare source mesh colliders** if surface clicks do not hit.
Escape cancels an unfinished line without saving it. Alt keeps Scene navigation available.

Select a line in the list or click its midpoint cube in Scene view. Drag either
endpoint handle, or enter exact world coordinates in the Inspector. The surface
snapping toggle also applies to these handles. In Game view, select a line and
use **Move start/end: click surface**. Lines can be renamed and deleted; Scene and
Inspector changes support Undo/Redo.

Lines can be placed independently of lane nodes, including across multiple lanes.
`TrafficLights` exposes stable IDs, names, and local-space `start`/`end` endpoints.
The red marking has no collider. Simulation senses lines crossing a vehicle's
committed path ahead, including bends and route joins, and supplies the nearest
reading to the behaviour model. Vehicles brake to a 0.5 m front-bumper buffer.
Select a line to configure **Red duration (s)**, **Green duration (s)**, and
**Cycle offset (s)** in the Inspector. New lines default to 30 seconds red and
30 seconds green. The cycle repeats red then green; a positive offset advances
it. For two opposing approaches with 30/30 timings, offsets 0 and 30 start them
in opposite phases. The runtime panel also offers timing sliders.

The simulation clock drives timing, so pause freezes the cycle and Reset restarts
it. Physics substeps use the current timed aspect; vehicles resume on green.
The line changes red/green to show its current aspect. Edit mode previews time 0.
Green duration 0 holds red; red duration 0 with a positive green duration holds
green; both zero hold red, including legacy records deserialized without timings.
Timing fields save with the scene and JSON. Durations must be finite and nonnegative;
offsets may be positive or negative. The configured red-light passing probability
still applies during red phases.

Save the scene or use the existing JSON Save/Load controls to retain lines.
The optional `trafficLights` array extends version 3 files; older files load with
an empty list. Display offsets are not stored in the endpoints.

## Spawn and dead end markings

Select a node and enable **Vehicle spawn point** or **Dead end / out of view** in the Inspector or runtime panel. The Scene view overlay also provides buttons to add and remove each mark. Dead ends display a red cross and `[DEAD END]` label; spawn points display green. A boundary shared by incoming and outgoing traffic may have both marks.

Mark dead ends at the last visible point where vehicles leave the camera view. Endpoint replay uses these explicit marks to distinguish valid exits from tracking failures away from an exit. Marking a node records authoring metadata; vehicle removal and replay behavior are supplied by Simulation.

`LaneNode.isDeadEnd` and `deadEndName` are stored alongside `isSpawnPoint` and `spawnPointName` in the existing version 3 JSON and Unity scene data. Existing files load with no dead end marks, so review the camera boundaries and mark exits before using endpoint replay.

