#if LEGACY_SUBLANE_RUNTIME // Old Unity simulation map; unused since the sublane system was removed.
using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>Immutable, directed world-space map compiled from the lane editor and authored controls.</summary>
public sealed class TrafficNetwork
{
    public static readonly string[] VehicleKinds = { "car", "motorcycle", "bus", "truck", "van" };
    public readonly Dictionary<string, TrafficSegment> segments = new Dictionary<string, TrafficSegment>(StringComparer.Ordinal);
    public readonly List<TrafficMovement> movements = new List<TrafficMovement>();
    public readonly List<TrafficConflictZone> conflictZones = new List<TrafficConflictZone>();
    public readonly List<TrafficArea> drivableAreas = new List<TrafficArea>();
    public readonly List<JunctionController> controllers = new List<JunctionController>();
    public float MaximumVehicleHeight { get; private set; }

    public static string EdgeId(string laneId, bool forward) => laneId + (forward ? ":forward" : ":reverse");
    public static string MovementId(string fromEdge, string toEdge, string junctionId) => "movement:" + fromEdge + ">" + toEdge + "@" + junctionId;
    public static string ConnectorId(string fromEdge, string toEdge, string junctionId, int fromSublane, int toSublane)
        => "connector:" + fromEdge + ":" + fromSublane + ">" + toEdge + ":" + toSublane + "@" + junctionId;

    /// <summary>Authoring catalog, including candidates omitted by Build until they have explicit bindings/envelopes.</summary>
    public static Dictionary<string, object> GeneratedIdentityCatalog(LaneEditor editor)
    {
        if (editor == null) throw new ArgumentNullException(nameof(editor));
        List<object> edges = new List<object>(), connectors = new List<object>();
        Dictionary<string, LaneEditor.LaneStrip> lanes = new Dictionary<string, LaneEditor.LaneStrip>();
        foreach (LaneEditor.LaneStrip lane in editor.LaneStrips)
        {
            if (lane == null || string.IsNullOrEmpty(lane.id) || lanes.ContainsKey(lane.id)) throw new InvalidOperationException("Lane strips need unique stable IDs before exporting candidate identities.");
            lanes.Add(lane.id, lane);
            if (lane.sections == null || lane.sections.Count < 2) continue;
            foreach (bool forward in new[] { true, false })
            {
                if (forward && lane.direction == LaneEditor.LaneDirection.Reverse || !forward && lane.direction == LaneEditor.LaneDirection.Forward) continue;
                edges.Add(TrafficJson.Obj("id", EdgeId(lane.id, forward), "lane_id", lane.id, "display_name", lane.name,
                    "direction", forward ? "forward" : "reverse", "start_node_id", lane.sections[forward ? 0 : lane.sections.Count - 1].nodeId,
                    "end_node_id", lane.sections[forward ? lane.sections.Count - 1 : 0].nodeId, "sublane_count", lane.sublaneCount));
            }
        }
        foreach (LaneEditor.LaneConnection connection in editor.GetConnections())
        {
            LaneEditor.LaneStrip source = lanes[connection.fromLaneId], destination = lanes[connection.toLaneId];
            string from = EdgeId(source.id, source.sections[source.sections.Count - 1].nodeId == connection.junctionId);
            string to = EdgeId(destination.id, destination.sections[0].nodeId == connection.junctionId);
            bool nondegenerate = false;
            for (int i = 1; i < connection.worldPoints.Length; i++) if (HorizontalDistance(connection.worldPoints[0], connection.worldPoints[i]) > 0.0001f) nondegenerate = true;
            connectors.Add(TrafficJson.Obj("id", ConnectorId(from, to, connection.junctionId, connection.fromSublane, connection.toSublane),
                "movement_id", MovementId(from, to, connection.junctionId), "junction_node_id", connection.junctionId,
                "from_edge_id", from, "to_edge_id", to, "from_sublane", connection.fromSublane, "to_sublane", connection.toSublane,
                "has_horizontal_geometry", nondegenerate));
        }
        return TrafficJson.Obj("lane_edges", edges, "connector_candidates", connectors);
    }

    public static TrafficNetwork Build(LaneEditor editor, JunctionController[] authoredControllers, float defaultSpeedLimit, bool allowUncontrolled = false, float maxVehicleHeight = 4.5f, bool allowDirectLaneJoins = false)
    {
        if (editor == null) throw new InvalidOperationException("Assign a Lane Editor before compiling the traffic network.");
        if (!Finite(defaultSpeedLimit) || defaultSpeedLimit <= 0) throw new InvalidOperationException("Speed limit must be a positive finite value.");
        if (!Finite(maxVehicleHeight) || maxVehicleHeight <= 0) throw new InvalidOperationException("Maximum vehicle height must be a positive finite value.");
        TrafficNetwork result = new TrafficNetwork { MaximumVehicleHeight = maxVehicleHeight };
        Dictionary<string, LaneEditor.LaneStrip> lanes = new Dictionary<string, LaneEditor.LaneStrip>(StringComparer.Ordinal);
        Dictionary<string, List<Vector3>> junctionBoundaries = new Dictionary<string, List<Vector3>>();
        HashSet<string> nodeIds = new HashSet<string>(StringComparer.Ordinal);
        foreach (LaneEditor.LaneNode node in editor.Nodes)
            if (node == null || string.IsNullOrEmpty(node.id) || !nodeIds.Add(node.id) || !Finite(node.position))
                throw new InvalidOperationException("Lane nodes need unique stable IDs and finite positions.");
        foreach (LaneEditor.LaneStrip lane in editor.LaneStrips)
        {
            if (lane == null || string.IsNullOrEmpty(lane.id) || lanes.ContainsKey(lane.id)) throw new InvalidOperationException("Lane strips need unique stable IDs.");
            lanes.Add(lane.id, lane);
            if (lane.sections == null || lane.sections.Count < 2) continue;
            if (!Finite(lane.width) || lane.width <= 0 || lane.sublaneCount < 1 || !Enum.IsDefined(typeof(LaneEditor.LaneDirection), lane.direction))
                throw new InvalidOperationException("Invalid width, sublane count or direction on lane " + lane.id);
            HashSet<string> visited = new HashSet<string>();
            foreach (LaneEditor.Section section in lane.sections)
                if (section == null || !nodeIds.Contains(section.nodeId) || !visited.Add(section.nodeId) || !Finite(section.widthScale) || section.widthScale <= 0 || !Finite(section.rotation))
                    throw new InvalidOperationException("Invalid section on lane " + lane.id);
            RoadMeshGeometry.Frame[] frames = RoadMeshGeometry.Frames(editor, lane);
            List<int> retained = new List<int>();
            for (int i = 0; i < frames.Length; i++)
                if (retained.Count == 0 || HorizontalDistance(frames[retained[retained.Count - 1]].center, frames[i].center) > 0.0001f) retained.Add(i);
            if (retained.Count < 2) throw new InvalidOperationException("Lane " + lane.id + " has no horizontal travel distance. Purely vertical roads are unsupported.");
            // Junction surfaces match the editor's authored cross sections, not arbitrary turn radii.
            for (int i = 0; i < frames.Length; i++)
            {
                string nodeId = lane.sections[i].nodeId;
                if (!junctionBoundaries.TryGetValue(nodeId, out List<Vector3> boundaries)) junctionBoundaries[nodeId] = boundaries = new List<Vector3>();
                boundaries.Add(frames[i].left); boundaries.Add(frames[i].right);
            }
            for (int i = 1; i < retained.Count; i++)
            {
                RoadMeshGeometry.Frame a = frames[retained[i - 1]], b = frames[retained[i]];
                result.drivableAreas.Add(new TrafficArea("road:" + lane.id + ":" + i, new[] { a.left, a.right, b.right, b.left }));
            }
            if (lane.direction != LaneEditor.LaneDirection.Reverse) result.AddEdge(lane, frames, retained, true, defaultSpeedLimit);
            if (lane.direction != LaneEditor.LaneDirection.Forward) result.AddEdge(lane, frames, retained, false, defaultSpeedLimit);
        }
        foreach (KeyValuePair<string, List<Vector3>> junction in junctionBoundaries)
        {
            Vector3[] hull = ConvexHull(junction.Value);
            if (hull.Length >= 3) result.drivableAreas.Add(new TrafficArea("junction:" + junction.Key, hull));
        }
        Dictionary<string, JunctionController.MovementBinding> bindings = new Dictionary<string, JunctionController.MovementBinding>(StringComparer.Ordinal);
        Dictionary<string, JunctionController> owners = new Dictionary<string, JunctionController>(StringComparer.Ordinal);
        Dictionary<string, JunctionController.ConnectorEnvelope> envelopes = new Dictionary<string, JunctionController.ConnectorEnvelope>(StringComparer.Ordinal);
        Dictionary<string, JunctionController> envelopeOwners = new Dictionary<string, JunctionController>(StringComparer.Ordinal);
        HashSet<string> controllerIds = new HashSet<string>(StringComparer.Ordinal);
        foreach (JunctionController controller in authoredControllers ?? new JunctionController[0])
        {
            if (controller == null) continue;
            if (string.IsNullOrEmpty(controller.id) || !controllerIds.Add(controller.id)) throw new InvalidOperationException("Junction controllers need unique stable IDs.");
            result.controllers.Add(controller);
            foreach (JunctionController.MovementBinding binding in controller.movements)
            {
                if (binding == null || string.IsNullOrEmpty(binding.movementId) || bindings.ContainsKey(binding.movementId))
                    throw new InvalidOperationException("Every authored movement must have exactly one controller binding.");
                bindings.Add(binding.movementId, binding); owners.Add(binding.movementId, controller);
            }
            foreach (JunctionController.ConnectorEnvelope envelope in controller.connectorEnvelopes)
            {
                if (envelope == null) throw new InvalidOperationException("Empty connector envelope slot in controller " + controller.id);
                string key = !string.IsNullOrEmpty(envelope.connectorId) ? envelope.connectorId : envelope.movementId;
                if (string.IsNullOrEmpty(key) || envelopes.ContainsKey(key)) throw new InvalidOperationException("Connector envelopes need a unique connector ID or movement ID.");
                envelopes.Add(key, envelope); envelopeOwners.Add(key, controller);
            }
        }
        Dictionary<string, TrafficMovement> byMovement = new Dictionary<string, TrafficMovement>(StringComparer.Ordinal);
        HashSet<string> omittedMovements = new HashSet<string>(StringComparer.Ordinal);
        foreach (LaneEditor.LaneConnection connection in editor.GetConnections())
        {
            LaneEditor.LaneStrip from = lanes[connection.fromLaneId], to = lanes[connection.toLaneId];
            bool forwardFrom = from.sections[from.sections.Count - 1].nodeId == connection.junctionId;
            bool forwardTo = to.sections[0].nodeId == connection.junctionId;
            string approachId = EdgeId(from.id, forwardFrom), departureId = EdgeId(to.id, forwardTo);
            string movementId = MovementId(approachId, departureId, connection.junctionId);
            string connectorId = ConnectorId(approachId, departureId, connection.junctionId, connection.fromSublane, connection.toSublane);
            bindings.TryGetValue(movementId, out JunctionController.MovementBinding binding);
            if (binding == null && !allowUncontrolled) { omittedMovements.Add(movementId); continue; }
            List<Vector3> curve = new List<Vector3>();
            foreach (Vector3 point in connection.worldPoints)
            {
                if (!Finite(point)) throw new InvalidOperationException("Non-finite connector geometry: " + connectorId);
                if (curve.Count == 0 || HorizontalDistance(curve[curve.Count - 1], point) > 0.0001f) curve.Add(point);
            }
            if (curve.Count < 2)
            {
                Debug.LogWarning("Omitted zero-length generated connector " + connectorId + ". Author distinct junction entry/exit geometry to route through it.", editor);
                continue;
            }
            string envelopeKey = envelopes.ContainsKey(connectorId) ? connectorId : movementId;
            if (!envelopes.TryGetValue(envelopeKey, out JunctionController.ConnectorEnvelope envelope))
            {
                if (binding != null) throw new InvalidOperationException("Author a Connector Envelope for " + connectorId + " (or its movement ID). GetConnections alone does not establish vehicle clearance.");
                omittedMovements.Add(movementId); continue;
            }
            TrafficSegment connector = new TrafficSegment
            {
                id = connectorId, type = "connector", laneId = null, forward = true, startNodeId = connection.junctionId, endNodeId = connection.junctionId,
                fromEdgeId = approachId, toEdgeId = departureId, fromSublane = connection.fromSublane, toSublane = connection.toSublane,
                movementId = movementId, points = curve.ToArray(), speedLimit = defaultSpeedLimit, sublaneCount = 0,
                sourceEdge = result.segments[approachId], destinationEdge = result.segments[departureId],
                allowedKinds = binding != null && binding.allowedKinds != null && binding.allowedKinds.Length > 0 ? binding.allowedKinds : (string[])VehicleKinds.Clone()
            };
            connector.PrepareConnector(envelope.widthProfileM);
            if (envelope.localPolygon != null && envelope.localPolygon.Length > 0)
            {
                JunctionController owner = envelopeOwners[envelopeKey];
                Vector3[] polygon = TransformPoints(owner.transform, envelope.localPolygon);
                ValidatePolygon(polygon, "connector envelope " + connectorId);
                TrafficArea area = new TrafficArea("envelope:" + connectorId, polygon);
                result.drivableAreas.Add(area);
                foreach (Vector3 corner in connector.RibbonSamples())
                    if (!area.ContainsHorizontal(corner) || corner.y < area.minY - 0.25f || corner.y > area.maxY + 0.25f)
                        throw new InvalidOperationException("Connector ribbon exceeds authored envelope: " + connectorId);
            }
            else
                foreach (Vector3 corner in connector.RibbonSamples())
                    if (!result.ContainsSurface(corner)) throw new InvalidOperationException("Connector envelope leaves the authored road/junction surface: " + connectorId + ". Author its local polygon or repair the road geometry.");
            result.segments.Add(connectorId, connector);
            if (!byMovement.TryGetValue(movementId, out TrafficMovement movement))
            {
                movement = new TrafficMovement
                {
                    id = movementId, approachEdgeId = approachId, departureEdgeId = departureId,
                    controllerId = binding != null ? owners[movementId].id : null,
                    signalGroupId = binding != null ? NullIfEmpty(binding.signalGroupId) : null,
                    stopLineId = binding != null ? NullIfEmpty(binding.stopLineId) : null,
                    priorityRule = binding != null ? NullIfEmpty(binding.uncontrolledPriorityRule) : "yield_to_conflicting_traffic",
                    allowedKinds = connector.allowedKinds,
                    conflictZoneIds = binding != null ? binding.conflictZoneIds ?? new string[0] : new string[0]
                };
                // An uncontrolled movement without an authored stop line enters at the approach endpoint.
                // CompileBindings replaces this with the authored approach-edge station when a line exists.
                movement.stopS = result.segments[approachId].length;
                byMovement.Add(movementId, movement); result.movements.Add(movement);
            }
            movement.connectorIds.Add(connectorId);
        }
        if (!allowDirectLaneJoins)
            foreach (string omitted in omittedMovements)
                Debug.LogWarning("Omitted unconfigured movement " + omitted + ". Bind it explicitly and author a drivable connector envelope before using it in a route.", editor);
        foreach (string boundId in bindings.Keys)
            if (!byMovement.ContainsKey(boundId)) throw new InvalidOperationException("Authored movement has no usable nondegenerate connector: " + boundId);
        // Resolve explicit stop-line and zone bindings; phase durations are checked once decision_dt is known.
        foreach (JunctionController controller in result.controllers) controller.CompileBindings(result);
        foreach (TrafficMovement movement in result.movements)
        {
            bool signalized = !string.IsNullOrEmpty(movement.signalGroupId);
            if (signalized && (!string.IsNullOrEmpty(movement.priorityRule) || string.IsNullOrEmpty(movement.stopLineId)))
                throw new InvalidOperationException("A signalized movement needs one stop line and cannot also have an uncontrolled priority rule: " + movement.id);
            if (!signalized && string.IsNullOrEmpty(movement.priorityRule)) throw new InvalidOperationException("Unsignalized movement needs an explicit priority rule: " + movement.id);
            foreach (string kind in movement.allowedKinds)
                if (Array.IndexOf(VehicleKinds, kind) < 0) throw new InvalidOperationException("Unsupported vehicle kind on movement " + movement.id);
        }
        result.ValidateConflictCoverage();
        return result;
    }

    private void AddEdge(LaneEditor.LaneStrip lane, RoadMeshGeometry.Frame[] frames, List<int> retained, bool forward, float speedLimit)
    {
        TrafficSegment edge = new TrafficSegment
        {
            id = EdgeId(lane.id, forward), type = "lane_edge", laneId = lane.id, forward = forward,
            startNodeId = lane.sections[forward ? 0 : lane.sections.Count - 1].nodeId,
            endNodeId = lane.sections[forward ? lane.sections.Count - 1 : 0].nodeId,
            sublaneCount = lane.sublaneCount, speedLimit = speedLimit, allowedKinds = (string[])VehicleKinds.Clone(),
            points = new Vector3[retained.Count], left = new Vector3[retained.Count], right = new Vector3[retained.Count],
            up = new Vector3[retained.Count], widths = new float[retained.Count]
        };
        for (int i = 0; i < retained.Count; i++)
        {
            RoadMeshGeometry.Frame frame = frames[retained[forward ? i : retained.Count - 1 - i]];
            edge.points[i] = frame.center; edge.left[i] = forward ? frame.left : frame.right; edge.right[i] = forward ? frame.right : frame.left;
            edge.up[i] = frame.up; edge.widths[i] = HorizontalDistance(frame.left, frame.right);
            if (edge.widths[i] <= 0.001f) throw new InvalidOperationException("Lane cross section has zero horizontal width: " + edge.id);
        }
        edge.Prepare(); segments.Add(edge.id, edge);
    }

    public bool ContainsSurface(Vector3 point)
    {
        foreach (TrafficArea area in drivableAreas)
            if (point.y >= area.minY - 0.25f && point.y <= area.maxY + 0.25f && area.ContainsHorizontal(point)) return true;
        return false;
    }

    public TrafficMovement FindMovement(string id) => movements.Find(m => m.id == id);
    public TrafficConflictZone FindZone(string id) => conflictZones.Find(z => z.id == id);

    public bool MovementsConflict(string first, string second)
    {
        if (first == second) return false;
        foreach (TrafficConflictZone zone in conflictZones)
            foreach (string[] pair in zone.movementPairs)
                if ((pair[0] == first && pair[1] == second) || (pair[1] == first && pair[0] == second)) return true;
        return false;
    }

    private void ValidateConflictCoverage()
    {
        // An authored shared zone is required whenever protected paths' envelopes intersect at the same elevation.
        // The test uses the validated connector ribbons, not just common editor node IDs.
        for (int i = 0; i < movements.Count; i++)
            for (int j = i + 1; j < movements.Count; j++)
            {
                TrafficMovement a = movements[i], b = movements[j];
                List<TrafficArea.ConflictSample> intersections = new List<TrafficArea.ConflictSample>();
                foreach (string first in a.connectorIds)
                    foreach (string second in b.connectorIds)
                    {
                        TrafficSegment sa = segments[first], sb = segments[second];
                        // Following/diverging paths may share one origin cell. Different authoring
                        // cells on the same approach still require explicit conflicts when they cross.
                        if (a.approachEdgeId == b.approachEdgeId && sa.fromSublane == sb.fromSublane) continue;
                        intersections.AddRange(TrafficArea.RibbonConflictSamples(sa, sb, MaximumVehicleHeight));
                    }
                if (intersections.Count != 0 && !MovementsConflict(a.id, b.id))
                    throw new InvalidOperationException("Crossing or merging movement envelopes need an authored height-aware conflict zone: " + a.id + " / " + b.id);
                foreach (TrafficArea.ConflictSample intersection in intersections)
                {
                    bool covered = false;
                    foreach (TrafficConflictZone zone in conflictZones)
                        if (zone.movementPairs.Exists(pair => pair[0] == a.id && pair[1] == b.id || pair[0] == b.id && pair[1] == a.id) &&
                            zone.ContainsHorizontal(intersection.point) && zone.minY <= intersection.minY + 0.001f && zone.maxY >= intersection.maxY - 0.001f)
                        { covered = true; break; }
                    if (!covered) throw new InvalidOperationException("Conflict zones must cover the crossing/merging envelope and its full configured vehicle height (" + MaximumVehicleHeight + " m): " + a.id + " / " + b.id);
                }
            }
    }

    public Dictionary<string, object> ToWire()
    {
        List<object> laneEdges = new List<object>(), connectors = new List<object>(), movementValues = new List<object>(), stopLines = new List<object>(), groups = new List<object>(), zones = new List<object>(), areas = new List<object>();
        foreach (TrafficSegment segment in segments.Values) { if (segment.type == "lane_edge") laneEdges.Add(segment.ToWire()); else connectors.Add(segment.ToWire()); }
        foreach (TrafficMovement movement in movements) movementValues.Add(movement.ToWire());
        foreach (JunctionController controller in controllers) { stopLines.AddRange(controller.StopLinesToWire(this)); groups.AddRange(controller.SignalGroupsToWire(this)); }
        foreach (TrafficConflictZone zone in conflictZones) zones.Add(zone.ToWire());
        foreach (TrafficArea area in drivableAreas) areas.Add(area.ToWire());
        return TrafficJson.Obj("lane_edges", laneEdges, "connectors", connectors, "movements", movementValues, "stop_lines", stopLines, "signal_groups", groups, "conflict_zones", zones, "drivable_areas", areas);
    }

    internal static object[] Vectors(Vector3[] values)
    {
        object[] result = new object[values.Length]; for (int i = 0; i < values.Length; i++) result[i] = TrafficJson.Vec(values[i]); return result;
    }
    internal static Vector3[] TransformPoints(Transform transform, Vector3[] points)
    {
        Vector3[] result = new Vector3[points.Length]; for (int i = 0; i < points.Length; i++) result[i] = transform.TransformPoint(points[i]); return result;
    }
    internal static string NullIfEmpty(string value) => string.IsNullOrWhiteSpace(value) ? null : value;
    internal static float HorizontalDistance(Vector3 a, Vector3 b) => new Vector2(a.x - b.x, a.z - b.z).magnitude;
    internal static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
    internal static bool Finite(Vector3 value) => Finite(value.x) && Finite(value.y) && Finite(value.z);
    internal static void ValidatePolygon(Vector3[] polygon, string name)
    {
        if (polygon == null || polygon.Length < 3) throw new InvalidOperationException(name + " needs at least three vertices.");
        float twiceArea = 0;
        for (int i = 0; i < polygon.Length; i++)
        {
            if (!Finite(polygon[i])) throw new InvalidOperationException(name + " has a non-finite vertex.");
            Vector3 next = polygon[(i + 1) % polygon.Length]; twiceArea += polygon[i].x * next.z - next.x * polygon[i].z;
        }
        if (Mathf.Abs(twiceArea) < 0.0001f) throw new InvalidOperationException(name + " has zero horizontal area.");
        for (int i = 0; i < polygon.Length; i++)
            for (int j = i + 1; j < polygon.Length; j++)
            {
                if (j == i + 1 || (i == 0 && j == polygon.Length - 1)) continue;
                if (TrafficArea.SegmentsCross(polygon[i], polygon[(i + 1) % polygon.Length], polygon[j], polygon[(j + 1) % polygon.Length]))
                    throw new InvalidOperationException(name + " has self-intersecting edges.");
            }
    }
    private static Vector3[] ConvexHull(List<Vector3> points)
    {
        List<Vector3> sorted = new List<Vector3>(points); sorted.Sort((a, b) => a.x == b.x ? a.z.CompareTo(b.z) : a.x.CompareTo(b.x));
        List<Vector3> unique = new List<Vector3>(); foreach (Vector3 p in sorted) if (unique.Count == 0 || HorizontalDistance(unique[unique.Count - 1], p) > 0.0001f) unique.Add(p);
        if (unique.Count < 3) return new Vector3[0];
        List<Vector3> hull = new List<Vector3>();
        foreach (Vector3 p in unique) { while (hull.Count >= 2 && TrafficArea.Cross(hull[hull.Count - 2], hull[hull.Count - 1], p) <= 0) hull.RemoveAt(hull.Count - 1); hull.Add(p); }
        int lower = hull.Count;
        for (int i = unique.Count - 2; i >= 0; i--) { Vector3 p = unique[i]; while (hull.Count > lower && TrafficArea.Cross(hull[hull.Count - 2], hull[hull.Count - 1], p) <= 0) hull.RemoveAt(hull.Count - 1); hull.Add(p); }
        hull.RemoveAt(hull.Count - 1); return hull.Count >= 3 ? hull.ToArray() : new Vector3[0];
    }
}

public sealed partial class TrafficSegment
{
    public string id, type, laneId, startNodeId, endNodeId, movementId, fromEdgeId, toEdgeId;
    public bool forward;
    public int sublaneCount, fromSublane, toSublane;
    public Vector3[] points, left, right, up;
    public float[] widths;
    public float length, speedLimit;
    public string[] allowedKinds;
    internal TrafficSegment sourceEdge, destinationEdge;
    private float[] stations;
    internal int GeometryRevision { get; private set; }
    public float MinY { get { float y = float.PositiveInfinity; foreach (Vector3 p in points) y = Mathf.Min(y, p.y); return y; } }
    public float MaxY { get { float y = float.NegativeInfinity; foreach (Vector3 p in points) y = Mathf.Max(y, p.y); return y; } }

    public void Prepare()
    {
        GeometryRevision++;
        surfaceGeometry = null;
        feasibilityCache.Clear();
        if (points == null || points.Length < 2) throw new InvalidOperationException("Segment needs at least two noncoincident samples: " + id);
        stations = new float[points.Length]; length = 0;
        for (int i = 1; i < points.Length; i++)
        {
            float distance = TrafficNetwork.HorizontalDistance(points[i - 1], points[i]);
            if (!TrafficNetwork.Finite(distance) || distance < 0.0001f) throw new InvalidOperationException("Degenerate horizontal geometry: " + id);
            length += distance; stations[i] = length;
        }
    }
    internal void PrepareConnector(float[] authoredWidths)
    {
        if (authoredWidths == null || (authoredWidths.Length != 1 && authoredWidths.Length != points.Length))
            throw new InvalidOperationException("Connector width profile needs one constant width or one width per retained curve sample: " + id);
        widths = new float[points.Length]; left = new Vector3[points.Length]; right = new Vector3[points.Length]; up = new Vector3[points.Length];
        Prepare();
        for (int i = 0; i < points.Length; i++)
        {
            widths[i] = authoredWidths[authoredWidths.Length == 1 ? 0 : i];
            if (!TrafficNetwork.Finite(widths[i]) || widths[i] <= 0) throw new InvalidOperationException("Connector widths must be positive: " + id);
            Vector3 tangent = Tangent(stations[i]); Vector3 side = new Vector3(tangent.z, 0, -tangent.x).normalized;
            left[i] = points[i] - side * widths[i] * 0.5f; right[i] = points[i] + side * widths[i] * 0.5f; up[i] = Vector3.up;
        }
    }
    private int Interval(float s, out float t)
    {
        s = Mathf.Clamp(s, 0, length); int index = Array.BinarySearch(stations, s);
        if (index < 0) index = ~index - 1; index = Mathf.Clamp(index, 0, stations.Length - 2);
        t = (s - stations[index]) / (stations[index + 1] - stations[index]); return index;
    }
    public Vector3 Sample(float s) { int i = Interval(s, out float t); return Vector3.Lerp(points[i], points[i + 1], t); }
    public Vector3 Tangent(float s) { int i = Interval(s, out float t); Vector3 delta = points[i + 1] - points[i]; delta.y = 0; return delta.normalized; }
    public float WidthAt(float s) { int i = Interval(s, out float t); return Mathf.Lerp(widths[i], widths[i + 1], t); }
    public Vector3 UpAt(float s) { int i = Interval(s, out float t); return Vector3.Lerp(up[i], up[i + 1], t).normalized; }
    public void Project(Vector3 point, out float s, out float lateral)
    {
        float distance = float.PositiveInfinity; s = 0; lateral = 0;
        for (int i = 1; i < points.Length; i++)
        {
            Vector3 delta = points[i] - points[i - 1]; delta.y = 0;
            Vector3 offset = point - points[i - 1]; offset.y = 0;
            float t = Mathf.Clamp01(Vector3.Dot(offset, delta) / delta.sqrMagnitude);
            Vector3 projection = Vector3.Lerp(points[i - 1], points[i], t);
            float candidate = TrafficNetwork.HorizontalDistance(point, projection);
            // Height disambiguates a returning/overpassing piece of the same polyline.
            candidate = candidate * candidate + (point.y - projection.y) * (point.y - projection.y);
            if (candidate >= distance) continue;
            distance = candidate; s = Mathf.Lerp(stations[i - 1], stations[i], t);
            Vector3 side = new Vector3(delta.z, 0, -delta.x).normalized; lateral = Vector3.Dot(point - projection, side);
        }
    }
    /// <summary>Tests the actual road ribbon rather than treating each polyline vertex as an endpoint.</summary>
    public bool ContainsSurfacePoint(Vector3 point, float startOverhang = 0, float endOverhang = 0, float verticalTolerance = 0.75f)
    {
        if (!TrafficNetwork.Finite(point) || !TrafficNetwork.Finite(startOverhang) || !TrafficNetwork.Finite(endOverhang) ||
            !TrafficNetwork.Finite(verticalTolerance) || startOverhang < 0 || endOverhang < 0 || verticalTolerance < 0 ||
            points == null || left == null || right == null || points.Length < 2 || left.Length != points.Length || right.Length != points.Length)
            return false;
        var surface = GetSurfaceGeometry();
        foreach (var triangle in surface.triangles)
            if (triangle.Contains(point, verticalTolerance)) return true;
        // Extend only mission endpoints, keeping the authored cross-section and physical side boundaries.
        // Scaling the full tangent by horizontal length preserves the surface's grade outside that endpoint.
        if (startOverhang > 0)
        {
            foreach (var triangle in GetEndpointTriangles(surface, true, startOverhang))
                if (triangle.Contains(point, verticalTolerance)) return true;
        }
        if (endOverhang > 0)
        {
            foreach (var triangle in GetEndpointTriangles(surface, false, endOverhang))
                if (triangle.Contains(point, verticalTolerance)) return true;
        }
        return false;
    }
    private static bool ContainsSurfaceQuad(Vector3 point, Vector3 startLeft, Vector3 startRight, Vector3 endLeft, Vector3 endRight, float verticalTolerance)
    {
        // Same diagonal as RoadMeshGeometry.Strip: a,c,b and b,c,d.
        return ContainsSurfaceTriangle(point, startLeft, endLeft, startRight, verticalTolerance) ||
            ContainsSurfaceTriangle(point, startRight, endLeft, endRight, verticalTolerance);
    }
    private static bool ContainsSurfaceTriangle(Vector3 point, Vector3 a, Vector3 b, Vector3 c, float verticalTolerance)
    {
        float abX = b.x - a.x, abZ = b.z - a.z, acX = c.x - a.x, acZ = c.z - a.z;
        float determinant = abX * acZ - abZ * acX;
        if (Mathf.Abs(determinant) < 0.00000001f) return false;
        float apX = point.x - a.x, apZ = point.z - a.z;
        float u = (apX * acZ - apZ * acX) / determinant, v = (abX * apZ - abZ * apX) / determinant;
        const float edgeTolerance = 0.000001f;
        if (u < -edgeTolerance || v < -edgeTolerance || u + v > 1 + edgeTolerance) return false;
        float surfaceY = a.y + u * (b.y - a.y) + v * (c.y - a.y);
        return Mathf.Abs(point.y - surfaceY) <= verticalTolerance;
    }
    public Vector3[] ReferencePath(int sublane, float vehicleWidth, float vehicleLength = 0)
    {
        if (!TrafficNetwork.Finite(vehicleWidth) || vehicleWidth <= 0) throw new InvalidOperationException("Vehicle width must be positive.");
        if (!TrafficNetwork.Finite(vehicleLength) || vehicleLength < 0) throw new InvalidOperationException("Vehicle length must be finite and nonnegative.");
        if (type == "lane_edge" && (sublane < 0 || sublane >= sublaneCount)) throw new InvalidOperationException("Invalid authoring sublane for " + id);
        float[] clearance = new float[points.Length];
        for (int i = 0; i < clearance.Length; i++) clearance[i] = vehicleWidth * 0.5f + 0.05f;
        if (vehicleLength > 0)
            for (int i = 1; i < points.Length - 1; i++)
            {
                Vector3 before = points[i] - points[i - 1], after = points[i + 1] - points[i]; before.y = after.y = 0;
                float cosine = Mathf.Clamp(Vector3.Dot(before.normalized, after.normalized), -1, 1);
                float sineHalf = Mathf.Sqrt((1 - cosine) * 0.5f), cosineHalf = Mathf.Sqrt((1 + cosine) * 0.5f);
                // A turning footprint projects its length across the road as well as its width.
                // Reserve another half-length projection so smooth steering can cut inside the
                // waypoint without placing the body on the physical road edge. Carry this room
                // across both adjoining spans; a sudden lateral shift at the node is not drivable.
                float bendClearance = vehicleLength * sineHalf + vehicleWidth * 0.5f * cosineHalf + 0.05f;
                for (int neighbour = i - 1; neighbour <= i + 1; neighbour++) clearance[neighbour] = Mathf.Max(clearance[neighbour], bendClearance);
            }
        Vector3[] path = new Vector3[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            if (widths[i] + 0.0001f < vehicleWidth + 0.1f) throw new InvalidOperationException("Vehicle is wider than the drivable envelope on " + id);
            float fraction = type == "lane_edge" ? (sublane + 0.5f) / sublaneCount : 0.5f;
            if (type == "lane_edge" && !forward) fraction = 1 - fraction;
            // The lane may have less tracking reserve than requested; its centre maximizes the
            // available clearance. Full rotated-footprint feasibility remains authoritative.
            float margin = Mathf.Min(0.5f, clearance[i] / widths[i]);
            fraction = Mathf.Clamp(fraction, margin, 1 - margin);
            path[i] = Vector3.Lerp(left[i], right[i], fraction);
        }
        if (type == "lane_edge" && vehicleLength > 0 && !ReferenceFits(path, vehicleLength, vehicleWidth))
        {
            // Width-only offsets can leave a long body's front or rear outside a
            // taper, even when its centre has enough clearance. Prefer the assigned
            // side, then try continuous offsets farther inward on the same lane.
            // Validate the same path that will be used by the runtime and connectors.
            for (int attempt = 1; attempt <= 4; attempt++)
            {
                var candidate = new Vector3[path.Length];
                float inward = attempt * 0.25f;
                for (int i = 0; i < path.Length; i++) candidate[i] = Vector3.Lerp(path[i], (left[i] + right[i]) * 0.5f, inward);
                if (ReferenceFits(candidate, vehicleLength, vehicleWidth)) { path = candidate; break; }
            }
        }
        if (type == "connector" && sourceEdge != null && destinationEdge != null)
        {
            Vector3[] approach = sourceEdge.ReferencePath(fromSublane, vehicleWidth, vehicleLength), departure = destinationEdge.ReferencePath(toSublane, vehicleWidth, vehicleLength);
            Vector3 startShift = approach[approach.Length - 1] - path[0], endShift = departure[0] - path[path.Length - 1];
            // Keep the authoring cell identities while joining the continuous, vehicle-width-aware lane references.
            // The immutable ribbon remains authoritative; IsFeasible and runtime footprint checks reject any unsafe shift.
            for (int i = 0; i < path.Length; i++)
            {
                float t = stations[i] / length, blend = t * t * (3 - 2 * t);
                path[i] += Vector3.Lerp(startShift, endShift, blend);
            }
        }
        return path;
    }
    public bool IsFeasible(float vehicleLength, float vehicleWidth, int sublane)
    {
        var key = (vehicleLength, vehicleWidth, sublane);
        if (feasibilityCache.TryGetValue(key, out bool feasible)) return feasible;
        feasible = ComputeFeasibility(vehicleLength, vehicleWidth, sublane);
        if (feasibilityCache.Count < 256) feasibilityCache.TryAdd(key, feasible);
        return feasible;
    }
    private bool ComputeFeasibility(float vehicleLength, float vehicleWidth, int sublane)
    {
        if (!TrafficNetwork.Finite(vehicleLength) || !TrafficNetwork.Finite(vehicleWidth) || vehicleLength <= 0 || vehicleWidth <= 0) return false;
        Vector3[] reference;
        try { reference = ReferencePath(sublane, vehicleWidth, vehicleLength); } catch (InvalidOperationException) { return false; }
        return ReferenceFits(reference, vehicleLength, vehicleWidth);
    }
    private bool ReferenceFits(Vector3[] reference, float vehicleLength, float vehicleWidth)
    {
        // Check the rotated full footprint through bends and interpolated narrowing sections.
        // Longitudinal overhang at endpoints is checked against the adjacent surface by the route/runtime.
        for (int i = 1; i < reference.Length; i++)
        {
            Vector3 delta = reference[i] - reference[i - 1]; delta.y = 0;
            if (delta.sqrMagnitude < 0.000001f) return false;
            Vector3 tangent = delta.normalized, side = new Vector3(tangent.z, 0, -tangent.x);
            int samples = Mathf.Max(1, Mathf.CeilToInt(delta.magnitude / Mathf.Min(0.5f, vehicleLength * 0.25f)));
            for (int step = 0; step <= samples; step++)
            {
                Vector3 centre = Vector3.Lerp(reference[i - 1], reference[i], (float)step / samples);
                foreach (float along in new[] { -0.5f, 0f, 0.5f })
                    foreach (float across in new[] { -0.5f, 0.5f })
                    {
                        Vector3 corner = centre + tangent * (vehicleLength * along) + side * (vehicleWidth * across);
                        if (!ContainsSurfacePoint(corner, vehicleLength * 0.5f + 0.5f, vehicleLength * 0.5f + 0.5f)) return false;
                    }
            }
        }
        return true;
    }
    public Vector3[] RibbonPolygon()
    {
        Vector3[] polygon = new Vector3[left.Length + right.Length];
        for (int i = 0; i < left.Length; i++) polygon[i] = left[i];
        for (int i = 0; i < right.Length; i++) polygon[left.Length + i] = right[right.Length - 1 - i];
        return polygon;
    }
    internal IEnumerable<Vector3> RibbonSamples()
    {
        // Validate interpolated sides and cross sections as well as the authored vertices.
        for (int i = 1; i < points.Length; i++)
        {
            int steps = Mathf.Max(1, Mathf.CeilToInt(TrafficNetwork.HorizontalDistance(points[i - 1], points[i]) / 0.2f));
            for (int j = 0; j <= steps; j++)
            {
                float t = (float)j / steps;
                Vector3 a = Vector3.Lerp(left[i - 1], left[i], t), b = Vector3.Lerp(right[i - 1], right[i], t);
                int across = Mathf.Max(1, Mathf.CeilToInt(TrafficNetwork.HorizontalDistance(a, b) / 0.2f));
                for (int k = 0; k <= across; k++) yield return Vector3.Lerp(a, b, (float)k / across);
            }
        }
    }
    public Dictionary<string, object> ToWire()
    {
        if (type == "connector") return TrafficJson.Obj("id", id, "from_edge_id", fromEdgeId, "to_edge_id", toEdgeId,
            "from_sublane", fromSublane, "to_sublane", toSublane, "movement_id", movementId, "curve_m", TrafficNetwork.Vectors(points),
            "length_m", length, "speed_limit_mps", speedLimit, "allowed_vehicle_kinds", allowedKinds,
            "drivable_envelope", TrafficJson.Obj("left_boundary_m", TrafficNetwork.Vectors(left), "right_boundary_m", TrafficNetwork.Vectors(right), "width_profile_m", widths, "min_y_m", MinY, "max_y_m", MaxY));
        int[] indices = new int[sublaneCount]; for (int i = 0; i < indices.Length; i++) indices[i] = i;
        return TrafficJson.Obj("id", id, "lane_id", laneId, "direction", forward ? "forward" : "reverse", "start_node_id", startNodeId, "end_node_id", endNodeId,
            "centerline_m", TrafficNetwork.Vectors(points), "left_boundary_m", TrafficNetwork.Vectors(left), "right_boundary_m", TrafficNetwork.Vectors(right), "up_vectors", TrafficNetwork.Vectors(up),
            "width_profile_m", widths, "length_m", length, "speed_limit_mps", speedLimit, "allowed_vehicle_kinds", allowedKinds, "sublane_count", sublaneCount, "authoring_sublane_indices", indices);
    }
}

public sealed class TrafficMovement
{
    public string id, approachEdgeId, departureEdgeId, signalGroupId, stopLineId, controllerId, priorityRule;
    public readonly List<string> connectorIds = new List<string>();
    public string[] allowedKinds, conflictZoneIds = new string[0];
    /// <summary>Approach-edge station of the authored stop line, or approach length for an uncontrolled connector entrance.</summary>
    public float stopS;
    public Vector3 stopA, stopB;
    public int firstSublane, lastSublane;
    public Dictionary<string, object> ToWire() => TrafficJson.Obj("id", id, "junction_controller_id", controllerId, "approach_edge_id", approachEdgeId, "departure_edge_id", departureEdgeId,
        "connector_ids", connectorIds, "allowed_vehicle_kinds", allowedKinds, "signal_group_id", signalGroupId, "stop_line_id", stopLineId, "uncontrolled_priority_rule", priorityRule, "conflict_zone_ids", conflictZoneIds);
}

public class TrafficArea
{
    public string id;
    public Vector3[] polygon;
    public float minY, maxY;
    public TrafficArea(string id, Vector3[] polygon)
    {
        this.id = id; this.polygon = polygon; minY = float.PositiveInfinity; maxY = float.NegativeInfinity;
        foreach (Vector3 p in polygon) { minY = Mathf.Min(minY, p.y); maxY = Mathf.Max(maxY, p.y); }
    }
    public bool ContainsHorizontal(Vector3 p) => Contains(polygon, p);
    public static bool Contains(Vector3[] polygon, Vector3 p)
    {
        bool inside = false;
        for (int i = 0, j = polygon.Length - 1; i < polygon.Length; j = i++)
        {
            Vector3 a = polygon[j], b = polygon[i];
            if (Mathf.Abs(Cross(a, b, p)) < 0.0001f && p.x >= Mathf.Min(a.x, b.x) - 0.0001f && p.x <= Mathf.Max(a.x, b.x) + 0.0001f && p.z >= Mathf.Min(a.z, b.z) - 0.0001f && p.z <= Mathf.Max(a.z, b.z) + 0.0001f) return true;
            if ((a.z > p.z) != (b.z > p.z) && p.x < (b.x - a.x) * (p.z - a.z) / (b.z - a.z) + a.x) inside = !inside;
        }
        return inside;
    }
    internal static float Cross(Vector3 a, Vector3 b, Vector3 c) => (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x);
    public static bool SegmentsCross(Vector3 a, Vector3 b, Vector3 c, Vector3 d)
    {
        float abC = Cross(a, b, c), abD = Cross(a, b, d), cdA = Cross(c, d, a), cdB = Cross(c, d, b);
        if (abC * abD < 0 && cdA * cdB < 0) return true;
        if (Mathf.Abs(abC) <= 0.0001f && OnSegment(a, b, c)) return true;
        if (Mathf.Abs(abD) <= 0.0001f && OnSegment(a, b, d)) return true;
        if (Mathf.Abs(cdA) <= 0.0001f && OnSegment(c, d, a)) return true;
        return Mathf.Abs(cdB) <= 0.0001f && OnSegment(c, d, b);
    }
    private static bool OnSegment(Vector3 a, Vector3 b, Vector3 p) => p.x >= Mathf.Min(a.x, b.x) - 0.0001f && p.x <= Mathf.Max(a.x, b.x) + 0.0001f && p.z >= Mathf.Min(a.z, b.z) - 0.0001f && p.z <= Mathf.Max(a.z, b.z) + 0.0001f;
    public static bool PolygonsOverlap(Vector3[] first, Vector3[] second)
    {
        foreach (Vector3 p in first) if (Contains(second, p)) return true;
        foreach (Vector3 p in second) if (Contains(first, p)) return true;
        for (int i = 0; i < first.Length; i++) for (int j = 0; j < second.Length; j++)
            if (SegmentsCross(first[i], first[(i + 1) % first.Length], second[j], second[(j + 1) % second.Length])) return true;
        return false;
    }
    internal struct ConflictSample { public Vector3 point; public float minY, maxY; }
    private struct SurfacePairPoint { public Vector3 point; public float firstY, secondY; }
    internal static List<ConflictSample> RibbonConflictSamples(TrafficSegment first, TrafficSegment second, float vehicleHeight)
    {
        List<ConflictSample> samples = new List<ConflictSample>();
        for (int i = 1; i < first.points.Length; i++)
            for (int j = 1; j < second.points.Length; j++)
            {
                Vector3[][] a = { new[] { first.left[i - 1], first.right[i - 1], first.right[i] }, new[] { first.left[i - 1], first.right[i], first.left[i] } };
                Vector3[][] b = { new[] { second.left[j - 1], second.right[j - 1], second.right[j] }, new[] { second.left[j - 1], second.right[j], second.left[j] } };
                foreach (Vector3[] ta in a) foreach (Vector3[] tb in b)
                    samples.AddRange(TriangleConflictSamples(ta, tb, vehicleHeight));
            }
        return samples;
    }
    private static List<ConflictSample> TriangleConflictSamples(Vector3[] first, Vector3[] second, float vehicleHeight)
    {
        List<Vector3> candidates = new List<Vector3>();
        foreach (Vector3 p in first) if (Contains(second, p)) candidates.Add(p);
        foreach (Vector3 p in second) if (Contains(first, p)) candidates.Add(p);
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
        {
            Vector3 a = first[i], b = first[(i + 1) % 3], c = second[j], d = second[(j + 1) % 3];
            float denominator = (b.x - a.x) * (d.z - c.z) - (b.z - a.z) * (d.x - c.x);
            if (Mathf.Abs(denominator) < 0.000001f || !SegmentsCross(a, b, c, d)) continue;
            float t = ((c.x - a.x) * (d.z - c.z) - (c.z - a.z) * (d.x - c.x)) / denominator;
            candidates.Add(Vector3.Lerp(a, b, t));
        }
        List<SurfacePairPoint> polygon = new List<SurfacePairPoint>();
        foreach (Vector3 p in candidates)
        {
            if (!TriangleHeight(first, p, out float aY) || !TriangleHeight(second, p, out float bY)) continue;
            if (polygon.Exists(existing => TrafficNetwork.HorizontalDistance(existing.point, p) < 0.0001f)) continue;
            polygon.Add(new SurfacePairPoint { point = p, firstY = aY, secondY = bY });
        }
        Vector3 centre = Vector3.zero; foreach (SurfacePairPoint p in polygon) centre += p.point;
        if (polygon.Count != 0) centre /= polygon.Count;
        polygon.Sort((a, b) => Mathf.Atan2(a.point.z - centre.z, a.point.x - centre.x).CompareTo(Mathf.Atan2(b.point.z - centre.z, b.point.x - centre.x)));
        // Height difference is affine over each triangle intersection. Clip the convex intersection
        // to the part where full vehicle bodies can overlap, preserving exact grade separation.
        polygon = ClipHeightDifference(polygon, vehicleHeight, 1);
        polygon = ClipHeightDifference(polygon, vehicleHeight, -1);
        List<ConflictSample> result = new List<ConflictSample>();
        foreach (SurfacePairPoint p in polygon) result.Add(new ConflictSample { point = p.point, minY = Mathf.Min(p.firstY, p.secondY), maxY = Mathf.Max(p.firstY, p.secondY) + vehicleHeight });
        // Include the crossing interior: checking only its corners could miss a notch or gap
        // between several explicitly authored zones.
        for (int i = 1; i + 1 < polygon.Count; i++)
        {
            SurfacePairPoint a = polygon[0], b = polygon[i], c = polygon[i + 1];
            float longest = Mathf.Max(TrafficNetwork.HorizontalDistance(a.point, b.point), TrafficNetwork.HorizontalDistance(b.point, c.point), TrafficNetwork.HorizontalDistance(c.point, a.point));
            int divisions = Mathf.Max(1, Mathf.CeilToInt(longest / 0.2f));
            for (int row = 0; row <= divisions; row++) for (int column = 0; column <= divisions - row; column++)
            {
                float wb = (float)row / divisions, wc = (float)column / divisions, wa = 1 - wb - wc;
                float firstY = wa * a.firstY + wb * b.firstY + wc * c.firstY, secondY = wa * a.secondY + wb * b.secondY + wc * c.secondY;
                result.Add(new ConflictSample { point = wa * a.point + wb * b.point + wc * c.point, minY = Mathf.Min(firstY, secondY), maxY = Mathf.Max(firstY, secondY) + vehicleHeight });
            }
        }
        return result;
    }
    private static List<SurfacePairPoint> ClipHeightDifference(List<SurfacePairPoint> polygon, float bound, float sign)
    {
        List<SurfacePairPoint> result = new List<SurfacePairPoint>();
        if (polygon.Count == 0) return result;
        SurfacePairPoint previous = polygon[polygon.Count - 1];
        float previousValue = (previous.firstY - previous.secondY) * sign - bound;
        foreach (SurfacePairPoint current in polygon)
        {
            float value = (current.firstY - current.secondY) * sign - bound;
            bool previousInside = previousValue <= 0.00001f, inside = value <= 0.00001f;
            if (previousInside != inside)
            {
                float t = previousValue / (previousValue - value);
                result.Add(new SurfacePairPoint { point = Vector3.Lerp(previous.point, current.point, t), firstY = Mathf.Lerp(previous.firstY, current.firstY, t), secondY = Mathf.Lerp(previous.secondY, current.secondY, t) });
            }
            if (inside) result.Add(current);
            previous = current; previousValue = value;
        }
        return result;
    }
    private static bool TriangleHeight(Vector3[] triangle, Vector3 point, out float height)
    {
        float area = Cross(triangle[0], triangle[1], triangle[2]); height = 0;
        if (Mathf.Abs(area) < 0.000001f) return false;
        float a = Cross(point, triangle[1], triangle[2]) / area, b = Cross(triangle[0], point, triangle[2]) / area;
        height = a * triangle[0].y + b * triangle[1].y + (1 - a - b) * triangle[2].y; return true;
    }
    public virtual Dictionary<string, object> ToWire() => TrafficJson.Obj("id", id, "polygon_m", TrafficNetwork.Vectors(polygon), "min_y_m", minY, "max_y_m", maxY);
}

public sealed class TrafficConflictZone : TrafficArea
{
    public readonly List<string[]> movementPairs = new List<string[]>();
    public TrafficConflictZone(string id, Vector3[] polygon, float minY, float maxY) : base(id, polygon) { this.minY = minY; this.maxY = maxY; }
    public bool OverlapsFootprint(Vector3 centre, float heading, float length, float width, float height)
    {
        if (centre.y > maxY || centre.y + height < minY) return false;
        Vector3 forward = new Vector3(Mathf.Sin(heading), 0, Mathf.Cos(heading)), side = new Vector3(forward.z, 0, -forward.x);
        Vector3 a = forward * length * 0.5f, b = side * width * 0.5f;
        return PolygonsOverlap(polygon, new[] { centre + a + b, centre + a - b, centre - a - b, centre - a + b });
    }
    public override Dictionary<string, object> ToWire() => TrafficJson.Obj("id", id, "polygon_m", TrafficNetwork.Vectors(polygon), "min_y_m", minY, "max_y_m", maxY, "movement_pairs", movementPairs);
}
#endif
