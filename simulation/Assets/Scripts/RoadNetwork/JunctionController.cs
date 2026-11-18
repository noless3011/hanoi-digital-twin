#if LEGACY_SUBLANE_RUNTIME // Old Unity simulation map; unused since the sublane system was removed.
using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>Authored protected signal movements. Only Simulation advances this controller's clock.</summary>
[DisallowMultipleComponent, AddComponentMenu("Digital Twin/Junction Controller")]
public sealed class JunctionController : MonoBehaviour
{
    [Serializable] public sealed class MovementBinding
    {
        [Tooltip("Stable TrafficNetwork movement ID, based on lane IDs, travel directions and shared endpoint ID.")]
        public string movementId;
        public string signalGroupId;
        public string stopLineId;
        [Tooltip("Required for unsignalized movements. Supported initial rule: yield_to_conflicting_traffic.")]
        public string uncontrolledPriorityRule;
        public string[] allowedKinds = { "car", "motorcycle", "bus", "truck", "van" };
        public string[] conflictZoneIds = new string[0];
    }
    [Serializable] public sealed class ConnectorEnvelope
    {
        [Tooltip("Bind one connector, or leave empty to apply the envelope settings to every connector of movementId.")]
        public string connectorId;
        public string movementId;
        [Tooltip("Explicit validated drivable width: one constant value, or one value per retained curve sample.")]
        public float[] widthProfileM = { 1f };
        [Tooltip("Optional controller-local junction surface polygon. If empty, the ribbon must fit the editor's existing road/junction surfaces.")]
        public Vector3[] localPolygon = new Vector3[0];
    }
    [Serializable] public sealed class StopLine
    {
        public string id = Guid.NewGuid().ToString("N");
        public string approachEdgeId;
        [Min(0)] public float edgeS;
        public Vector3 localA, localB;
        [Min(0)] public int firstSublane;
        [Min(0)] public int lastSublane;
    }
    [Serializable] public sealed class SignalHead
    {
        [Tooltip("Optional scene head. Its world position/orientation overrides the local fields below.")]
        public Transform sceneHead;
        public Vector3 localPosition;
        public float localHeadingDegrees;
        [Tooltip("Optional lamp renderer, tinted directly from the authoritative group state.")]
        public Renderer lamp;
        public GameObject redLamp, amberLamp, greenLamp;
    }
    [Serializable] public sealed class SignalGroup
    {
        public string id = Guid.NewGuid().ToString("N");
        public List<SignalHead> heads = new List<SignalHead>();
    }
    [Serializable] public sealed class MovementPair
    {
        public string first, second;
    }
    [Serializable] public sealed class ConflictZone
    {
        public string id = Guid.NewGuid().ToString("N");
        [Tooltip("Controller-local horizontal boundary, ordered around the polygon.")]
        public Vector3[] localPolygon = new Vector3[0];
        [Tooltip("Controller-local vertical extent, including vehicle height. Keep grade-separated road surfaces in distinct zones.")]
        public float minLocalY, maxLocalY = 5;
        public List<MovementPair> movementPairs = new List<MovementPair>();
    }
    [Serializable] public sealed class Phase
    {
        public string name;
        public string[] greenGroupIds = new string[0];
        [Min(0.05f)] public float greenSeconds = 25;
        [Min(0.05f)] public float amberSeconds = 3;
        [Min(0.05f)] public float allRedSeconds = 1;
    }

    public string id = Guid.NewGuid().ToString("N");
    public List<MovementBinding> movements = new List<MovementBinding>();
    public List<ConnectorEnvelope> connectorEnvelopes = new List<ConnectorEnvelope>();
    public List<StopLine> stopLines = new List<StopLine>();
    public List<SignalGroup> signalGroups = new List<SignalGroup>();
    public List<ConflictZone> conflictZones = new List<ConflictZone>();
    public List<Phase> phases = new List<Phase>();
    [Tooltip("Optional Lane Editor used only by the component's Copy generated traffic IDs context-menu command.")]
    public LaneEditor authoringLaneEditor;
    private enum Stage { Startup, Green, Amber, AllRed }
    private Stage stage = Stage.Startup;
    private int phaseIndex;
    private double elapsed;
    private TrafficNetwork network;
    private readonly Dictionary<string, List<TrafficMovement>> groupMovements = new Dictionary<string, List<TrafficMovement>>(StringComparer.Ordinal);
    private readonly Dictionary<string, string> aspects = new Dictionary<string, string>(StringComparer.Ordinal);
    public bool HoldingForOccupancy { get; private set; }

    [ContextMenu("Copy generated traffic IDs")]
    public void CopyGeneratedTrafficIds()
    {
        try
        {
            LaneEditor editor = authoringLaneEditor;
            if (editor == null)
            {
                LaneEditor[] candidates = FindObjectsByType<LaneEditor>();
                if (candidates.Length != 1) throw new InvalidOperationException("Assign Authoring Lane Editor when the scene has zero or several Lane Editors.");
                editor = candidates[0];
            }
            string catalog = TrafficJson.Encode(TrafficNetwork.GeneratedIdentityCatalog(editor));
            GUIUtility.systemCopyBuffer = catalog;
            Debug.Log("Copied stable lane, movement and connector candidate IDs to the clipboard. Candidates still require explicit control bindings and drivable envelopes.\n" + catalog, this);
        }
        catch (Exception error) { Debug.LogError("Cannot export traffic IDs: " + error.Message, this); }
    }

    internal void CompileBindings(TrafficNetwork map)
    {
        network = map; groupMovements.Clear(); aspects.Clear();
        HashSet<string> groupIds = new HashSet<string>(StringComparer.Ordinal), lineIds = new HashSet<string>(StringComparer.Ordinal);
        foreach (SignalGroup group in signalGroups)
        {
            if (group == null || string.IsNullOrEmpty(group.id) || !groupIds.Add(group.id)) throw new InvalidOperationException("Signal groups need unique IDs in controller " + id);
            foreach (JunctionController other in map.controllers)
                if (other != this && other.signalGroups.Exists(g => g != null && g.id == group.id)) throw new InvalidOperationException("Signal group ID occurs in several controllers: " + group.id);
            if (group.heads == null || group.heads.Count == 0) throw new InvalidOperationException("Signal group needs an explicitly bound physical head: " + group.id);
            foreach (SignalHead head in group.heads)
                if (head == null || !TrafficNetwork.Finite(HeadPosition(head)) || !TrafficNetwork.Finite(HeadHeading(head))) throw new InvalidOperationException("Invalid physical signal head in " + group.id);
            groupMovements.Add(group.id, new List<TrafficMovement>()); aspects.Add(group.id, "red");
        }
        foreach (StopLine line in stopLines)
        {
            if (line == null || string.IsNullOrEmpty(line.id) || !lineIds.Add(line.id)) throw new InvalidOperationException("Stop lines need unique IDs in controller " + id);
            foreach (JunctionController other in map.controllers)
                if (other != this && other.stopLines.Exists(l => l != null && l.id == line.id)) throw new InvalidOperationException("Stop-line ID occurs in several controllers: " + line.id);
            if (!map.segments.TryGetValue(line.approachEdgeId ?? "", out TrafficSegment approach) || approach.type != "lane_edge") throw new InvalidOperationException("Unknown directed stop-line approach: " + line.approachEdgeId);
            if (!TrafficNetwork.Finite(line.edgeS) || line.edgeS < 0 || line.edgeS >= approach.length || line.firstSublane < 0 || line.lastSublane < line.firstSublane || line.lastSublane >= approach.sublaneCount)
                throw new InvalidOperationException("Invalid stop-line station or authoring sublane range: " + line.id);
            Vector3 a = transform.TransformPoint(line.localA), b = transform.TransformPoint(line.localB);
            if (!TrafficNetwork.Finite(a) || !TrafficNetwork.Finite(b) || TrafficNetwork.HorizontalDistance(a, b) < 0.01f) throw new InvalidOperationException("Stop line needs distinct finite endpoints: " + line.id);
            approach.Project(a, out float aS, out float aLateral); approach.Project(b, out float bS, out float bLateral);
            if (Mathf.Abs(aS - line.edgeS) > 0.25f || Mathf.Abs(bS - line.edgeS) > 0.25f || Mathf.Abs(a.y - approach.Sample(line.edgeS).y) > 0.5f || Mathf.Abs(b.y - approach.Sample(line.edgeS).y) > 0.5f)
                throw new InvalidOperationException("Stop-line endpoints must lie at its directed approach edge station: " + line.id);
            float physicalFirst = approach.forward ? line.firstSublane : approach.sublaneCount - 1 - line.lastSublane;
            float physicalLast = approach.forward ? line.lastSublane : approach.sublaneCount - 1 - line.firstSublane;
            float cell = approach.WidthAt(line.edgeS) / approach.sublaneCount;
            float requiredLeft = -approach.WidthAt(line.edgeS) * 0.5f + physicalFirst * cell;
            float requiredRight = -approach.WidthAt(line.edgeS) * 0.5f + (physicalLast + 1) * cell;
            if (Mathf.Min(aLateral, bLateral) > requiredLeft + 0.1f || Mathf.Max(aLateral, bLateral) < requiredRight - 0.1f)
                throw new InvalidOperationException("Stop-line endpoints do not span the governed sublanes: " + line.id);
        }
        foreach (MovementBinding binding in movements)
        {
            TrafficMovement movement = map.FindMovement(binding.movementId);
            if (movement == null) throw new InvalidOperationException("Unknown authored movement: " + binding.movementId);
            if (!string.IsNullOrEmpty(movement.priorityRule) && movement.priorityRule != "yield_to_conflicting_traffic")
                throw new InvalidOperationException("Unsupported uncontrolled priority rule on " + movement.id + "; initial version supports yield_to_conflicting_traffic.");
            if (!string.IsNullOrEmpty(movement.signalGroupId))
            {
                if (!groupMovements.TryGetValue(movement.signalGroupId, out List<TrafficMovement> governed)) throw new InvalidOperationException("Movement references a signal group outside its controller: " + movement.id);
                governed.Add(movement);
            }
            if (!string.IsNullOrEmpty(movement.stopLineId))
            {
                StopLine line = stopLines.Find(l => l.id == movement.stopLineId);
                if (line == null || line.approachEdgeId != movement.approachEdgeId) throw new InvalidOperationException("Movement stop line must bind its exact directed approach: " + movement.id);
                foreach (string connectorId in movement.connectorIds)
                {
                    int cell = map.segments[connectorId].fromSublane;
                    if (cell < line.firstSublane || cell > line.lastSublane) throw new InvalidOperationException("Movement connector is outside its stop-line sublane range: " + connectorId);
                }
                movement.stopS = line.edgeS; movement.stopA = transform.TransformPoint(line.localA); movement.stopB = transform.TransformPoint(line.localB);
                movement.firstSublane = line.firstSublane; movement.lastSublane = line.lastSublane;
            }
        }
        foreach (KeyValuePair<string, List<TrafficMovement>> group in groupMovements)
            if (group.Value.Count == 0) throw new InvalidOperationException("Signal group governs no movement: " + group.Key);
        foreach (StopLine line in stopLines)
            if (!map.movements.Exists(m => m.controllerId == id && m.stopLineId == line.id)) throw new InvalidOperationException("Stop line governs no movement: " + line.id);
        foreach (ConflictZone authored in conflictZones)
        {
            if (Vector3.Dot(transform.TransformVector(Vector3.up).normalized, Vector3.up) < 0.9999f) throw new InvalidOperationException("Conflict-zone controller transforms must remain upright; rotate around world Y and author explicit world-vertical height bounds.");
            if (authored == null || string.IsNullOrEmpty(authored.id) || map.FindZone(authored.id) != null) throw new InvalidOperationException("Conflict zones need globally unique stable IDs.");
            Vector3[] polygon = TrafficNetwork.TransformPoints(transform, authored.localPolygon ?? new Vector3[0]);
            TrafficNetwork.ValidatePolygon(polygon, "conflict zone " + authored.id);
            if (!TrafficNetwork.Finite(authored.minLocalY) || !TrafficNetwork.Finite(authored.maxLocalY) || authored.minLocalY >= authored.maxLocalY)
                throw new InvalidOperationException("Conflict zone needs a positive finite height interval: " + authored.id);
            float minY = float.PositiveInfinity, maxY = float.NegativeInfinity;
            foreach (Vector3 p in authored.localPolygon)
                foreach (float y in new[] { authored.minLocalY, authored.maxLocalY })
                {
                    float worldY = transform.TransformPoint(new Vector3(p.x, y, p.z)).y; minY = Mathf.Min(minY, worldY); maxY = Mathf.Max(maxY, worldY);
                }
            TrafficConflictZone zone = new TrafficConflictZone(authored.id, polygon, minY, maxY);
            if (authored.movementPairs == null || authored.movementPairs.Count == 0) throw new InvalidOperationException("Conflict zone needs explicit incompatible movement pairs: " + authored.id);
            HashSet<string> pairs = new HashSet<string>();
            foreach (MovementPair pair in authored.movementPairs)
            {
                TrafficMovement first = pair == null ? null : map.FindMovement(pair.first), second = pair == null ? null : map.FindMovement(pair.second);
                if (first == null || second == null || first.id == second.id) throw new InvalidOperationException("Invalid movement pair in zone " + zone.id);
                if (Array.IndexOf(first.conflictZoneIds, zone.id) < 0 || Array.IndexOf(second.conflictZoneIds, zone.id) < 0)
                    throw new InvalidOperationException("Both movements must bind their conflict zone: " + zone.id);
                string key = string.CompareOrdinal(first.id, second.id) < 0 ? first.id + "|" + second.id : second.id + "|" + first.id;
                if (!pairs.Add(key)) throw new InvalidOperationException("Duplicate movement pair in zone " + zone.id);
                if (!MovementIntersectsZone(map, first, zone) || !MovementIntersectsZone(map, second, zone))
                    throw new InvalidOperationException("Conflict zone must overlap both movement corridors at their road elevation: " + zone.id);
                // Separate clocks cannot promise protected green across a shared conflict.
                if (!string.IsNullOrEmpty(first.signalGroupId) && !string.IsNullOrEmpty(second.signalGroupId) && first.controllerId != second.controllerId)
                    throw new InvalidOperationException("Conflicting protected movements must share one JunctionController: " + first.id + " / " + second.id);
                zone.movementPairs.Add(new[] { first.id, second.id });
            }
            map.conflictZones.Add(zone);
        }
        ResetController();
    }

    private static bool MovementIntersectsZone(TrafficNetwork map, TrafficMovement movement, TrafficConflictZone zone)
    {
        foreach (string connectorId in movement.connectorIds)
        {
            TrafficSegment segment = map.segments[connectorId];
            if (segment.MaxY >= zone.minY - 0.01f && segment.MinY <= zone.maxY + 0.01f && TrafficArea.PolygonsOverlap(segment.RibbonPolygon(), zone.polygon)) return true;
        }
        return false;
    }

    public void Validate(TrafficNetwork map, float decisionDt)
    {
        if (network != map) throw new InvalidOperationException("Compile controller bindings before validation.");
        if (!TrafficNetwork.Finite(decisionDt) || decisionDt <= 0) throw new InvalidOperationException("Decision interval must be positive.");
        foreach (MovementBinding binding in movements)
        {
            TrafficMovement movement = map.FindMovement(binding.movementId);
            foreach (string zoneId in movement.conflictZoneIds)
            {
                TrafficConflictZone zone = map.FindZone(zoneId);
                if (zone == null || !zone.movementPairs.Exists(p => p[0] == movement.id || p[1] == movement.id))
                    throw new InvalidOperationException("Movement references an unknown or unrelated conflict zone: " + movement.id + " / " + zoneId);
            }
        }
        if (signalGroups.Count == 0)
        {
            if (phases.Count != 0) throw new InvalidOperationException("A controller without signal groups cannot have a signal plan.");
            return;
        }
        if (phases.Count == 0) throw new InvalidOperationException("Author at least one phase for controller " + id);
        HashSet<string> servedGroups = new HashSet<string>();
        foreach (Phase phase in phases)
        {
            if (phase == null || phase.greenGroupIds == null || phase.greenGroupIds.Length == 0) throw new InvalidOperationException("Every phase needs at least one green signal group.");
            foreach (float duration in new[] { phase.greenSeconds, phase.amberSeconds, phase.allRedSeconds })
                if (!TrafficNetwork.Finite(duration) || duration < decisionDt - 0.00001f || Math.Abs(duration / decisionDt - Math.Round(duration / decisionDt)) > 0.0001)
                    throw new InvalidOperationException("Green, amber and all-red durations must be positive integer multiples of decision_dt_s in controller " + id);
            HashSet<string> unique = new HashSet<string>();
            List<TrafficMovement> concurrent = new List<TrafficMovement>();
            foreach (string groupId in phase.greenGroupIds)
            {
                if (!unique.Add(groupId) || !groupMovements.TryGetValue(groupId, out List<TrafficMovement> governed)) throw new InvalidOperationException("Unknown/duplicate phase group: " + groupId);
                servedGroups.Add(groupId); concurrent.AddRange(governed);
            }
            for (int i = 0; i < concurrent.Count; i++) for (int j = i + 1; j < concurrent.Count; j++)
                if (map.MovementsConflict(concurrent[i].id, concurrent[j].id)) throw new InvalidOperationException("Protected phase gives conflicting movements green together: " + concurrent[i].id + " / " + concurrent[j].id);
        }
        foreach (SignalGroup group in signalGroups)
            if (!servedGroups.Contains(group.id)) throw new InvalidOperationException("Signal group is never served by the configured phases: " + group.id);
        ResetController();
    }

    public void ResetController()
    {
        stage = Stage.Startup; phaseIndex = 0; elapsed = 0; HoldingForOccupancy = false;
        foreach (SignalGroup group in signalGroups) if (group != null && !string.IsNullOrEmpty(group.id)) aspects[group.id] = "red";
        RenderHeads();
    }

    public void Tick(float dt, ISet<string> occupiedMovements, ISet<string> occupiedZones)
    {
        if (!TrafficNetwork.Finite(dt) || dt < 0) throw new ArgumentOutOfRangeException(nameof(dt));
        if (phases.Count == 0 || network == null) return;
        elapsed += dt;
        Phase phase = phases[phaseIndex];
        HoldingForOccupancy = false;
        if (stage == Stage.Startup)
        {
            if (elapsed + 0.000001 >= phase.allRedSeconds)
            {
                if (CanRelease(phase, occupiedMovements, occupiedZones)) Enter(Stage.Green);
                else HoldingForOccupancy = true;
            }
        }
        else if (stage == Stage.Green && elapsed + 0.000001 >= phase.greenSeconds) Enter(Stage.Amber);
        else if (stage == Stage.Amber && elapsed + 0.000001 >= phase.amberSeconds) Enter(Stage.AllRed);
        else if (stage == Stage.AllRed && elapsed + 0.000001 >= phase.allRedSeconds)
        {
            int next = (phaseIndex + 1) % phases.Count;
            if (CanRelease(phases[next], occupiedMovements, occupiedZones)) { phaseIndex = next; Enter(Stage.Green); }
            else HoldingForOccupancy = true;
        }
        RenderHeads();
    }

    private bool CanRelease(Phase phase, ISet<string> occupiedMovements, ISet<string> occupiedZones)
    {
        foreach (string group in phase.greenGroupIds)
            foreach (TrafficMovement candidate in groupMovements[group])
            {
                if (occupiedMovements != null)
                    foreach (string occupied in occupiedMovements) if (network.MovementsConflict(candidate.id, occupied)) return false;
                if (occupiedZones != null)
                    foreach (TrafficConflictZone zone in network.conflictZones)
                        if (occupiedZones.Contains(zone.id) && zone.movementPairs.Exists(pair => pair[0] == candidate.id || pair[1] == candidate.id)) return false;
            }
        return true;
    }

    private void Enter(Stage value)
    {
        stage = value; elapsed = 0;
        foreach (SignalGroup group in signalGroups) aspects[group.id] = "red";
        if (value == Stage.Green || value == Stage.Amber)
            foreach (string groupId in phases[phaseIndex].greenGroupIds) aspects[groupId] = value == Stage.Green ? "green" : "amber";
    }

    public string Aspect(string groupId) => groupId != null && aspects.TryGetValue(groupId, out string value) ? value : "unknown";
    public bool HeadVisible(string groupId, Vector3 ego, float radius)
    {
        SignalGroup group = signalGroups.Find(g => g.id == groupId);
        if (group == null || !TrafficNetwork.Finite(radius) || radius < 0) return false;
        foreach (SignalHead head in group.heads) if ((HeadPosition(head) - ego).sqrMagnitude <= radius * radius) return true;
        return false;
    }
    public Vector3 HeadPosition(SignalHead head) => head.sceneHead != null ? head.sceneHead.position : transform.TransformPoint(head.localPosition);
    public float HeadHeading(SignalHead head)
    {
        Vector3 forward = head.sceneHead != null ? head.sceneHead.forward : transform.TransformDirection(Quaternion.Euler(0, head.localHeadingDegrees, 0) * Vector3.forward);
        return Mathf.Atan2(forward.x, forward.z);
    }
    private void RenderHeads()
    {
        foreach (SignalGroup group in signalGroups)
        {
            if (group == null || group.heads == null) continue;
            string aspect = Aspect(group.id);
            foreach (SignalHead head in group.heads)
            {
                if (head == null) continue;
                if (head.redLamp != null) head.redLamp.SetActive(aspect == "red");
                if (head.amberLamp != null) head.amberLamp.SetActive(aspect == "amber");
                if (head.greenLamp != null) head.greenLamp.SetActive(aspect == "green");
                if (head.lamp == null) continue;
                Color color = aspect == "green" ? Color.green : aspect == "amber" ? new Color(1, 0.65f, 0) : Color.red;
                MaterialPropertyBlock block = new MaterialPropertyBlock(); head.lamp.GetPropertyBlock(block);
                block.SetColor("_BaseColor", color); block.SetColor("_Color", color); block.SetColor("_EmissionColor", color); head.lamp.SetPropertyBlock(block);
            }
        }
    }

    internal List<object> StopLinesToWire(TrafficNetwork map)
    {
        List<object> result = new List<object>();
        foreach (StopLine line in stopLines)
        {
            List<string> governed = new List<string>(); foreach (TrafficMovement movement in map.movements) if (movement.stopLineId == line.id) governed.Add(movement.id);
            result.Add(TrafficJson.Obj("id", line.id, "approach_edge_id", line.approachEdgeId, "edge_s_m", line.edgeS,
                "endpoints_m", new[] { TrafficJson.Vec(transform.TransformPoint(line.localA)), TrafficJson.Vec(transform.TransformPoint(line.localB)) },
                "first_sublane", line.firstSublane, "last_sublane", line.lastSublane, "movement_ids", governed));
        }
        return result;
    }
    internal List<object> SignalGroupsToWire(TrafficNetwork map)
    {
        List<object> result = new List<object>();
        foreach (SignalGroup group in signalGroups)
        {
            List<string> governed = new List<string>(); foreach (TrafficMovement movement in map.movements) if (movement.signalGroupId == group.id) governed.Add(movement.id);
            List<object> heads = new List<object>(); foreach (SignalHead head in group.heads) heads.Add(TrafficJson.Obj("position_m", TrafficJson.Vec(HeadPosition(head)), "heading_rad", HeadHeading(head)));
            result.Add(TrafficJson.Obj("id", group.id, "movement_ids", governed, "heads", heads));
        }
        return result;
    }

    private void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.white;
        foreach (StopLine line in stopLines) if (line != null) Gizmos.DrawLine(transform.TransformPoint(line.localA), transform.TransformPoint(line.localB));
        Gizmos.color = new Color(1, 0.2f, 0.2f, 0.8f);
        foreach (ConflictZone zone in conflictZones)
            if (zone != null && zone.localPolygon != null && zone.localPolygon.Length >= 3)
                for (int i = 0; i < zone.localPolygon.Length; i++) Gizmos.DrawLine(transform.TransformPoint(zone.localPolygon[i]), transform.TransformPoint(zone.localPolygon[(i + 1) % zone.localPolygon.Length]));
        Gizmos.color = Color.yellow;
        foreach (SignalGroup group in signalGroups) if (group != null && group.heads != null) foreach (SignalHead head in group.heads) if (head != null) Gizmos.DrawSphere(HeadPosition(head), 0.15f);
    }
}
#endif
