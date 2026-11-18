using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>The authored road layout: lanes joined at shared nodes, spawn and dead end marks, and traffic
/// light stop lines. A lane is a directed line of nodes with a width; traffic runs from its first node to
/// its last. Editing happens in the Scene view (LaneEditorInspector). This component stores the data,
/// provides the graph operations, and draws the lanes.</summary>
[ExecuteAlways, DisallowMultipleComponent, SelectionBase]
[AddComponentMenu("Digital Twin/Lane Editor")]
public class LaneEditor : MonoBehaviour
{
    [Serializable] public class TrafficLightLine
    {
        public string id = Guid.NewGuid().ToString("N");
        public string name = "Traffic light";
        // Authoring coordinates, without the display-only surface offset.
        public Vector3 start, end;
        [Min(0)] public float redDurationS = 30;
        [Min(0)] public float greenDurationS = 30;
        public float cycleOffsetS;
        public string AspectAt(double time) => AspectAt(time, redDurationS, greenDurationS, cycleOffsetS);
        public static bool ValidTiming(float red, float green, float offset) => Finite(red) && red >= 0 && Finite(green) && green >= 0 && Finite(offset);
        public static string AspectAt(double time, float red, float green, float offset)
        {
            if (!ValidTiming(red, green, offset) || double.IsNaN(time) || double.IsInfinity(time)) throw new ArgumentException("Invalid traffic light timing.");
            if (green == 0) return "red"; // Also preserves older files without timing fields.
            if (red == 0) return "green";
            double cycle = (double)red + green;
            double phase = (time % cycle + offset % cycle) % cycle;
            if (phase < 0) phase += cycle;
            return phase < red ? "red" : "green";
        }
    }
    [Serializable] public class LaneNode
    {
        public string id = Guid.NewGuid().ToString("N");
        public Vector3 position;
        public bool isSpawnPoint;
        public string spawnPointName = "";
        [Tooltip("The last visible point of a lane, where vehicles leave the camera view.")]
        public bool isDeadEnd;
        public string deadEndName = "";
    }
    /// <summary>One lane's cross-section at a node. Nodes are shared between lanes; sections are not.</summary>
    [Serializable] public class Section
    {
        public string nodeId;
        [Min(0.1f)] public float widthScale = 1;
        [Tooltip("Rotation offset from the lane tangent (pitch, yaw, roll in degrees).")]
        public Vector3 rotation;
    }
    [Serializable] public class LaneStrip
    {
        public string id = Guid.NewGuid().ToString("N");
        public string name = "Lane";
        [Min(0.1f)] public float width = 3.2f;
        public List<Section> sections = new List<Section>();
        public void Normalize()
        {
            width = Finite(width) ? Mathf.Max(0.1f, width) : 3.2f;
            foreach (Section section in sections) section.widthScale = Mathf.Clamp(section.widthScale, 0.1f, 10);
        }
    }
    [Serializable] private class LaneFile
    {
        public int version = 3;
        public List<LaneNode> nodes;
        public List<LaneStrip> laneStrips;
        public List<TrafficLightLine> trafficLights;
    }

    [HideInInspector] public List<LaneNode> nodes = new List<LaneNode>();
    [HideInInspector] public List<LaneStrip> laneStrips = new List<LaneStrip>();
    [HideInInspector] public List<TrafficLightLine> trafficLights = new List<TrafficLightLine>();

    [Tooltip("Road model that receives clicks. Empty accepts any collider on the Surface Layers.")]
    public Transform surfaceRoot;
    public LayerMask surfaceLayers = ~0;
    [Tooltip("Add mesh colliders to the road model when drawing starts, so clicks can land on it.")]
    public bool prepareMeshColliders = true;

    [Tooltip("Fill each lane's area. Lane edges and direction arrows are always drawn.")]
    public bool showFill = true;
    public Color stripColor = new Color(0.16f, 0.21f, 0.24f);
    [Tooltip("Optional material for the lane fill. Empty uses Strip Color.")]
    public Material stripMaterial;
    [Tooltip("Optional material for edges and arrows.")]
    public Material laneMaterial;
    [Tooltip("Draw lanes as an overlay above uneven surfaces. Disable for normal scene depth, including vehicle occlusion.")]
    public bool drawAboveSurface = true;
    [Tooltip("Show lanes in Play mode. L toggles it while playing.")]
    public bool showLanesInGame = true;
    [Min(0.01f)] public float lineWidth = 0.08f;
    [Min(0.1f)] public float arrowSize = 0.8f;
    [Tooltip("Visible node marker size multiplier. Click targets use Node Pick Radius.")]
    [Range(0.1f, 5)] public float nodeMarkerSize = 1;
    [Tooltip("Node click radius in Scene view points. Enlarges selecting, dragging and joining without enlarging the marker.")]
    [Range(8, 64)] public float nodePickRadius = 26;
    public float surfaceOffset = 0.04f;

    [Tooltip("Feed-aligned camera to project lanes through. Empty uses the Main Camera.")]
    public Camera exportCamera;
    [Tooltip("Video resolution of the camera feed. The projection uses its aspect ratio, not the Game view's.")]
    public Vector2Int exportImageSize = new Vector2Int(384, 216);
    [Tooltip("File written under StreamingAssets/Traffic. Empty uses <scene>.camera-lanes.json.")]
    public string exportFileName = "";
    // Scenes saved before the export camera existed kept the feed camera here.
    [SerializeField, HideInInspector] private Camera drawingCamera;

    public IReadOnlyList<LaneStrip> LaneStrips => laneStrips;
    public IReadOnlyList<LaneNode> Nodes => nodes;
    public IReadOnlyList<TrafficLightLine> TrafficLights => trafficLights;
    public float NodePickRadius => Mathf.Clamp(nodePickRadius, 8, 64);
    public float NodeMarkerSize => Mathf.Clamp(nodeMarkerSize, 0.1f, 5);
    public string Status { get; private set; } = "";

    // Selection is editing state: it is not saved.
    public int SelectedLane { get; set; } = -1;
    public int SelectedSection { get; set; } = -1;
    public int SelectedTrafficLight { get; set; } = -1;
    public LaneStrip CurrentLane => SelectedLane >= 0 && SelectedLane < laneStrips.Count ? laneStrips[SelectedLane] : null;
    public LaneNode SelectedNode => CurrentLane != null && SelectedSection >= 0 && SelectedSection < CurrentLane.sections.Count
        ? FindNode(CurrentLane.sections[SelectedSection].nodeId) : null;
    public TrafficLightLine CurrentTrafficLight => SelectedTrafficLight >= 0 && SelectedTrafficLight < trafficLights.Count ? trafficLights[SelectedTrafficLight] : null;
    public bool IsDrawingTrafficLight { get; private set; }
    public bool HasTrafficLightStart { get; private set; }
    public Vector3 TrafficLightStartWorld => transform.TransformPoint(trafficLightStart);
    public double TrafficLightDisplayTime { get; private set; }

    private const string GeneratedName = "Generated lane network";
    private Vector3 trafficLightStart;
    private GameObject generatedRoot;
    private readonly List<Mesh> generatedMeshes = new List<Mesh>();
    private readonly Dictionary<string, LineRenderer> trafficLightRenderers = new Dictionary<string, LineRenderer>();
    private Material defaultMaterial, stripDisplayMaterial, markingDisplayMaterial;
    private Matrix4x4 generatedTransform;
    private bool dirty = true, renderedInPlayMode;

    private void OnEnable()
    {
        if (exportCamera == null) exportCamera = drawingCamera;
        drawingCamera = null;
        dirty = true;
    }
    private void OnValidate() { dirty = true; }
    private void OnDisable() { ClearGenerated(); ClearDisplayMaterials(); Release(defaultMaterial); defaultMaterial = null; }
    private void Update()
    {
        if (renderedInPlayMode != Application.isPlaying || generatedTransform != transform.localToWorldMatrix) dirty = true;
        if (dirty) RebuildVisuals();
        if (Application.isPlaying && VisibilityKeyPressed()) SetLaneVisibility(!showLanesInGame);
    }
    private static bool VisibilityKeyPressed()
    {
#if ENABLE_INPUT_SYSTEM
        return Keyboard.current != null && Keyboard.current.lKey.wasPressedThisFrame;
#elif ENABLE_LEGACY_INPUT_MANAGER
        return Input.GetKeyDown(KeyCode.L);
#else
        return false;
#endif
    }

    // ---------------------------------------------------------------- lanes and nodes

    public LaneNode FindNode(string id) => nodes.Find(node => node.id == id);
    public bool IsNodeInSelectedLane(string nodeId) => CurrentLane != null && CurrentLane.sections.Exists(s => s.nodeId == nodeId);
    public float LaneLength(LaneStrip lane)
    {
        float length = 0; LaneNode previous = null;
        foreach (Section section in lane.sections)
        {
            LaneNode node = FindNode(section.nodeId);
            if (node == null) continue;
            if (previous != null) length += Vector3.Distance(transform.TransformPoint(previous.position), transform.TransformPoint(node.position));
            previous = node;
        }
        return length;
    }
    /// <summary>Select a node, staying on the current lane when it passes through that node.</summary>
    public void SelectNode(string nodeId)
    {
        int section = CurrentLane == null ? -1 : CurrentLane.sections.FindIndex(s => s.nodeId == nodeId);
        if (section >= 0) { SelectedSection = section; return; }
        for (int i = 0; i < laneStrips.Count; i++)
        {
            section = laneStrips[i].sections.FindIndex(s => s.nodeId == nodeId);
            if (section >= 0) { SelectedLane = i; SelectedSection = section; return; }
        }
    }
    /// <summary>Add an empty lane and select it.</summary>
    public LaneStrip AddLane(float width = 3.2f)
    {
        LaneStrip lane = new LaneStrip { name = NextLaneName(), width = width };
        lane.Normalize(); laneStrips.Add(lane);
        SelectedLane = laneStrips.Count - 1; SelectedSection = -1; dirty = true;
        return lane;
    }
    /// <summary>Start a lane at an existing node. It takes the width of a lane already there.</summary>
    public LaneStrip StartLaneAt(string nodeId, float width = 3.2f)
    {
        if (FindNode(nodeId) == null) return null;
        LaneStrip neighbour = laneStrips.Find(l => l.sections.Exists(s => s.nodeId == nodeId));
        SplitLanesAtNode(nodeId);
        LaneStrip lane = AddLane(neighbour != null ? neighbour.width : width);
        lane.sections.Add(new Section { nodeId = nodeId }); SelectedSection = 0;
        Status = "Lane started at a shared node.";
        RebuildVisuals(); return lane;
    }
    /// <summary>Append a new node to the current lane.</summary>
    public bool AppendPoint(Vector3 worldPoint)
    {
        LaneStrip lane = CurrentLane;
        if (lane == null || !FinitePoint(worldPoint)) return false;
        if (lane.sections.Count > 0)
        {
            LaneNode last = FindNode(lane.sections[lane.sections.Count - 1].nodeId);
            if (last != null && Vector3.Distance(worldPoint, transform.TransformPoint(last.position)) < 0.1f) return false;
        }
        LaneNode node = new LaneNode { position = transform.InverseTransformPoint(worldPoint) };
        nodes.Add(node); lane.sections.Add(new Section { nodeId = node.id });
        SelectedSection = lane.sections.Count - 1;
        RebuildVisuals(); return true;
    }
    /// <summary>End the current lane on an existing node, joining the lanes that meet there.</summary>
    public bool AppendExistingNode(string nodeId)
    {
        LaneStrip lane = CurrentLane;
        if (lane == null || FindNode(nodeId) == null) return false;
        if (lane.sections.Exists(s => s.nodeId == nodeId)) { Status = "This lane already passes through that node. Use another lane to close a loop."; return false; }
        SplitLanesAtNode(nodeId);
        lane.sections.Add(new Section { nodeId = nodeId }); SelectedSection = lane.sections.Count - 1;
        Status = "Lanes joined at a shared node."; RebuildVisuals(); return true;
    }
    /// <summary>Move the first or last node of the current lane onto an existing node.</summary>
    public bool JoinEndpoint(bool start, string nodeId)
    {
        LaneStrip lane = CurrentLane;
        if (lane == null || lane.sections.Count < 2 || FindNode(nodeId) == null) return false;
        if (lane.sections.Exists(s => s.nodeId == nodeId)) { Status = "This lane already passes through that node."; return false; }
        SplitLanesAtNode(nodeId);
        SelectedSection = start ? 0 : lane.sections.Count - 1;
        lane.sections[SelectedSection].nodeId = nodeId;
        RemoveUnusedNodes(); Status = "Lanes joined at a shared node."; RebuildVisuals(); return true;
    }
    /// <summary>The node at a point along a lane segment: an end node when the point is within half a metre
    /// of it, otherwise a new node inserted there. The lane keeps its shape.</summary>
    public string NodeOnSegment(int laneIndex, int segment, float t)
    {
        if (laneIndex < 0 || laneIndex >= laneStrips.Count || !Finite(t)) return null;
        LaneStrip lane = laneStrips[laneIndex];
        if (segment < 0 || segment >= lane.sections.Count - 1) return null;
        Section first = lane.sections[segment], next = lane.sections[segment + 1];
        LaneNode a = FindNode(first.nodeId), b = FindNode(next.nodeId);
        if (a == null || b == null) return null;
        t = Mathf.Clamp01(t);
        float length = Vector3.Distance(transform.TransformPoint(a.position), transform.TransformPoint(b.position));
        if (t * length < 0.5f && t <= 0.5f) return a.id;
        if ((1 - t) * length < 0.5f) return b.id;
        RoadMeshGeometry.Frame[] frames = RoadMeshGeometry.Frames(this, lane);
        RoadMeshGeometry.Frame desired = InterpolatedFrame(frames[segment], frames[segment + 1], t);
        LaneNode node = new LaneNode { position = Vector3.Lerp(a.position, b.position, t) };
        nodes.Add(node);
        lane.sections.Insert(segment + 1, new Section { nodeId = node.id, widthScale = Mathf.Lerp(first.widthScale, next.widthScale, t) });
        PreserveFrame(lane, segment + 1, desired);
        if (laneIndex == SelectedLane && SelectedSection > segment) SelectedSection++;
        dirty = true;
        return node.id;
    }
    /// <summary>Insert a node halfway along a segment of the current lane and select it.</summary>
    public bool InsertNode(int segment)
    {
        LaneStrip lane = CurrentLane;
        string id = NodeOnSegment(SelectedLane, segment, 0.5f);
        if (id == null) return false;
        SelectedSection = lane.sections.FindIndex(s => s.nodeId == id);
        RebuildVisuals(); return true;
    }
    public bool MoveNode(string nodeId, Vector3 worldPosition)
    {
        LaneNode node = FindNode(nodeId);
        if (node == null || !FinitePoint(worldPosition)) return false;
        node.position = transform.InverseTransformPoint(worldPosition);
        RebuildVisuals(); return true;
    }
    /// <summary>Swap the travel direction of the current lane. Its edges stay where they are.</summary>
    public void ReverseSelectedLane()
    {
        LaneStrip lane = CurrentLane;
        if (lane == null || lane.sections.Count < 2) return;
        RoadMeshGeometry.Frame[] before = RoadMeshGeometry.Frames(this, lane);
        lane.sections.Reverse(); Array.Reverse(before);
        Quaternion turn = Quaternion.Euler(0, 180, 0);
        for (int i = 0; i < before.Length; i++)
            PreserveFrame(lane, i, new RoadMeshGeometry.Frame { left = before[i].right, right = before[i].left, rotation = before[i].rotation * turn });
        if (SelectedSection >= 0) SelectedSection = lane.sections.Count - 1 - SelectedSection;
        Status = "Lane direction reversed."; RebuildVisuals();
    }
    public void RemoveLastSection()
    {
        LaneStrip lane = CurrentLane;
        if (lane == null || lane.sections.Count == 0) return;
        lane.sections.RemoveAt(lane.sections.Count - 1);
        SelectedSection = lane.sections.Count - 1; RemoveUnusedNodes(); RebuildVisuals();
    }
    /// <summary>Remove the selected node from the current lane. A lane left with one node is deleted.</summary>
    public void DeleteSelectedNode()
    {
        LaneStrip lane = CurrentLane;
        if (lane == null || SelectedSection < 0 || SelectedSection >= lane.sections.Count) return;
        if (lane.sections.Count <= 2) { DeleteSelectedLane(); Status = "Lane deleted: a lane needs at least two nodes."; return; }
        lane.sections.RemoveAt(SelectedSection);
        SelectedSection = Mathf.Min(SelectedSection, lane.sections.Count - 1);
        RemoveUnusedNodes(); RebuildVisuals();
    }
    public void DeleteSelectedLane()
    {
        if (CurrentLane == null) return;
        laneStrips.RemoveAt(SelectedLane);
        SelectedLane = -1; SelectedSection = -1;
        RemoveUnusedNodes(); Status = "Lane deleted."; RebuildVisuals();
    }

    private string NextLaneName()
    {
        for (int number = laneStrips.Count + 1; ; number++)
        {
            string name = "Lane " + number;
            if (!laneStrips.Exists(l => l.name == name)) return name;
        }
    }
    // Lanes connect only at their ends, so a lane passing through a junction node is cut in two there.
    private void SplitLanesAtNode(string nodeId)
    {
        int count = laneStrips.Count;
        for (int i = 0; i < count; i++)
        {
            LaneStrip lane = laneStrips[i]; int at = lane.sections.FindIndex(s => s.nodeId == nodeId);
            if (at <= 0 || at >= lane.sections.Count - 1) continue;
            RoadMeshGeometry.Frame cut = RoadMeshGeometry.Frames(this, lane)[at];
            LaneStrip rest = new LaneStrip { name = NextLaneName(), width = lane.width };
            for (int j = at; j < lane.sections.Count; j++)
                rest.sections.Add(new Section { nodeId = lane.sections[j].nodeId, widthScale = lane.sections[j].widthScale, rotation = lane.sections[j].rotation });
            lane.sections.RemoveRange(at + 1, lane.sections.Count - at - 1); laneStrips.Add(rest);
            PreserveFrame(lane, lane.sections.Count - 1, cut);
            PreserveFrame(rest, 0, cut);
            if (i == SelectedLane && SelectedSection > at) SelectedSection = at;
        }
    }
    // Set a section's rotation and width so that its cross-section matches the given one.
    private void PreserveFrame(LaneStrip lane, int sectionIndex, RoadMeshGeometry.Frame desired)
    {
        RoadMeshGeometry.Frame frame = RoadMeshGeometry.Frames(this, lane)[sectionIndex];
        Section section = lane.sections[sectionIndex];
        section.rotation = (Quaternion.Inverse(frame.automaticRotation) * desired.rotation).eulerAngles;
        section.widthScale = Mathf.Clamp(section.widthScale * Vector3.Distance(desired.left, desired.right) /
            Mathf.Max(0.001f, Vector3.Distance(frame.left, frame.right)), 0.1f, 10);
    }
    private static RoadMeshGeometry.Frame InterpolatedFrame(RoadMeshGeometry.Frame a, RoadMeshGeometry.Frame b, float t)
    {
        Vector3 left = Vector3.Lerp(a.left, b.left, t), right = Vector3.Lerp(a.right, b.right, t);
        Vector3 side = (right - left).normalized;
        Quaternion interpolated = Quaternion.Slerp(a.rotation, b.rotation, t);
        Vector3 up = Vector3.ProjectOnPlane(interpolated * Vector3.up, side).normalized;
        Quaternion rotation = side.sqrMagnitude > 0.001f && up.sqrMagnitude > 0.001f
            ? Quaternion.LookRotation(Vector3.Cross(side, up).normalized, up) : interpolated;
        return new RoadMeshGeometry.Frame { left = left, right = right, rotation = rotation };
    }
    private void RemoveUnusedNodes() => nodes.RemoveAll(n => !laneStrips.Exists(l => l.sections.Exists(s => s.nodeId == n.id)));

    // ---------------------------------------------------------------- traffic light lines

    public void BeginTrafficLightDrawing()
    {
        IsDrawingTrafficLight = true; HasTrafficLightStart = false;
        Status = "Click the start of the stop line.";
    }
    public void CancelTrafficLightDrawing() { IsDrawingTrafficLight = false; HasTrafficLightStart = false; }
    public bool PlaceTrafficLightPoint(Vector3 worldPoint)
    {
        if (!IsDrawingTrafficLight || !FinitePoint(worldPoint)) return false;
        if (!HasTrafficLightStart)
        {
            trafficLightStart = transform.InverseTransformPoint(worldPoint); HasTrafficLightStart = true;
            Status = "Click the end of the stop line."; return true;
        }
        if (!ValidTrafficLightSpan(TrafficLightStartWorld, worldPoint))
        { Status = "Stop line endpoints must be at least 0.1 m apart."; return false; }
        trafficLights.Add(new TrafficLightLine { name = "Traffic light " + (trafficLights.Count + 1), start = trafficLightStart, end = transform.InverseTransformPoint(worldPoint) });
        SelectedTrafficLight = trafficLights.Count - 1; CancelTrafficLightDrawing(); dirty = true;
        Status = "Traffic light line added."; return true;
    }
    public bool MoveTrafficLightEndpoint(int index, bool start, Vector3 worldPoint)
    {
        if (index < 0 || index >= trafficLights.Count || !FinitePoint(worldPoint)) return false;
        TrafficLightLine line = trafficLights[index];
        if (!ValidTrafficLightSpan(worldPoint, transform.TransformPoint(start ? line.end : line.start))) return false;
        if (start) line.start = transform.InverseTransformPoint(worldPoint); else line.end = transform.InverseTransformPoint(worldPoint);
        dirty = true; return true;
    }
    public void DeleteSelectedTrafficLight()
    {
        if (CurrentTrafficLight == null) return;
        trafficLights.RemoveAt(SelectedTrafficLight); SelectedTrafficLight = -1;
        CancelTrafficLightDrawing(); dirty = true; Status = "Traffic light line deleted.";
    }
    /// <summary>Show every line's red or green aspect at this time. Edit mode shows time 0.</summary>
    public void SetTrafficLightDisplayTime(double time)
    {
        TrafficLightDisplayTime = time;
        foreach (TrafficLightLine light in trafficLights)
            if (trafficLightRenderers.TryGetValue(light.id, out LineRenderer line) && line != null)
                Tint(line, light.AspectAt(time) == "red" ? Color.red : Color.green);
    }
    private static bool ValidTrafficLightSpan(Vector3 start, Vector3 end) => FinitePoint(start) && FinitePoint(end) && (end - start).sqrMagnitude >= 0.01f;

    // ---------------------------------------------------------------- road surface

    /// <summary>The nearest point of the road model under a ray. Generated lane meshes are ignored.</summary>
    public bool TryGetSurfacePoint(Ray ray, out Vector3 point)
    {
        Physics.SyncTransforms(); point = default; float nearest = float.PositiveInfinity; bool found = false;
        foreach (RaycastHit hit in Physics.RaycastAll(ray, Mathf.Infinity, surfaceLayers, QueryTriggerInteraction.Ignore))
        {
            if (generatedRoot != null && hit.transform.IsChildOf(generatedRoot.transform)) continue;
            if (surfaceRoot != null && !hit.transform.IsChildOf(surfaceRoot)) continue;
            if (hit.distance >= nearest) continue;
            nearest = hit.distance; point = hit.point; found = true;
        }
        return found;
    }
    public int PrepareSurfaceColliders(Func<GameObject, MeshCollider> addCollider = null)
    {
        Transform root = surfaceRoot != null ? surfaceRoot : transform; int added = 0;
        foreach (MeshFilter filter in root.GetComponentsInChildren<MeshFilter>(true))
        {
            if (generatedRoot != null && filter.transform.IsChildOf(generatedRoot.transform)) continue;
            if (filter.sharedMesh == null || (surfaceLayers.value & (1 << filter.gameObject.layer)) == 0) continue;
            bool exists = false;
            foreach (Collider collider in filter.GetComponents<Collider>()) if (collider.enabled && !collider.isTrigger) exists = true;
            Rigidbody body = filter.GetComponentInParent<Rigidbody>();
            if (exists || (body != null && !body.isKinematic)) continue;
            MeshCollider created = addCollider != null ? addCollider(filter.gameObject) : filter.gameObject.AddComponent<MeshCollider>();
            created.convex = false; created.sharedMesh = filter.sharedMesh; added++;
        }
        Physics.SyncTransforms();
        if (added > 0) Status = "Added " + added + " mesh colliders to the road model.";
        return added;
    }

    // ---------------------------------------------------------------- display

    public void RebuildVisuals()
    {
        dirty = false; if (!isActiveAndEnabled) return;
        renderedInPlayMode = Application.isPlaying;
        generatedTransform = transform.localToWorldMatrix;
        foreach (LaneStrip lane in laneStrips) lane.Normalize();
        ClearGenerated(); ClearDisplayMaterials();
        // A rebuildable cache: it is never written into the scene file.
        generatedRoot = new GameObject(GeneratedName) { hideFlags = HideFlags.DontSave };
        generatedRoot.transform.SetParent(transform, false);
        generatedRoot.transform.localPosition = transform.InverseTransformVector(Vector3.up * surfaceOffset);
        Material fallback = DefaultMaterial();
        Material fillMaterial = stripMaterial != null ? stripMaterial : fallback;
        Material markingMaterial = laneMaterial != null ? laneMaterial : fallback;
        if (drawAboveSurface)
        {
            Shader overlayShader = Resources.Load<Shader>("LaneEditor/LaneOverlay");
            if (overlayShader != null)
            {
                stripDisplayMaterial = CreateOverlayMaterial(overlayShader, stripMaterial, stripColor, 3010);
                markingDisplayMaterial = CreateOverlayMaterial(overlayShader, laneMaterial, Color.white, 3011);
                markingDisplayMaterial.SetFloat("_UseVertexColor", 1);
                fillMaterial = stripDisplayMaterial;
                markingMaterial = markingDisplayMaterial;
            }
            else Debug.LogWarning("Lane overlay shader is missing. Import Assets/Resources/LaneEditor/LaneOverlay.shader and rebuild lane meshes.", this);
        }
        Dictionary<string, List<Vector3>> junctionEdges = new Dictionary<string, List<Vector3>>();
        foreach (LaneStrip lane in laneStrips)
        {
            if (lane.sections.Count < 2) continue;
            RoadMeshGeometry.Frame[] frames = RoadMeshGeometry.Frames(this, lane);
            Vector3[] left = new Vector3[frames.Length], right = new Vector3[frames.Length];
            for (int i = 0; i < frames.Length; i++)
            {
                left[i] = frames[i].left; right[i] = frames[i].right;
                if (!showFill) continue;
                string id = lane.sections[i].nodeId;
                if (!junctionEdges.TryGetValue(id, out List<Vector3> edges)) { edges = new List<Vector3>(); junctionEdges[id] = edges; }
                edges.Add(left[i]); edges.Add(right[i]);
            }
            if (showFill) CreateFill(lane.name, RoadMeshGeometry.Strip(transform, frames), fillMaterial);
            CreateLine("Lane edge", left, Color.white, lineWidth, markingMaterial);
            CreateLine("Lane edge", right, Color.white, lineWidth, markingMaterial);
            for (int i = 1; i < frames.Length; i++)
            {
                Vector3 a = frames[i - 1].center, b = frames[i].center;
                Arrow((a + b) * 0.5f, (b - a).normalized, Mathf.Min(arrowSize, Vector3.Distance(a, b) * 0.25f), markingMaterial);
            }
        }
        // Lanes meeting at a node leave gaps between their ends; one patch per node closes them.
        foreach (List<Vector3> edges in junctionEdges.Values) CreateFill("Junction", RoadMeshGeometry.Junction(transform, edges), fillMaterial);
        foreach (TrafficLightLine light in trafficLights)
            trafficLightRenderers[light.id] = CreateLine("Traffic light " + light.id, new[] { transform.TransformPoint(light.start), transform.TransformPoint(light.end) },
                light.AspectAt(Application.isPlaying ? TrafficLightDisplayTime : 0) == "red" ? Color.red : Color.green, lineWidth * 2, markingMaterial);
        SetLaneVisibility(showLanesInGame);
    }
    public void SetLaneVisibility(bool visible)
    {
        showLanesInGame = visible;
        if (generatedRoot == null) return;
        foreach (Renderer renderer in generatedRoot.GetComponentsInChildren<Renderer>()) renderer.enabled = !Application.isPlaying || visible;
    }
    private Material DefaultMaterial()
    {
        if (defaultMaterial != null) return defaultMaterial;
        Shader shader = Shader.Find("Universal Render Pipeline/Unlit") ?? Shader.Find("Sprites/Default");
        if (shader != null) defaultMaterial = new Material(shader) { hideFlags = HideFlags.HideAndDontSave };
        return defaultMaterial;
    }
    private static Material CreateOverlayMaterial(Shader shader, Material source, Color color, int renderQueue)
    {
        Material overlay = new Material(shader) { name = "Lane display overlay", hideFlags = HideFlags.HideAndDontSave, renderQueue = renderQueue };
        if (source != null)
        {
            if (source.HasProperty("_BaseColor")) color = source.GetColor("_BaseColor");
            else if (source.HasProperty("_Color")) color = source.GetColor("_Color");
            string textureProperty = source.HasProperty("_BaseMap") ? "_BaseMap" : source.HasProperty("_MainTex") ? "_MainTex" : null;
            if (textureProperty != null)
            {
                overlay.SetTexture("_BaseMap", source.GetTexture(textureProperty));
                overlay.SetTextureScale("_BaseMap", source.GetTextureScale(textureProperty));
                overlay.SetTextureOffset("_BaseMap", source.GetTextureOffset(textureProperty));
            }
        }
        overlay.SetColor("_BaseColor", color);
        return overlay;
    }
    private GameObject NewChild(string name)
    {
        GameObject child = new GameObject(name) { hideFlags = HideFlags.DontSave };
        child.transform.SetParent(generatedRoot.transform, false);
        return child;
    }
    private void CreateFill(string name, Mesh mesh, Material material)
    {
        if (mesh == null) return;
        generatedMeshes.Add(mesh);
        GameObject child = NewChild(name);
        child.AddComponent<MeshFilter>().sharedMesh = mesh;
        MeshRenderer renderer = child.AddComponent<MeshRenderer>(); renderer.sharedMaterial = material;
        if (stripMaterial == null) Tint(renderer, stripColor);
        if (drawAboveSurface)
        {
            renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            renderer.allowOcclusionWhenDynamic = false;
        }
        HideSelectionOutline(renderer);
    }
    private LineRenderer CreateLine(string name, Vector3[] worldPoints, Color color, float width, Material material)
    {
        LineRenderer line = NewChild(name).AddComponent<LineRenderer>(); line.useWorldSpace = false;
        line.sharedMaterial = material; Tint(line, color);
        if (drawAboveSurface) line.allowOcclusionWhenDynamic = false;
        line.startWidth = line.endWidth = width; line.numCornerVertices = 2; line.positionCount = worldPoints.Length;
        Vector3[] points = new Vector3[worldPoints.Length];
        for (int i = 0; i < points.Length; i++) points[i] = transform.InverseTransformPoint(worldPoints[i] + Vector3.up * 0.015f);
        line.SetPositions(points); line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        HideSelectionOutline(line);
        return line;
    }
    private void Arrow(Vector3 center, Vector3 direction, float size, Material material)
    {
        Vector3 side = Vector3.Cross(Vector3.up, direction).normalized * size * 0.45f;
        CreateLine("Direction", new[] { center - direction * size * 0.5f + side, center + direction * size * 0.5f, center - direction * size * 0.5f - side }, Color.yellow, lineWidth, material);
    }
    private static void Tint(Renderer renderer, Color color)
    {
        MaterialPropertyBlock block = new MaterialPropertyBlock(); block.SetColor("_BaseColor", color); block.SetColor("_Color", color); renderer.SetPropertyBlock(block);
    }
    // The Scene view highlights the selection itself; Unity's outline around every generated line only hides it.
    private static void HideSelectionOutline(Renderer renderer)
    {
#if UNITY_EDITOR
        UnityEditor.EditorUtility.SetSelectedRenderState(renderer, UnityEditor.EditorSelectedRenderState.Hidden);
#endif
    }
    private void ClearDisplayMaterials()
    {
        Release(stripDisplayMaterial); Release(markingDisplayMaterial);
        stripDisplayMaterial = markingDisplayMaterial = null;
    }
    private void ClearGenerated()
    {
        trafficLightRenderers.Clear();
        foreach (Mesh mesh in generatedMeshes) Release(mesh);
        generatedMeshes.Clear(); generatedRoot = null;
        // By name, so a copy that an older version saved into the scene is removed as well.
        for (int i = transform.childCount - 1; i >= 0; i--)
        {
            GameObject child = transform.GetChild(i).gameObject;
            if (child.name != GeneratedName) continue;
            child.SetActive(false); Release(child);
        }
    }
    private static void Release(UnityEngine.Object value)
    {
        if (value == null) return;
        if (Application.isPlaying) Destroy(value); else DestroyImmediate(value);
    }
    private void OnDrawGizmos()
    {
        if (Application.isPlaying && !showLanesInGame) return;
        foreach (LaneNode node in nodes)
        {
            Vector3 point = transform.TransformPoint(node.position) + Vector3.up * surfaceOffset;
            Gizmos.color = node.isDeadEnd ? Color.red : node.isSpawnPoint ? Color.green : IsNodeInSelectedLane(node.id) ? Color.yellow : Color.cyan;
            if (node.isDeadEnd) Gizmos.DrawCube(point, Vector3.one * 0.2f * NodeMarkerSize);
            else Gizmos.DrawSphere(point, 0.12f * NodeMarkerSize);
            if (node.isDeadEnd && node.isSpawnPoint)
            {
                Gizmos.color = Color.green;
                Gizmos.DrawWireSphere(point, 0.2f * NodeMarkerSize);
            }
        }
    }

    // ---------------------------------------------------------------- JSON backup

    public string ToJson()
    {
        LaneFile file = new LaneFile { nodes = nodes, laneStrips = laneStrips, trafficLights = trafficLights };
        ValidateFile(file);
        return JsonUtility.ToJson(file, true);
    }
    /// <summary>Replace the network with one written by ToJson. Throws on an invalid file and changes nothing.</summary>
    public void LoadJson(string json)
    {
        LaneFile file = JsonUtility.FromJson<LaneFile>(json);
        ValidateFile(file);
        nodes = file.nodes; laneStrips = file.laneStrips; trafficLights = file.trafficLights;
        SelectedLane = SelectedSection = SelectedTrafficLight = -1; CancelTrafficLightDrawing();
        Status = "Loaded " + laneStrips.Count + " lanes."; RebuildVisuals();
    }
    private static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
    private static bool FinitePoint(Vector3 point) => Finite(point.x) && Finite(point.y) && Finite(point.z);
    private static void ValidateFile(LaneFile file)
    {
        if (file == null || file.version != 3 || file.nodes == null || file.laneStrips == null) throw new IOException("Unsupported lane file: version 3 with nodes and laneStrips is required.");
        // Additive v3 field: older networks have no traffic light lines.
        if (file.trafficLights == null) file.trafficLights = new List<TrafficLightLine>();
        HashSet<string> lightIds = new HashSet<string>();
        foreach (TrafficLightLine light in file.trafficLights)
            // Stored endpoints are local; a scaled editor can have a valid world-space
            // line whose local length is below the interactive 0.1 m drawing minimum.
            if (light == null || string.IsNullOrWhiteSpace(light.id) || !lightIds.Add(light.id) ||
                !FinitePoint(light.start) || !FinitePoint(light.end) || (light.end - light.start).sqrMagnitude <= 0 ||
                !TrafficLightLine.ValidTiming(light.redDurationS, light.greenDurationS, light.cycleOffsetS))
                throw new IOException("Invalid traffic light line: unique ID, distinct finite endpoints and nonnegative finite durations are required.");
        HashSet<string> nodeIds = new HashSet<string>(), laneIds = new HashSet<string>();
        foreach (LaneNode node in file.nodes)
            if (node == null || string.IsNullOrEmpty(node.id) || !nodeIds.Add(node.id) || !FinitePoint(node.position)) throw new IOException("Invalid node.");
        foreach (LaneStrip lane in file.laneStrips)
        {
            if (lane == null || string.IsNullOrEmpty(lane.id) || !laneIds.Add(lane.id) || lane.sections == null || !Finite(lane.width) || lane.width <= 0) throw new IOException("Invalid lane.");
            HashSet<string> visited = new HashSet<string>();
            foreach (Section section in lane.sections)
                if (section == null || !nodeIds.Contains(section.nodeId) || !visited.Add(section.nodeId) || !Finite(section.widthScale) || section.widthScale <= 0 ||
                    !FinitePoint(section.rotation)) throw new IOException("Invalid section.");
        }
    }
}
