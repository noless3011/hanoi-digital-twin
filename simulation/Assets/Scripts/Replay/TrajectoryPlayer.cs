using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Plays back a vehicle-trajectories JSON: simulation output from the Python core (twin/) or video tracks.
/// Unity does not simulate; it only shows where each vehicle was at each time.
/// world_position is Unity x, y, z metres. Heading comes from the file when present, otherwise from motion.
/// </summary>
public sealed class TrajectoryPlayer : MonoBehaviour
{
    [Tooltip("Trajectories file. Used before Trajectory Path when set.")]
    public TextAsset trajectoryFile;
    [Tooltip("JSON file relative to StreamingAssets, or an absolute path.")]
    public string trajectoryPath = "Traffic/simulation.trajectories.json";
    public bool playOnStart = true;
    public bool loop = true;
    [Min(0f)] public float playbackSpeed = 1f;
    [Tooltip("Extra yaw applied to every vehicle model, if its mesh does not face +Z.")]
    public float modelYawOffset;
    [Tooltip("Lift above the recorded ground height, in metres.")]
    public float heightOffset;

    public double Time { get; private set; }
    public double Duration { get; private set; }
    public bool Playing { get; set; }
    public int VehicleCount => vehicles.Count;
    public string Status { get; private set; } = "Nothing loaded";

    [Serializable] private class Observation { public double time_s; public float[] world_position; public float[] heading; }
    [Serializable] private class VehicleData { public long track_id; public string kind; public Observation[] observations; }
    [Serializable] private class FileData { public string schema; public string source; public double duration_s; public VehicleData[] vehicles; }

    private sealed class Vehicle
    {
        public string kind;
        public double[] t;
        public Vector3[] position;
        public Vector2[] heading;   // x, z; zero when unknown
        public GameObject body;
    }

    private static readonly Dictionary<string, string> ModelPaths = new Dictionary<string, string> {
        { "car", "Models/Vehicles/sedan" }, { "van", "Models/Vehicles/sedan" }, { "bus", "Models/Vehicles/bus" },
        { "truck", "Models/Vehicles/bus" }, { "motorcycle", "Models/Vehicles/motorcycle" }, { "bicycle", "Models/Vehicles/motorcycle" } };
    private static readonly Dictionary<string, Vector3> Sizes = new Dictionary<string, Vector3> {
        { "motorcycle", new Vector3(0.75f, 1.2f, 1.9f) }, { "bicycle", new Vector3(0.6f, 1.2f, 1.7f) },
        { "car", new Vector3(1.8f, 1.5f, 4.4f) }, { "van", new Vector3(1.95f, 2f, 5f) },
        { "truck", new Vector3(2.3f, 2.8f, 7f) }, { "bus", new Vector3(2.5f, 3f, 10.5f) } };

    private readonly List<Vehicle> vehicles = new List<Vehicle>();
    private Transform root;

    private void Start()
    {
        if (!playOnStart) return;
        try { Load(); Playing = true; }
        catch (Exception e) { Status = "Load failed: " + e.Message; Debug.LogError(Status); }
    }

    public void Load()
    {
        string json = trajectoryFile != null ? trajectoryFile.text : File.ReadAllText(Path.IsPathRooted(trajectoryPath)
            ? trajectoryPath : Path.Combine(Application.streamingAssetsPath, trajectoryPath));
        LoadJson(json);
    }

    public void LoadJson(string json)
    {
        var data = JsonUtility.FromJson<FileData>(json);
        if (data == null || data.schema != "vehicle-trajectories" || data.vehicles == null)
            throw new FormatException("Not a vehicle-trajectories file.");
        Clear();
        root = new GameObject("Trajectory vehicles").transform;
        root.SetParent(transform, false);
        double end = data.duration_s;
        foreach (var v in data.vehicles)
        {
            if (v.observations == null || v.observations.Length == 0) continue;
            int n = v.observations.Length;
            var vehicle = new Vehicle { kind = v.kind ?? "car", t = new double[n], position = new Vector3[n], heading = new Vector2[n] };
            int kept = 0;
            for (int i = 0; i < n; i++)
            {
                var o = v.observations[i];
                if (o.world_position == null || o.world_position.Length < 3) continue;
                vehicle.t[kept] = o.time_s;
                vehicle.position[kept] = new Vector3(o.world_position[0], o.world_position[1] + heightOffset, o.world_position[2]);
                vehicle.heading[kept] = o.heading != null && o.heading.Length >= 2 ? new Vector2(o.heading[0], o.heading[1]) : Vector2.zero;
                kept++;
            }
            if (kept == 0) continue;
            Array.Resize(ref vehicle.t, kept); Array.Resize(ref vehicle.position, kept); Array.Resize(ref vehicle.heading, kept);
            FillMissingHeadings(vehicle);
            vehicle.body = CreateBody(vehicle.kind, v.track_id);
            vehicle.body.SetActive(false);
            vehicles.Add(vehicle);
            end = Math.Max(end, vehicle.t[kept - 1]);
        }
        Duration = end;
        Time = 0;
        Status = "Loaded " + vehicles.Count + " vehicles, " + Duration.ToString("F1") + " s (" + (data.source ?? "video") + ")";
        Apply();
    }

    public void Clear()
    {
        foreach (var v in vehicles) if (v.body != null) Destroy(v.body);
        vehicles.Clear();
        if (root != null) Destroy(root.gameObject);
        root = null;
    }

    public void Seek(double time) { Time = Math.Max(0, Math.Min(time, Duration)); Apply(); }

    private void Update()
    {
        if (!Playing || vehicles.Count == 0) return;
        Time += UnityEngine.Time.deltaTime * playbackSpeed;
        if (Time > Duration) Time = loop ? 0 : Duration;
        Apply();
    }

    private void Apply()
    {
        foreach (var v in vehicles)
        {
            bool visible = Time >= v.t[0] && Time <= v.t[v.t.Length - 1];
            if (v.body.activeSelf != visible) v.body.SetActive(visible);
            if (!visible) continue;
            int k = Array.BinarySearch(v.t, Time);
            if (k < 0) k = ~k - 1;
            k = Mathf.Clamp(k, 0, v.t.Length - 1);
            int k1 = Math.Min(k + 1, v.t.Length - 1);
            float f = k1 == k ? 0f : (float)((Time - v.t[k]) / (v.t[k1] - v.t[k]));
            v.body.transform.position = Vector3.Lerp(v.position[k], v.position[k1], f);
            Vector2 h = Vector2.Lerp(v.heading[k], v.heading[k1], f);
            if (h.sqrMagnitude > 1e-6f)
                v.body.transform.rotation = Quaternion.LookRotation(new Vector3(h.x, 0, h.y)) * Quaternion.Euler(0, modelYawOffset, 0);
        }
    }

    private static void FillMissingHeadings(Vehicle v)
    {
        Vector2 last = Vector2.zero;
        for (int i = 0; i < v.t.Length; i++)
        {
            if (v.heading[i].sqrMagnitude > 1e-6f) { last = v.heading[i]; continue; }
            int j = Math.Min(i + 1, v.t.Length - 1), a = Math.Max(i - 1, 0);
            Vector3 d = v.position[j] - v.position[a];
            var move = new Vector2(d.x, d.z);
            if (move.magnitude > 0.05f) last = move.normalized;
            v.heading[i] = last;
        }
    }

    private GameObject CreateBody(string kind, long id)
    {
        GameObject body = null;
        if (ModelPaths.TryGetValue(kind, out string path))
        {
            var model = Resources.Load<GameObject>(path);
            if (model != null) body = Instantiate(model);
        }
        if (body == null)
        {
            // Box placeholder resting on the ground point.
            body = new GameObject();
            var box = GameObject.CreatePrimitive(PrimitiveType.Cube);
            var collider = box.GetComponent<Collider>();
            if (collider != null) Destroy(collider);
            Vector3 size = Sizes.TryGetValue(kind, out var s) ? s : Sizes["car"];
            box.transform.SetParent(body.transform, false);
            box.transform.localScale = size;
            box.transform.localPosition = new Vector3(0, size.y / 2, 0);
        }
        body.name = kind + " " + id;
        body.transform.SetParent(root, false);
        var target = body.AddComponent<CameraTrackingTarget>();
        target.kind = kind;
        return body;
    }
}
