using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

/// <summary>Projects the Lane Editor's lanes through a feed-aligned camera.
/// The file holds the camera calibration and the road layout that the Python twin builds its routes from.
/// Image points are normalized, with x right and y down, as in the video exports.</summary>
public static class CameraLaneExport
{
    public const string Schema = "camera-lane-layout";
    public const int Version = 2;
    private const float SampleStepM = 0.5f;
    private const float MaxGroundDistanceM = 300f;
    private const float EndNodeRadiusM = 5f;

    /// <summary>The Lane Editor's Export Camera, else the main camera, or the scene's only camera.</summary>
    public static Camera ResolveCamera(LaneEditor editor)
    {
        if (editor == null) return null;
        if (editor.exportCamera != null) return editor.exportCamera;
        if (Camera.main != null) return Camera.main;
        var cameras = UnityEngine.Object.FindObjectsByType<Camera>(FindObjectsSortMode.None)
            .Where(c => c.gameObject.scene == editor.gameObject.scene).ToArray();
        return cameras.Length == 1 ? cameras[0] : null;
    }

    public static string DefaultPath(LaneEditor editor)
    {
        string name = string.IsNullOrWhiteSpace(editor.exportFileName) ? editor.gameObject.scene.name + ".camera-lanes.json" : editor.exportFileName.Trim();
        return Path.Combine(Application.streamingAssetsPath, "Traffic", Path.GetFileName(name));
    }

    public static Dictionary<string, object> Build(LaneEditor editor, Camera camera, int imageWidth, int imageHeight)
    {
        if (editor == null) throw new ArgumentNullException(nameof(editor));
        if (camera == null) throw new InvalidOperationException("No camera found. Assign Export Camera on the Lane Editor.");
        if (camera.orthographic) throw new InvalidOperationException("The export camera must use a perspective projection.");
        if (imageWidth <= 0 || imageHeight <= 0) throw new InvalidOperationException("Export image size must be positive.");
        var lanes = editor.LaneStrips.Where(l => l != null && l.sections != null && l.sections.Count >= 2).ToList();
        if (lanes.Count == 0) throw new InvalidOperationException("The Lane Editor has no lane with at least two nodes.");

        // Feed coordinates refer to the complete video image, not the editor Game view.
        var projection = new FeedProjection(camera, imageWidth, imageHeight);
        var frames = lanes.Select(l => RoadMeshGeometry.Frames(editor, l)).ToList();
        float groundY = frames.SelectMany(f => f).Average(f => f.center.y);
        var calibration = GroundCalibration(projection, groundY);
        if (calibration.imagePoints.Count < 4) throw new InvalidOperationException("Too little of the lane plane is in view to calibrate the camera.");
        var nodes = editor.Nodes.Select(n =>
        {
            Vector3 world = editor.transform.TransformPoint(n.position);
            return (object)TrafficJson.Obj("id", n.id, "spawn_point", n.isSpawnPoint, "dead_end", n.isDeadEnd,
                "name", n.isSpawnPoint ? n.spawnPointName ?? "" : n.isDeadEnd ? n.deadEndName ?? "" : "",
                "world", Vec(world), "image", Image(projection, world));
        }).ToList();
        return TrafficJson.Obj(
            "schema", Schema, "version", Version,
            "scene", editor.gameObject.scene.name, "lane_manager", editor.gameObject.name,
            "exported_utc", DateTime.UtcNow.ToString("o"),
            "image", TrafficJson.Obj("width", imageWidth, "height", imageHeight, "coordinates", "normalized, x right, y down"),
            "camera", CameraOut(camera, projection),
            "ground", TrafficJson.Obj("model", "plane at the mean lane height", "y", (double)groundY),
            // These top-level keys match perception/vehicle_endpoints.py Calibration, so the file can also be
            // passed to export_vehicle_endpoints.py as --calibration.
            "ground_y", (double)groundY,
            "image_points", calibration.imagePoints, "world_points", calibration.worldPoints,
            "end_nodes", editor.Nodes.Where(n => n.isDeadEnd).Select(n => (object)TrafficJson.Obj(
                "id", n.id, "position", Vec(editor.transform.TransformPoint(n.position)), "radius_m", (double)EndNodeRadiusM)).ToList(),
            "nodes", nodes,
            "lanes", lanes.Select((l, i) => (object)LaneOut(l, frames[i], projection)).ToList());
    }

    public static void Write(string path, Dictionary<string, object> layout)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, TrafficJson.Encode(layout));
    }

    private static Dictionary<string, object> CameraOut(Camera camera, FeedProjection projection)
    {
        var t = camera.transform;
        Matrix4x4 m = projection.WorldToClip;
        var result = TrafficJson.Obj("name", camera.name, "position", Vec(t.position),
            "rotation_xyzw", new[] { t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w },
            "vertical_fov_deg", (double)camera.fieldOfView, "near_m", (double)camera.nearClipPlane,
            "physical", camera.usePhysicalProperties,
            // Exactly the projection the export and the simulation recorder use, at the feed's aspect.
            // Row-major; image x = (clip.x / clip.w + 1) / 2, image y (down) = 1 - (clip.y / clip.w + 1) / 2.
            "world_to_clip", new[] { m.m00, m.m01, m.m02, m.m03, m.m10, m.m11, m.m12, m.m13,
                                     m.m20, m.m21, m.m22, m.m23, m.m30, m.m31, m.m32, m.m33 });
        if (camera.usePhysicalProperties)
        {
            result["focal_length_mm"] = (double)camera.focalLength;
            result["sensor_size_mm"] = new[] { camera.sensorSize.x, camera.sensorSize.y };
            result["lens_shift"] = new[] { camera.lensShift.x, camera.lensShift.y };
            result["gate_fit"] = camera.gateFit.ToString();
        }
        return result;
    }

    private static Dictionary<string, object> LaneOut(LaneEditor.LaneStrip lane, RoadMeshGeometry.Frame[] frames, FeedProjection projection)
    {
        var samples = new List<object>();
        float station = 0;
        for (int section = 0; section < frames.Length - 1; section++)
        {
            var a = frames[section]; var b = frames[section + 1];
            float length = Vector3.Distance(a.center, b.center);
            int steps = Mathf.Max(1, Mathf.CeilToInt(length / SampleStepM));
            for (int step = 0; step <= steps; step++)
            {
                if (step == 0 && section > 0) continue; // Shared with the previous section's last sample.
                float f = (float)step / steps;
                Vector3 left = Vector3.Lerp(a.left, b.left, f), right = Vector3.Lerp(a.right, b.right, f);
                samples.Add(SampleOut(station + f * length, section, left, right, projection));
            }
            station += length;
        }
        // Traffic always runs from the first section to the last; left and right are as seen that way.
        return TrafficJson.Obj("id", lane.id, "name", lane.name, "direction", "Forward",
            "width_m", (double)lane.width, "length_m", (double)station,
            "sections", frames.Select(f => (object)TrafficJson.Obj("center", Vec(f.center), "left", Vec(f.left), "right", Vec(f.right))).ToList(),
            "samples", samples);
    }

    private static Dictionary<string, object> SampleOut(float station, int section, Vector3 left, Vector3 right, FeedProjection projection)
    {
        Vector2? leftImage = projection.WorldToImage(left), rightImage = projection.WorldToImage(right);
        var pixels = new Vector2(projection.width, projection.height);
        Vector3 center = (left + right) / 2;
        return TrafficJson.Obj("station_m", (double)station, "section", section,
            "center", Vec(center), "left", Vec(left), "right", Vec(right),
            "distance_m", (double)Vector3.Distance(projection.position, center),
            "in_view", InImage(leftImage) || InImage(rightImage),
            // Lane width here in video pixels; 0 when an edge is behind the camera.
            "width_px", leftImage.HasValue && rightImage.HasValue
                ? (double)Vector2.Distance(Vector2.Scale(leftImage.Value, pixels), Vector2.Scale(rightImage.Value, pixels)) : 0d,
            // The left and right edge in the image; null for an edge behind the camera.
            "boundaries_image", new List<object> { ImagePoint(leftImage), ImagePoint(rightImage) });
    }

    private static bool InImage(Vector2? p) => p.HasValue && p.Value.x >= 0 && p.Value.x <= 1 && p.Value.y >= 0 && p.Value.y <= 1;
    private static object ImagePoint(Vector2? p) => p.HasValue ? new[] { p.Value.x, p.Value.y } : null;

    private static (List<object> imagePoints, List<object> worldPoints) GroundCalibration(FeedProjection projection, float groundY)
    {
        var image = new List<object>(); var world = new List<object>();
        var plane = new Plane(Vector3.up, new Vector3(0, groundY, 0));
        for (int row = 0; row <= 8; row++)
            for (int column = 0; column <= 8; column++)
            {
                float u = column / 8f, v = row / 8f;
                Ray ray = projection.ViewportToRay(u, 1 - v);
                if (!plane.Raycast(ray, out float distance) || distance > MaxGroundDistanceM) continue;
                image.Add(new[] { u, v }); world.Add(Vec(ray.GetPoint(distance)));
            }
        return (image, world);
    }

    private static object Image(FeedProjection projection, Vector3 world) => ImagePoint(projection.WorldToImage(world));

    private static float[] Vec(Vector3 value) => new[] { value.x, value.y, value.z };
}
