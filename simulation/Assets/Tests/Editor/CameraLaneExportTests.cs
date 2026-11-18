#if UNITY_INCLUDE_TESTS
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

public sealed class CameraLaneExportTests
{
    private readonly List<Object> created = new List<Object>();
    [TearDown] public void TearDown() { foreach (var item in created) if (item != null) Object.DestroyImmediate(item); created.Clear(); }

    private Camera NewCamera(bool physical)
    {
        var go = new GameObject("Feed camera"); created.Add(go);
        var camera = go.AddComponent<Camera>();
        go.transform.position = new Vector3(2, 5, -12);
        go.transform.rotation = Quaternion.LookRotation(new Vector3(-0.1f, -0.35f, 1));
        camera.usePhysicalProperties = physical;
        if (physical)
        {
            camera.focalLength = 24; camera.sensorSize = new Vector2(36, 24);
            camera.gateFit = Camera.GateFitMode.Horizontal; camera.lensShift = new Vector2(0.02f, -0.01f);
        }
        else camera.fieldOfView = 40;
        return camera;
    }

    private LaneEditor NewLanes()
    {
        var go = new GameObject("Lane manager"); created.Add(go);
        var editor = go.AddComponent<LaneEditor>(); editor.enabled = false;
        editor.nodes.Add(new LaneEditor.LaneNode { id = "start", position = new Vector3(0, 0, 0), isSpawnPoint = true });
        editor.nodes.Add(new LaneEditor.LaneNode { id = "end", position = new Vector3(0, 0, 40), isDeadEnd = true });
        var lane = new LaneEditor.LaneStrip { id = "road", name = "Road", width = 6 };
        lane.sections.Add(new LaneEditor.Section { nodeId = "start" }); lane.sections.Add(new LaneEditor.Section { nodeId = "end" });
        editor.laneStrips.Add(lane);
        return editor;
    }

    [TestCase(true)]
    [TestCase(false)]
    public void FeedProjectionMatchesTheCameraAtTheFeedAspect(bool physical)
    {
        var camera = NewCamera(physical);
        var projection = new FeedProjection(camera, 384, 216);
        camera.aspect = 384f / 216f;
        foreach (var point in new[] { new Vector3(0, 0, 10), new Vector3(-3, 0, 25), new Vector3(4, 1, 6), new Vector3(1, 0, 60) })
        {
            Vector3 expected = camera.WorldToViewportPoint(point), actual = projection.WorldToViewport(point);
            Assert.That(actual.x, Is.EqualTo(expected.x).Within(1e-4f), "x of " + point);
            Assert.That(actual.y, Is.EqualTo(expected.y).Within(1e-4f), "y of " + point);
            Assert.That(actual.z, Is.EqualTo(expected.z).Within(1e-3f), "depth of " + point);
            Ray ray = projection.ViewportToRay(expected.x, expected.y);
            Assert.That(Vector3.Cross(ray.direction, (point - ray.origin).normalized).magnitude, Is.LessThan(1e-4f), "ray through " + point);
        }
    }

    [Test] public void ExportDoesNotChangeTheCamerasAspect()
    {
        var camera = NewCamera(true);
        camera.aspect = 1.25f;
        CameraLaneExport.Build(NewLanes(), camera, 384, 216);
        Assert.That(camera.aspect, Is.EqualTo(1.25f));
    }

    [Test] public void ExportDescribesLaneEdgesAndACalibrationThatReturnsToTheLane()
    {
        var camera = NewCamera(true);
        var layout = CameraLaneExport.Build(NewLanes(), camera, 384, 216);
        var lanes = (List<object>)layout["lanes"];
        Assert.That(lanes.Count, Is.EqualTo(1));
        var lane = (Dictionary<string, object>)lanes[0];
        Assert.That(lane["direction"], Is.EqualTo("Forward"));
        Assert.That(lane.ContainsKey("sublane_count"), Is.False);
        var samples = (List<object>)lane["samples"];
        Assert.That(samples.Count, Is.EqualTo(81), "40 m sampled every 0.5 m");
        var sample = (Dictionary<string, object>)samples[20];
        var boundaries = (List<object>)sample["boundaries_image"];
        Assert.That(boundaries.Count, Is.EqualTo(2), "The left and the right edge");
        Assert.That((bool)sample["in_view"], Is.True);
        Assert.That((double)sample["width_px"], Is.GreaterThan(16));

        // The image point of a lane edge leads back to the same ground point through the camera.
        var projection = new FeedProjection(camera, 384, 216);
        var image = (float[])boundaries[0];
        Ray ray = projection.ViewportToRay(image[0], 1 - image[1]);
        Assert.That(new Plane(Vector3.up, Vector3.zero).Raycast(ray, out float distance), Is.True);
        var left = (float[])sample["left"];
        Assert.That(Vector3.Distance(ray.GetPoint(distance), new Vector3(left[0], left[1], left[2])), Is.LessThan(0.01f));
        Assert.That(((List<object>)layout["image_points"]).Count, Is.GreaterThanOrEqualTo(4));
        Assert.That(((List<object>)layout["end_nodes"]).Count, Is.EqualTo(1));

        // The exported matrix reproduces the projection, for Python's footprint fitting.
        var m = (float[])((Dictionary<string, object>)layout["camera"])["world_to_clip"];
        var p = new Vector3(1.5f, 0.8f, 18);
        var clip = new Vector4(m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3], m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
                               m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11], m[12] * p.x + m[13] * p.y + m[14] * p.z + m[15]);
        Vector3 viewport = projection.WorldToViewport(p);
        Assert.That((clip.x / clip.w + 1) / 2, Is.EqualTo(viewport.x).Within(1e-5f));
        Assert.That((clip.y / clip.w + 1) / 2, Is.EqualTo(viewport.y).Within(1e-5f));
    }
}
#endif
