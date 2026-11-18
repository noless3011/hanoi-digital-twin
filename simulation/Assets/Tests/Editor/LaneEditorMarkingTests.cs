#if UNITY_INCLUDE_TESTS
using System;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;

public sealed class LaneEditorMarkingTests
{
    private static readonly Type FileType = typeof(LaneEditor).GetNestedType("LaneFile", BindingFlags.NonPublic);
    private static readonly MethodInfo Validate = typeof(LaneEditor).GetMethod("ValidateFile", BindingFlags.Static | BindingFlags.NonPublic);
    private const string Network = "{\"version\":3,\"editMode\":0,\"nodes\":["
        + "{\"id\":\"entry\",\"position\":{\"x\":0,\"y\":0,\"z\":0},\"isSpawnPoint\":true,\"spawnPointName\":\"Inbound\"},"
        + "{\"id\":\"exit\",\"position\":{\"x\":0,\"y\":0,\"z\":10}}],"
        + "\"laneStrips\":[{\"id\":\"road\",\"name\":\"Road\",\"width\":3.2,\"sublaneCount\":4,\"direction\":0,"
        + "\"sections\":[{\"nodeId\":\"entry\",\"widthScale\":1},{\"nodeId\":\"exit\",\"widthScale\":1}]}]}";

    private static object Parse(string json)
    {
        object file = JsonUtility.FromJson(json, FileType);
        Validate.Invoke(null, new[] { file });
        return file;
    }
    private static List<LaneEditor.LaneNode> Nodes(object file) => (List<LaneEditor.LaneNode>)FileType.GetField("nodes").GetValue(file);

    [Test] public void ExistingVersionThreeFilesLoadWithoutInventingExitMarks()
    {
        List<LaneEditor.LaneNode> nodes = Nodes(Parse(Network));
        Assert.That(nodes[0].isSpawnPoint, Is.True);
        Assert.That(nodes[0].spawnPointName, Is.EqualTo("Inbound"));
        Assert.That(nodes[0].isDeadEnd, Is.False);
        Assert.That(nodes[1].isDeadEnd, Is.False, "An unmarked lane end must not silently become a valid video exit.");
    }

    [Test] public void LaneFileRoundTripKeepsEntryAndExitMarksOnATwoWayBoundary()
    {
        object file = Parse(Network);
        List<LaneEditor.LaneNode> nodes = Nodes(file);
        nodes[0].isDeadEnd = true; nodes[0].deadEndName = "North camera edge";
        nodes[1].isDeadEnd = true; nodes[1].deadEndName = "South camera edge";

        List<LaneEditor.LaneNode> restored = Nodes(Parse(JsonUtility.ToJson(file)));
        Assert.That(restored[0].id, Is.EqualTo("entry"));
        Assert.That(restored[0].isSpawnPoint && restored[0].isDeadEnd, Is.True);
        Assert.That(restored[0].spawnPointName, Is.EqualTo("Inbound"));
        Assert.That(restored[0].deadEndName, Is.EqualTo("North camera edge"));
        Assert.That(restored[1].isSpawnPoint, Is.False);
        Assert.That(restored[1].isDeadEnd, Is.True);
        Assert.That(restored[1].deadEndName, Is.EqualTo("South camera edge"));
    }
    [Test] public void TrafficLightLinesAreOptionalAndRoundTripWithStableIdentity()
    {
        object file = Parse(Network);
        var lights = (List<LaneEditor.TrafficLightLine>)FileType.GetField("trafficLights").GetValue(file);
        Assert.That(lights, Is.Empty);
        lights.Add(new LaneEditor.TrafficLightLine { id = "north-signal", name = "North approach", start = new Vector3(-4, 2, 6), end = new Vector3(4, 2, 6) });
        lights[0].redDurationS = 12; lights[0].greenDurationS = 23; lights[0].cycleOffsetS = -5;
        object restored = Parse(JsonUtility.ToJson(file));
        var result = (List<LaneEditor.TrafficLightLine>)FileType.GetField("trafficLights").GetValue(restored);
        Assert.That(result.Count, Is.EqualTo(1));
        Assert.That(result[0].id, Is.EqualTo("north-signal"));
        Assert.That(result[0].name, Is.EqualTo("North approach"));
        Assert.That(result[0].start, Is.EqualTo(lights[0].start));
        Assert.That(result[0].end, Is.EqualTo(lights[0].end));
        Assert.That(result[0].redDurationS, Is.EqualTo(12));
        Assert.That(result[0].greenDurationS, Is.EqualTo(23));
        Assert.That(result[0].cycleOffsetS, Is.EqualTo(-5));
    }

    [Test] public void InvalidTrafficLightGeometryAndDuplicateIdsAreRejected()
    {
        // Exercise the production validator without requiring Unity's native JSON engine.
        object file = Activator.CreateInstance(FileType, true);
        FileType.GetField("nodes").SetValue(file, new List<LaneEditor.LaneNode>());
        FileType.GetField("laneStrips").SetValue(file, new List<LaneEditor.LaneStrip>());
        Validate.Invoke(null, new[] { file });
        var lights = (List<LaneEditor.TrafficLightLine>)FileType.GetField("trafficLights").GetValue(file);
        Assert.That(lights, Is.Empty, "Old files without trafficLights inherit an empty list.");
        var light = new LaneEditor.TrafficLightLine { start = Vector3.zero, end = Vector3.zero };
        lights.Add(light);
        Assert.Throws<TargetInvocationException>(() => Validate.Invoke(null, new[] { file }));
        light.end = new Vector3(float.NaN, 0, 1);
        Assert.Throws<TargetInvocationException>(() => Validate.Invoke(null, new[] { file }));
        light.end = Vector3.right * 4;
        Validate.Invoke(null, new[] { file }); // A valid light does not require a lane or node.
        light.redDurationS = -1;
        Assert.Throws<TargetInvocationException>(() => Validate.Invoke(null, new[] { file }));
        light.redDurationS = 30; light.greenDurationS = float.NaN;
        Assert.Throws<TargetInvocationException>(() => Validate.Invoke(null, new[] { file }));
        light.greenDurationS = 30;
        light.end = Vector3.right * 0.01f;
        Validate.Invoke(null, new[] { file }); // Local coordinates may be small on scaled editors.
        light.end = Vector3.right * 4;
        string id = light.id; light.id = "";
        Assert.Throws<TargetInvocationException>(() => Validate.Invoke(null, new[] { file }));
        light.id = id;
        lights.Add(new LaneEditor.TrafficLightLine { id = light.id, start = Vector3.zero, end = Vector3.forward * 4 });
        Assert.Throws<TargetInvocationException>(() => Validate.Invoke(null, new[] { file }));
    }

    [Test] public void TrafficLightTimingUsesSimulationTimeOffsetsAndExactBoundaries()
    {
        var light = new LaneEditor.TrafficLightLine { redDurationS = 10, greenDurationS = 20 };
        Assert.That(light.AspectAt(0), Is.EqualTo("red"));
        Assert.That(light.AspectAt(9.999), Is.EqualTo("red"));
        Assert.That(light.AspectAt(10), Is.EqualTo("green"));
        Assert.That(light.AspectAt(29.999), Is.EqualTo("green"));
        Assert.That(light.AspectAt(30), Is.EqualTo("red"));
        Assert.That(light.AspectAt(1e9), Is.EqualTo("green"));
        light.cycleOffsetS = 10;
        Assert.That(light.AspectAt(0), Is.EqualTo("green"));
        light.cycleOffsetS = -1;
        Assert.That(light.AspectAt(0), Is.EqualTo("green"));
        Assert.That(light.AspectAt(1), Is.EqualTo("red"));
        light.greenDurationS = 0;
        Assert.That(light.AspectAt(100), Is.EqualTo("red"));
        light.redDurationS = 0;
        Assert.That(light.AspectAt(100), Is.EqualTo("red"));
        light.greenDurationS = 5;
        Assert.That(light.AspectAt(100), Is.EqualTo("green"));
        Assert.That(LaneEditor.TrafficLightLine.ValidTiming(1, 2, float.PositiveInfinity), Is.False);
        Assert.Throws<ArgumentException>(() => LaneEditor.TrafficLightLine.AspectAt(0, -1, 1, 0));
    }

    [Test] public void TrafficLightDrawingUsesLocalCoordinatesAndCancelsIncompleteLines()
    {
        var go = new GameObject("Traffic light authoring test");
        try
        {
            go.transform.position = new Vector3(10, 3, 20);
            go.transform.rotation = Quaternion.Euler(0, 35, 0);
            var editor = go.AddComponent<LaneEditor>();
            editor.BeginTrafficLightDrawing();
            Assert.That(editor.PlaceTrafficLightPoint(go.transform.TransformPoint(new Vector3(-3, 0, 5))), Is.True);
            editor.CancelTrafficLightDrawing();
            Assert.That(editor.TrafficLights, Is.Empty);
            editor.BeginTrafficLightDrawing();
            Vector3 start = go.transform.TransformPoint(new Vector3(-3, 0, 5));
            Assert.That(editor.PlaceTrafficLightPoint(start), Is.True);
            Assert.That(editor.PlaceTrafficLightPoint(start), Is.False);
            Assert.That(editor.PlaceTrafficLightPoint(go.transform.TransformPoint(new Vector3(3, 0, 5))), Is.True);
            Assert.That(editor.IsDrawingTrafficLight, Is.False);
            Assert.That(editor.MoveTrafficLightEndpoint(0, true, go.transform.TransformPoint(editor.CurrentTrafficLight.end)), Is.False);
            Assert.That(Vector3.Distance(editor.CurrentTrafficLight.start, new Vector3(-3, 0, 5)), Is.LessThan(0.0001f));
            Assert.That(editor.MoveTrafficLightEndpoint(0, true, go.transform.TransformPoint(new Vector3(-4, 0, 5))), Is.True);
            Assert.That(Vector3.Distance(editor.CurrentTrafficLight.start, new Vector3(-4, 0, 5)), Is.LessThan(0.0001f));
            editor.RebuildVisuals();
            var line = Array.Find(go.GetComponentsInChildren<LineRenderer>(), r => r.name.StartsWith("Traffic light "));
            Assert.That(line, Is.Not.Null);
            var tint = new MaterialPropertyBlock(); line.GetPropertyBlock(tint);
            Assert.That(tint.GetColor("_Color"), Is.EqualTo(Color.red));
            Assert.That(line.GetComponent<Collider>(), Is.Null, "Stop lines are authored data, not physical barriers.");
            editor.DeleteSelectedTrafficLight();
            Assert.That(editor.TrafficLights, Is.Empty);
        }
        finally { UnityEngine.Object.DestroyImmediate(go); }
    }
}
#endif
