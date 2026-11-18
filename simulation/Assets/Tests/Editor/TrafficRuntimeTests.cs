#if UNITY_INCLUDE_TESTS && LEGACY_SUBLANE_RUNTIME
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;

/// <summary>Contract-level map and signal acceptance tests, runnable in Unity's EditMode Test Runner.</summary>
public sealed class TrafficRuntimeTests
{
    private readonly List<GameObject> objects = new List<GameObject>();
    private LaneEditor NewEditor()
    {
        GameObject instance = new GameObject("Traffic test map"); objects.Add(instance);
        LaneEditor editor = instance.AddComponent<LaneEditor>();
        editor.showRuntimeUI = false; editor.prepareMeshColliders = false; editor.generateLaneColliders = false;
        return editor;
    }
    private LaneEditor.LaneNode Node(LaneEditor editor, string id, Vector3 point)
    {
        LaneEditor.LaneNode node = new LaneEditor.LaneNode { id = id, position = point }; editor.nodes.Add(node); return node;
    }
    private LaneEditor.LaneStrip Lane(LaneEditor editor, string id, string start, string end, LaneEditor.LaneDirection direction = LaneEditor.LaneDirection.Forward)
    {
        LaneEditor.LaneStrip lane = new LaneEditor.LaneStrip { id = id, name = "Same display name", width = 6, sublaneCount = 2, direction = direction };
        lane.sections.Add(new LaneEditor.Section { nodeId = start }); lane.sections.Add(new LaneEditor.Section { nodeId = end }); editor.laneStrips.Add(lane); return lane;
    }
    [TearDown] public void DestroySceneObjects()
    {
        foreach (GameObject instance in objects) if (instance != null) UnityEngine.Object.DestroyImmediate(instance);
        objects.Clear();
    }

    [Test] public void BothDirectionsUseWorldCoordinatesAndPreserveAuthoringCells()
    {
        LaneEditor editor = NewEditor();
        editor.transform.position = new Vector3(12, 2, -7);
        Node(editor, "start", Vector3.zero); Node(editor, "end", new Vector3(0, 0, 20));
        LaneEditor.LaneStrip lane = Lane(editor, "stable-lane", "start", "end", LaneEditor.LaneDirection.Both); lane.sublaneCount = 6;
        TrafficNetwork map = TrafficNetwork.Build(editor, new JunctionController[0], 12);
        TrafficSegment forward = map.segments["stable-lane:forward"], reverse = map.segments["stable-lane:reverse"];
        Assert.That(forward.startNodeId, Is.EqualTo("start")); Assert.That(reverse.startNodeId, Is.EqualTo("end"));
        Assert.That(forward.Sample(0), Is.EqualTo(editor.transform.TransformPoint(Vector3.zero)));
        Assert.That(reverse.Sample(0), Is.EqualTo(forward.Sample(forward.length)));
        Assert.That(forward.Tangent(5).z, Is.EqualTo(1).Within(0.0001)); Assert.That(reverse.Tangent(5).z, Is.EqualTo(-1).Within(0.0001));
        // A wide car may cover several authoring cells; its continuous reference is clamped inside the full strip.
        Vector3 f = forward.ReferencePath(0, 4)[0], r = reverse.ReferencePath(0, 4)[1];
        Assert.That(f.x, Is.EqualTo(r.x).Within(0.0001));
        Assert.That(f.x - editor.transform.position.x - 2, Is.GreaterThanOrEqualTo(-3.0001));
        Assert.That(f.x - editor.transform.position.x + 2, Is.LessThanOrEqualTo(3.0001));
        Assert.That(forward.IsFeasible(4.4f, 4, 0), Is.True); Assert.That(reverse.IsFeasible(4.4f, 4, 0), Is.True);
        reverse.Project(reverse.Sample(5) + Vector3.left, out float station, out float lateral);
        Assert.That(station, Is.EqualTo(5).Within(0.0001)); Assert.That(lateral, Is.EqualTo(1).Within(0.0001));
    }

    [Test] public void WidthNarrowingRejectsTheWholeVehicleRoute()
    {
        LaneEditor editor = NewEditor(); Node(editor, "a", Vector3.zero); Node(editor, "b", new Vector3(0, 0, 20));
        LaneEditor.LaneStrip lane = Lane(editor, "narrowing", "a", "b"); lane.sections[1].widthScale = 0.25f;
        TrafficSegment edge = TrafficNetwork.Build(editor, new JunctionController[0], 12).segments["narrowing:forward"];
        Assert.That(edge.WidthAt(10), Is.EqualTo(3.75f).Within(0.0001));
        Assert.That(edge.IsFeasible(4.4f, 1.8f, 0), Is.False);
        Assert.That(edge.IsFeasible(2, 0.7f, 0), Is.True);
    }

    private static TrafficSegment SurfaceTestLane(Vector3[] points, float halfWidth = 2)
    {
        TrafficSegment segment = new TrafficSegment { id = "surface-test", type = "lane_edge", forward = true, sublaneCount = 1,
            points = points, left = points.Select(p => p + Vector3.left * halfWidth).ToArray(),
            right = points.Select(p => p + Vector3.right * halfWidth).ToArray(),
            widths = points.Select(p => halfWidth * 2).ToArray(), up = points.Select(p => Vector3.up).ToArray() };
        segment.Prepare(); return segment;
    }

    [Test] public void InternalBendUsesTheContinuousRibbonWithoutAnArtificialEndCap()
    {
        TrafficSegment segment = SurfaceTestLane(new[] { Vector3.zero, new Vector3(0, 0, 10), new Vector3(5, 0, 20) });
        Vector3 point = new Vector3(-1.5f, 0, 10.25f);
        segment.Project(point, out float station, out _);
        Assert.That(station, Is.EqualTo(10).Within(0.0001));
        Assert.That(Vector3.Dot(point - segment.Sample(station), segment.Tangent(station)), Is.LessThan(-0.05f),
            "The old projection/tangent test incorrectly treated the inside of this bend as beyond an endpoint.");
        Assert.That(segment.ContainsSurfacePoint(point), Is.True);
        Assert.That(segment.ContainsSurfacePoint(new Vector3(-2.5f, 0, 10.25f)), Is.False, "Physical side boundaries still apply.");
        Assert.That(segment.ContainsSurfacePoint(point + Vector3.up * 5), Is.False, "An overpass is not this road surface.");
    }

    [Test] public void EndpointOverhangIsFiniteAndDoesNotWidenTheRoad()
    {
        TrafficSegment segment = SurfaceTestLane(new[] { Vector3.zero, new Vector3(0, 2, 10) });
        Assert.That(segment.ContainsSurfacePoint(new Vector3(0, -0.38f, -1.9f)), Is.False);
        Assert.That(segment.ContainsSurfacePoint(new Vector3(0, -0.38f, -1.9f), 2, 2), Is.True);
        Assert.That(segment.ContainsSurfacePoint(new Vector3(0, 2.38f, 11.9f), 2, 2), Is.True);
        Assert.That(segment.ContainsSurfacePoint(new Vector3(0, -0.42f, -2.1f), 2, 2), Is.False);
        Assert.That(segment.ContainsSurfacePoint(new Vector3(0, 2.42f, 12.1f), 2, 2), Is.False);
        Assert.That(segment.ContainsSurfacePoint(new Vector3(2.1f, -0.2f, -1), 2, 2), Is.False);
        Assert.That(segment.ContainsSurfacePoint(new Vector3(0, 5, -1), 2, 2), Is.False);
    }

    [Test] public void SurfaceHeightMatchesTheLaneMeshTriangleDiagonal()
    {
        TrafficSegment segment = SurfaceTestLane(new[] { Vector3.zero, new Vector3(0, 2, 10) });
        segment.left[1].y = 0; segment.right[1].y = 4;
        Assert.That(segment.ContainsSurfacePoint(new Vector3(1, 2, 7.5f), verticalTolerance: 0.01f), Is.True);
        Assert.That(segment.ContainsSurfacePoint(new Vector3(1, 2.25f, 7.5f), verticalTolerance: 0.01f), Is.False,
            "Use the same triangle diagonal as RoadMeshGeometry rather than bilinear height interpolation.");
        Assert.That(segment.ContainsSurfacePoint(new Vector3(-1, 0, 2.5f), verticalTolerance: 0.01f), Is.True);
    }

    [Test] public void SurfaceContainmentRejectsInvalidGeometryInputs()
    {
        TrafficSegment segment = SurfaceTestLane(new[] { Vector3.zero, new Vector3(0, 0, 10) });
        Assert.That(segment.ContainsSurfacePoint(new Vector3(float.NaN, 0, 0)), Is.False);
        Assert.That(segment.ContainsSurfacePoint(Vector3.zero, float.PositiveInfinity), Is.False);
        Assert.That(segment.ContainsSurfacePoint(Vector3.zero, endOverhang: -1), Is.False);
        Assert.That(segment.ContainsSurfacePoint(Vector3.zero, verticalTolerance: -1), Is.False);
    }

    [Test] public void BendReferencesReserveVehicleLengthClearanceWithoutChangingAuthoringCells()
    {
        TrafficSegment segment = SurfaceTestLane(new[] { Vector3.zero, new Vector3(0, 0, 30), new Vector3(8, 0, 60) }, 1.845f);
        segment.sublaneCount = 4;
        Vector3[] widthOnly = segment.ReferencePath(3, 1.8f), car = segment.ReferencePath(3, 1.8f, 4.4f);
        Vector3[] shortVehicle = segment.ReferencePath(3, 1.8f, 2);
        Assert.That(car[0].x, Is.GreaterThan(0), "The same authoring side remains preferred when safe.");
        Assert.That(car[0].x, Is.LessThan(widthOnly[0].x - 0.1f), "A bend needs clearance for length and smooth steering.");
        Assert.That(car[0].x, Is.LessThan(shortVehicle[0].x), "A longer body needs more lateral room.");
        Assert.That(car[0].x, Is.EqualTo(car[1].x).Within(0.0001), "Reserve room across the incoming span instead of shifting at the node.");
        Assert.That(car[2].x - segment.points[2].x, Is.EqualTo(car[1].x).Within(0.0001));
        Assert.That(segment.IsFeasible(4.4f, 1.8f, 3), Is.True);
        Assert.That(segment.IsFeasible(4.4f, 1.8f, 0), Is.True, "Both sides of an ordinary bend must remain usable.");
        TrafficSegment reverse = new TrafficSegment { id = "reverse-surface-test", type = "lane_edge", forward = false, sublaneCount = 4,
            points = segment.points.Reverse().ToArray(), left = segment.right.Reverse().ToArray(), right = segment.left.Reverse().ToArray(),
            widths = segment.widths.Reverse().ToArray(), up = segment.up.Reverse().ToArray() };
        reverse.Prepare(); Vector3[] reverseCar = reverse.ReferencePath(3, 1.8f, 4.4f);
        for (int i = 0; i < car.Length; i++) Assert.That(Vector3.Distance(car[i], reverseCar[car.Length - 1 - i]), Is.LessThan(0.0001));
    }

    [Test] public void TaperedLaneReferencesReserveRoomForTheWholeBodyAtEndpoints()
    {
        var segment = new TrafficSegment { id = "tapered", type = "lane_edge", forward = true, sublaneCount = 4,
            points = new[] { Vector3.zero, new Vector3(0, 0, 12), new Vector3(0, 0, 24) },
            left = new[] { new Vector3(-1.85f, 0, 0), new Vector3(-1.4f, 0, 12), new Vector3(-1.85f, 0, 24) },
            right = new[] { new Vector3(1.85f, 0, 0), new Vector3(1.4f, 0, 12), new Vector3(1.85f, 0, 24) },
            widths = new[] { 3.7f, 2.8f, 3.7f }, up = new[] { Vector3.up, Vector3.up, Vector3.up } };
        segment.Prepare();
        Vector3[] widthOnly = segment.ReferencePath(0, 1.8f), car = segment.ReferencePath(0, 1.8f, 4.4f);
        Vector3 outgoing = (widthOnly[2] - widthOnly[1]).normalized;
        Vector3 outerFrontCorner = widthOnly[2] + outgoing * 2.2f - new Vector3(outgoing.z, 0, -outgoing.x) * 0.9f;
        Assert.That(segment.ContainsSurfacePoint(outerFrontCorner, 2.7f, 2.7f), Is.False,
            "The old width-only path leaves the physical side boundary at the endpoint.");
        Assert.That(car[0].x, Is.GreaterThan(widthOnly[0].x));
        Assert.That(car[0].x, Is.LessThan(0), "Keep the assigned side when an inward offset is sufficient.");
        Assert.That(segment.IsFeasible(4.4f, 1.8f, 0), Is.True);
        Assert.That(segment.IsFeasible(4.4f, 1.8f, 3), Is.True);
        Assert.That(segment.IsFeasible(4.4f, 3, 0), Is.False, "A real narrowing obstruction still rejects the body.");
    }

    [Test] public void UnboundConnectionsDoNotInventTrafficAuthority()
    {
        LaneEditor editor = NewEditor();
        Node(editor, "w", new Vector3(-20, 0, 0)); Node(editor, "j", Vector3.zero); Node(editor, "n", new Vector3(0, 0, 20));
        Lane(editor, "west", "w", "j"); Lane(editor, "north", "j", "n");
        TrafficNetwork map = TrafficNetwork.Build(editor, new JunctionController[0], 12);
        Assert.That(map.segments.Count, Is.EqualTo(2)); Assert.That(map.movements, Is.Empty);
        Assert.That(map.segments.Values.All(s => s.type == "lane_edge"), Is.True);
    }

    [Test] public void CoincidentSamplesAreRemovedAndDistancesRemainHorizontal()
    {
        LaneEditor editor = NewEditor();
        Node(editor, "a", Vector3.zero); Node(editor, "b", new Vector3(0, 4, 0)); Node(editor, "c", new Vector3(0, 4, 10));
        LaneEditor.LaneStrip lane = Lane(editor, "grade", "a", "c"); lane.sections.Insert(1, new LaneEditor.Section { nodeId = "b" });
        TrafficSegment edge = TrafficNetwork.Build(editor, new JunctionController[0], 12).segments["grade:forward"];
        Assert.That(edge.points.Length, Is.EqualTo(2)); Assert.That(edge.length, Is.EqualTo(10).Within(0.0001));
    }

    [Test] public void WideVehicleConnectorReferencesJoinTheirClampedAuthoringCells()
    {
        LaneEditor editor = NewEditor();
        Node(editor, "w", new Vector3(-20, 0, 0)); Node(editor, "j", Vector3.zero); Node(editor, "n", new Vector3(0, 0, 20));
        Lane(editor, "west", "w", "j").sublaneCount = 6; Lane(editor, "north", "j", "n").sublaneCount = 6;
        GameObject instance = new GameObject("Uncontrolled authored turn"); objects.Add(instance);
        JunctionController controller = instance.AddComponent<JunctionController>();
        string movementId = TrafficNetwork.MovementId("west:forward", "north:forward", "j");
        JunctionController.MovementBinding binding = new JunctionController.MovementBinding { movementId = movementId, uncontrolledPriorityRule = "yield_to_conflicting_traffic" };
        controller.movements.Add(binding);
        controller.connectorEnvelopes.Add(new JunctionController.ConnectorEnvelope { movementId = movementId, widthProfileM = new[] { 6f },
            localPolygon = new[] { new Vector3(-10, 0, -10), new Vector3(-10, 0, 10), new Vector3(10, 0, 10), new Vector3(10, 0, -10) } });
        TrafficNetwork map = TrafficNetwork.Build(editor, new[] { controller }, 12); controller.Validate(map, 0.05f);
        TrafficSegment connector = map.segments[TrafficNetwork.ConnectorId("west:forward", "north:forward", "j", 0, 0)];
        Vector3[] reference = connector.ReferencePath(0, 1.8f);
        Assert.That(Vector3.Distance(reference[0], map.segments["west:forward"].ReferencePath(0, 1.8f).Last()), Is.LessThan(0.0001f));
        Assert.That(Vector3.Distance(reference.Last(), map.segments["north:forward"].ReferencePath(0, 1.8f)[0]), Is.LessThan(0.0001f));
        Assert.That(Vector3.Distance(reference[0], connector.points[0]), Is.GreaterThan(0.1f), "Regression: a wide car's boundary cell reference must shift away from the road edge.");
        Assert.That(map.FindMovement(movementId).stopS, Is.EqualTo(map.segments["west:forward"].length), "Without an authored line the uncontrolled entry boundary is the approach endpoint.");
        binding.stopLineId = "yield-line";
        controller.stopLines.Add(new JunctionController.StopLine { id = "yield-line", approachEdgeId = "west:forward", edgeS = 12,
            firstSublane = 0, lastSublane = 5, localA = new Vector3(-8, 0, -3), localB = new Vector3(-8, 0, 3) });
        map = TrafficNetwork.Build(editor, new[] { controller }, 12); controller.Validate(map, 0.05f);
        Assert.That(map.FindMovement(movementId).stopS, Is.EqualTo(12));
    }

    [Test] public void AuthoringCatalogIncludesUnconfiguredStableCandidateIds()
    {
        LaneEditor editor = NewEditor();
        Node(editor, "w", new Vector3(-20, 0, 0)); Node(editor, "j", Vector3.zero); Node(editor, "n", new Vector3(0, 0, 20));
        Lane(editor, "west", "w", "j"); Lane(editor, "north", "j", "n");
        Dictionary<string, object> catalog = TrafficNetwork.GeneratedIdentityCatalog(editor);
        var edges = (List<object>)catalog["lane_edges"]; var connectors = (List<object>)catalog["connector_candidates"];
        Assert.That(edges.Count, Is.EqualTo(2)); Assert.That(connectors.Count, Is.EqualTo(2));
        Assert.That(((Dictionary<string, object>)connectors[0])["movement_id"], Is.EqualTo(TrafficNetwork.MovementId("west:forward", "north:forward", "j")));
        Assert.That(((Dictionary<string, object>)connectors[0])["has_horizontal_geometry"], Is.EqualTo(true));
        Assert.That(connectors.Select(c => ((Dictionary<string, object>)c)["id"]).Distinct().Count(), Is.EqualTo(2));
    }

    private TrafficNetwork ControlledCrossing(out JunctionController controller, out string firstMovement, out string secondMovement,
        float secondHeight = 0, bool authorConflict = true, float zoneTop = 5)
    {
        LaneEditor editor = NewEditor();
        Node(editor, "w", new Vector3(-20, 0, 0)); Node(editor, "s", new Vector3(0, secondHeight, -20));
        Node(editor, "j", Vector3.zero); Node(editor, "n", new Vector3(0, 0, 20)); Node(editor, "e", new Vector3(20, secondHeight, 0));
        string secondJunction = secondHeight == 0 ? "j" : "j-high";
        if (secondJunction != "j") Node(editor, secondJunction, new Vector3(0, secondHeight, 0));
        Lane(editor, "west", "w", "j"); Lane(editor, "south", "s", secondJunction); Lane(editor, "north", "j", "n"); Lane(editor, "east", secondJunction, "e");
        GameObject instance = new GameObject("Traffic test controller"); objects.Add(instance); controller = instance.AddComponent<JunctionController>(); controller.id = "controller";
        firstMovement = TrafficNetwork.MovementId("west:forward", "north:forward", "j"); secondMovement = TrafficNetwork.MovementId("south:forward", "east:forward", secondJunction);
        controller.movements.Add(new JunctionController.MovementBinding { movementId = firstMovement, signalGroupId = "a", stopLineId = "stop-w", conflictZoneIds = authorConflict ? new[] { "crossing" } : new string[0] });
        controller.movements.Add(new JunctionController.MovementBinding { movementId = secondMovement, signalGroupId = "b", stopLineId = "stop-s", conflictZoneIds = authorConflict ? new[] { "crossing" } : new string[0] });
        controller.stopLines.Add(new JunctionController.StopLine { id = "stop-w", approachEdgeId = "west:forward", edgeS = 10, firstSublane = 0, lastSublane = 1, localA = new Vector3(-10, 0, -3), localB = new Vector3(-10, 0, 3) });
        controller.stopLines.Add(new JunctionController.StopLine { id = "stop-s", approachEdgeId = "south:forward", edgeS = 10, firstSublane = 0, lastSublane = 1, localA = new Vector3(-3, secondHeight, -10), localB = new Vector3(3, secondHeight, -10) });
        controller.signalGroups.Add(new JunctionController.SignalGroup { id = "a", heads = new List<JunctionController.SignalHead> { new JunctionController.SignalHead { localPosition = new Vector3(-10, 3, 0) } } });
        controller.signalGroups.Add(new JunctionController.SignalGroup { id = "b", heads = new List<JunctionController.SignalHead> { new JunctionController.SignalHead { localPosition = new Vector3(0, secondHeight + 3, -10) } } });
        Vector3[] polygon = { new Vector3(-4, 0, -4), new Vector3(-4, 0, 4), new Vector3(4, 0, 4), new Vector3(4, 0, -4) };
        controller.connectorEnvelopes.Add(new JunctionController.ConnectorEnvelope { movementId = firstMovement, widthProfileM = new[] { 1.2f }, localPolygon = polygon });
        controller.connectorEnvelopes.Add(new JunctionController.ConnectorEnvelope { movementId = secondMovement, widthProfileM = new[] { 1.2f }, localPolygon = polygon.Select(p => p + Vector3.up * secondHeight).ToArray() });
        if (authorConflict) controller.conflictZones.Add(new JunctionController.ConflictZone { id = "crossing", localPolygon = polygon, minLocalY = 0, maxLocalY = zoneTop,
            movementPairs = new List<JunctionController.MovementPair> { new JunctionController.MovementPair { first = firstMovement, second = secondMovement } } });
        controller.phases.Add(new JunctionController.Phase { greenGroupIds = new[] { "a" }, greenSeconds = 0.1f, amberSeconds = 0.1f, allRedSeconds = 0.1f });
        controller.phases.Add(new JunctionController.Phase { greenGroupIds = new[] { "b" }, greenSeconds = 0.1f, amberSeconds = 0.1f, allRedSeconds = 0.1f });
        return TrafficNetwork.Build(editor, new[] { controller }, 12);
    }

    [Test] public void ProtectedPhasesRetainAmberAndHoldAllRedUntilConflictsClear()
    {
        TrafficNetwork map = ControlledCrossing(out JunctionController controller, out string first, out string second); controller.Validate(map, 0.05f);
        HashSet<string> empty = new HashSet<string>();
        Assert.That(controller.Aspect("a"), Is.EqualTo("red")); Assert.That(controller.Aspect("b"), Is.EqualTo("red"));
        controller.Tick(0.1f, empty, empty); Assert.That(controller.Aspect("a"), Is.EqualTo("green")); Assert.That(controller.Aspect("b"), Is.EqualTo("red"));
        controller.Tick(0.1f, empty, empty); Assert.That(controller.Aspect("a"), Is.EqualTo("amber")); Assert.That(controller.Aspect("b"), Is.EqualTo("red"));
        controller.Tick(0.1f, empty, empty); Assert.That(controller.Aspect("a"), Is.EqualTo("red")); Assert.That(controller.Aspect("b"), Is.EqualTo("red"));
        controller.Tick(0.1f, new HashSet<string> { first }, empty);
        Assert.That(controller.HoldingForOccupancy, Is.True); Assert.That(controller.Aspect("b"), Is.EqualTo("red"));
        controller.Tick(0.05f, empty, new HashSet<string> { "crossing" });
        Assert.That(controller.HoldingForOccupancy, Is.True); Assert.That(controller.Aspect("b"), Is.EqualTo("red"));
        controller.Tick(0.05f, empty, empty); Assert.That(controller.Aspect("b"), Is.EqualTo("green"));
        controller.ResetController(); controller.Tick(0.1f, new HashSet<string> { second }, empty);
        Assert.That(controller.Aspect("a"), Is.EqualTo("red")); Assert.That(controller.HoldingForOccupancy, Is.True);
    }

    [Test] public void UnsafeSimultaneousProtectedGreensAreRejected()
    {
        TrafficNetwork map = ControlledCrossing(out JunctionController controller, out _, out _);
        controller.phases[0].greenGroupIds = new[] { "a", "b" };
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() => controller.Validate(map, 0.05f));
        Assert.That(error.Message, Does.Contain("conflicting movements green"));
    }

    [Test] public void SlightlyRaisedRoadsStillConflictWhenVehicleBodiesOverlap()
    {
        InvalidOperationException missing = Assert.Throws<InvalidOperationException>(() => ControlledCrossing(out _, out _, out _, secondHeight: 1, authorConflict: false));
        Assert.That(missing.Message, Does.Contain("height-aware conflict zone"));
        InvalidOperationException shortZone = Assert.Throws<InvalidOperationException>(() => ControlledCrossing(out _, out _, out _, secondHeight: 1, zoneTop: 5));
        Assert.That(shortZone.Message, Does.Contain("full configured vehicle height"));
        TrafficNetwork map = ControlledCrossing(out JunctionController controller, out string first, out string second, secondHeight: 1, zoneTop: 6);
        controller.Validate(map, 0.05f); Assert.That(map.MovementsConflict(first, second), Is.True);
    }

    [Test] public void OverpassGeometryDoesNotRequireAFalseCrossingConflict()
    {
        TrafficNetwork map = ControlledCrossing(out JunctionController controller, out string first, out string second, secondHeight: 6, authorConflict: false);
        Assert.That(map.MovementsConflict(first, second), Is.False); Assert.That(map.conflictZones, Is.Empty);
        controller.phases[0].greenGroupIds = new[] { "a", "b" };
        Assert.DoesNotThrow(() => controller.Validate(map, 0.05f));
    }

    [Test] public void CrossingDifferentSourceCellsOnOneApproachRequiresAConflict()
    {
        LaneEditor editor = NewEditor(); Node(editor, "start", new Vector3(-20, 0, 0)); Node(editor, "end", Vector3.zero);
        Lane(editor, "approach", "start", "end");
        TrafficNetwork map = TrafficNetwork.Build(editor, new JunctionController[0], 12);
        TrafficSegment first = TestConnector("first", 0, new Vector3(-2, 0, -2), new Vector3(2, 0, 2));
        TrafficSegment second = TestConnector("second", 1, new Vector3(-2, 0, 2), new Vector3(2, 0, -2));
        map.segments.Add(first.id, first); map.segments.Add(second.id, second);
        TrafficMovement a = new TrafficMovement { id = "turn-a", approachEdgeId = "approach:forward", departureEdgeId = "out-a" };
        TrafficMovement b = new TrafficMovement { id = "turn-b", approachEdgeId = "approach:forward", departureEdgeId = "out-b" };
        a.connectorIds.Add(first.id); b.connectorIds.Add(second.id); map.movements.Add(a); map.movements.Add(b);
        MethodInfo validate = typeof(TrafficNetwork).GetMethod("ValidateConflictCoverage", BindingFlags.Instance | BindingFlags.NonPublic);
        TargetInvocationException error = Assert.Throws<TargetInvocationException>(() => validate.Invoke(map, null));
        Assert.That(error.InnerException, Is.TypeOf<InvalidOperationException>()); Assert.That(error.InnerException.Message, Does.Contain("height-aware conflict zone"));
        // The same origin cell is an ordinary following/diverging corridor, not an independent crossing stream.
        second.fromSublane = first.fromSublane;
        Assert.DoesNotThrow(() => validate.Invoke(map, null));
    }

    private static TrafficSegment TestConnector(string id, int fromCell, Vector3 start, Vector3 end)
    {
        Vector3 tangent = (end - start).normalized, side = Vector3.Cross(Vector3.up, tangent) * 0.5f;
        TrafficSegment connector = new TrafficSegment { id = id, type = "connector", fromSublane = fromCell,
            points = new[] { start, end }, left = new[] { start - side, end - side }, right = new[] { start + side, end + side },
            up = new[] { Vector3.up, Vector3.up }, widths = new[] { 1f, 1f } };
        connector.Prepare(); return connector;
    }

    [Test] public void HeadsAreObservedOnlyForTheirExactGroupAndRange()
    {
        TrafficNetwork map = ControlledCrossing(out JunctionController controller, out _, out _); controller.Validate(map, 0.05f);
        Assert.That(controller.HeadVisible("a", new Vector3(-10, 0, 0), 4), Is.True);
        Assert.That(controller.HeadVisible("b", new Vector3(-10, 0, 0), 4), Is.False);
        Assert.That(controller.HeadVisible("unbound", Vector3.zero, 100), Is.False);
    }

    [Test] public void ZoneOccupancyIncludesRotatedFootprintAndExcludesOverpass()
    {
        TrafficConflictZone zone = new TrafficConflictZone("zone", new[] { new Vector3(-1, 0, -1), new Vector3(-1, 0, 1), new Vector3(1, 0, 1), new Vector3(1, 0, -1) }, 0, 3);
        // The centre is outside: the front corner of a rotated long vehicle occupies the zone.
        Assert.That(zone.OverlapsFootprint(new Vector3(2, 0, 2), Mathf.PI / 4, 4.4f, 1.8f, 1.5f), Is.True);
        Assert.That(zone.OverlapsFootprint(new Vector3(0, 5, 0), 0, 4.4f, 1.8f, 1.5f), Is.False);
    }
}
#endif
