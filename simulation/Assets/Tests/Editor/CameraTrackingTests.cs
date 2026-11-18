#if UNITY_INCLUDE_TESTS
using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

public sealed class CameraTrackingTests
{
    private static CameraTrackTracker.Detection D(float x = 100, float score = 1) => new CameraTrackTracker.Detection(new Rect(x, 100, 60, 40), "car", score);
    private static readonly CameraTrackTracker.Detection[] Empty = new CameraTrackTracker.Detection[0];

    [Test] public void ConfirmsRecoversAndExpiresWithObservedTrailsOnly()
    {
        var tracker = new CameraTrackTracker(2, 3);
        tracker.Update(new[] { D() }, 640, 480);
        Assert.That(tracker.Tracks[0].State, Is.EqualTo("NEW"));
        tracker.Update(new[] { D(102) }, 640, 480);
        Assert.That(tracker.Tracks[0].State, Is.EqualTo("NEW"));
        tracker.Update(new[] { D(104) }, 640, 480);
        Assert.That(tracker.Tracks[0].State, Is.EqualTo("TRACKED"));
        tracker.Update(Empty, 640, 480);
        Assert.That(tracker.Tracks[0].State, Is.EqualTo("LOST"));
        Assert.That(tracker.Tracks[0].Box.x, Is.EqualTo(104));
        Assert.That(tracker.Tracks[0].Trail.Count, Is.EqualTo(3));
        tracker.Update(new[] { D(108) }, 640, 480);
        Assert.That(tracker.Tracks[0].State, Is.EqualTo("RECOVERED"));
        Assert.That(tracker.Tracks[0].Id, Is.EqualTo(1));
        Assert.That(tracker.Tracks[0].Trail[2], Is.EqualTo(new Vector2(138, 140)));
        for (int i = 0; i < 3; i++) tracker.Update(Empty, 640, 480);
        Assert.That(tracker.Tracks, Is.Empty);
    }

    [Test] public void LowConfidenceRetainsConfirmedTracksButCannotInitializeOrConfirmTentativeTracks()
    {
        var tracker = new CameraTrackTracker();
        tracker.Update(new[] { D(100, .2f) }, 640, 480);
        Assert.That(tracker.Tracks, Is.Empty);
        for (int i = 0; i < 3; i++) tracker.Update(new[] { D() }, 640, 480);
        tracker.Update(new[] { D(101, .2f) }, 640, 480);
        Assert.That(tracker.Tracks[0].Id, Is.EqualTo(1));
        Assert.That(tracker.Tracks[0].Missed, Is.Zero);
        var tentative = new CameraTrackTracker();
        tentative.Update(new[] { D() }, 640, 480);
        tentative.Update(new[] { D(100, .2f) }, 640, 480);
        Assert.That(tentative.Tracks, Is.Empty);
    }

    [Test] public void AssociationDoesNotDependOnDetectionOrderOrObjectIdentity()
    {
        var tracker = new CameraTrackTracker();
        for (int i = 0; i < 5; i++)
        {
            var a = D(100 + i * 3); var b = D(400 - i * 3);
            tracker.Update(i % 2 == 0 ? new[] { a, b } : new[] { b, a }, 640, 480);
        }
        Assert.That(tracker.Tracks.Count, Is.EqualTo(2));
        Assert.That(tracker.Tracks[0].Box.x, Is.EqualTo(112));
        Assert.That(tracker.Tracks[1].Box.x, Is.EqualTo(388));
        Assert.That(tracker.TotalIds, Is.EqualTo(2));
    }

    [Test] public void ResizingRetiresTracksWithoutReusingIdsAndRejectsInvalidInput()
    {
        var tracker = new CameraTrackTracker();
        tracker.Update(new[] { D() }, 640, 480);
        tracker.Update(new[] { D() }, 1280, 720);
        Assert.That(tracker.Tracks[0].Id, Is.EqualTo(2));
        Assert.That(tracker.Tracks[0].State, Is.EqualTo("NEW"));
        Assert.Throws<ArgumentException>(() => tracker.Update(new[] { D(float.NaN) }, 640, 480));
    }

    [Test] public void ProjectsPerspectiveOrthographicAndClipsNearFarAndViewport()
    {
        var go = new GameObject("Projection camera");
        var texture = new RenderTexture(640, 480, 0);
        try
        {
            var camera = go.AddComponent<Camera>(); camera.targetTexture = texture;
            camera.nearClipPlane = .1f; camera.farClipPlane = 100; camera.fieldOfView = 60;
            var bounds = new Bounds(Vector3.zero, Vector3.one * 2);
            Assert.That(CameraBoxProjection.TryProject(camera, bounds, Matrix4x4.Translate(new Vector3(0, 0, 10)), out Rect box), Is.True);
            Assert.That(box.center.x, Is.EqualTo(319.5f).Within(.1));
            Assert.That(box.center.y, Is.EqualTo(239.5f).Within(.1));
            Assert.That(box.width, Is.GreaterThan(50));
            Assert.That(CameraBoxProjection.TryProject(camera, bounds, Matrix4x4.Translate(new Vector3(0, 0, -10)), out _), Is.False);
            Assert.That(CameraBoxProjection.TryProject(camera, bounds, Matrix4x4.Translate(new Vector3(0, 0, 102)), out _), Is.False);
            Assert.That(CameraBoxProjection.TryProject(camera, bounds, Matrix4x4.Translate(new Vector3(100, 0, 10)), out _), Is.False);
            Assert.That(CameraBoxProjection.TryProject(camera, bounds, Matrix4x4.Translate(new Vector3(0, 0, .15f)), out box), Is.True);
            Assert.That(box.xMin, Is.Zero); Assert.That(box.xMax, Is.EqualTo(639));
            camera.orthographic = true; camera.orthographicSize = 5;
            Assert.That(CameraBoxProjection.TryProject(camera, bounds, Matrix4x4.Translate(new Vector3(0, 0, 10)), out box), Is.True);
            Assert.That(box.height, Is.EqualTo(95.8f).Within(.1));
            Assert.That(CameraBoxProjection.TryProject(camera, bounds, Matrix4x4.Translate(new Vector3(6, 0, 10)), out box), Is.True);
            Assert.That(box.xMax, Is.EqualTo(639));
        }
        finally { UnityEngine.Object.DestroyImmediate(go); UnityEngine.Object.DestroyImmediate(texture); }
    }

    [Test] public void OverlayCombinesMeshChildrenAndHonorsRendererAndLayerVisibility()
    {
        var go = new GameObject("Projection camera"); var root = new GameObject("Vehicle");
        var texture = new RenderTexture(640, 480, 0);
        try
        {
            var camera = go.AddComponent<Camera>(); camera.targetTexture = texture;
            var overlay = go.AddComponent<CameraTrackingOverlay>(); overlay.sourceCamera = camera;
            var child = GameObject.CreatePrimitive(PrimitiveType.Cube); child.transform.SetParent(root.transform, false);
            root.transform.position = new Vector3(0, 0, 10);
            var target = root.AddComponent<CameraTrackingTarget>(); target.kind = "motorcycle";
            // Runtime MonoBehaviour lifecycle is driven explicitly by this Edit Mode test.
            typeof(CameraTrackingTarget).GetMethod("OnEnable", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic).Invoke(target, null);
            overlay.Capture(); overlay.Capture(); overlay.Capture();
            Assert.That(overlay.Tracker.Visible, Is.EqualTo(1));
            Assert.That(overlay.Tracker.Tracks[0].Kind, Is.EqualTo("motorcycle"));
            child.GetComponent<Renderer>().enabled = false; overlay.Capture();
            Assert.That(overlay.Tracker.Lost, Is.EqualTo(1));
            child.GetComponent<Renderer>().enabled = true; camera.cullingMask = 0; overlay.Capture();
            Assert.That(overlay.Tracker.Visible, Is.Zero);
            camera.cullingMask = -1; overlay.Capture();
            Assert.That(overlay.Tracker.Tracks[0].State, Is.EqualTo("RECOVERED"));
            root.SetActive(false); overlay.Capture(); Assert.That(overlay.Tracker.Visible, Is.Zero);
        }
        finally
        {
            var target = root.GetComponent<CameraTrackingTarget>();
            if (target != null) typeof(CameraTrackingTarget).GetMethod("OnDisable", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic).Invoke(target, null);
            UnityEngine.Object.DestroyImmediate(root); UnityEngine.Object.DestroyImmediate(go); UnityEngine.Object.DestroyImmediate(texture);
        }
    }
}
#endif
