using System.Collections.Generic;
using UnityEngine;

/// <summary>Projects traffic meshes once per rendered frame and displays box-only TrackTrack results.</summary>
[DisallowMultipleComponent]
[DefaultExecutionOrder(1000)]
public sealed class CameraTrackingOverlay : MonoBehaviour
{
    [Tooltip("Defaults to the camera on this object, then Camera.main.")]
    public Camera sourceCamera;
    public bool drawBoxes = true, drawTrails = true, drawLost = true;
    [Min(0)] public int maxMissed = 30;
    [Min(1)] public int trailLength = 40;
    [Min(1)] public float minimumBoxPixels = 2;
    private CameraTrackTracker tracker;
    private Camera previousCamera;
    private readonly List<CameraTrackTracker.Detection> detections = new List<CameraTrackTracker.Detection>();
    public CameraTrackTracker Tracker => tracker;
    private static readonly Color[] Colors = {
        new Color32(255,230,100,255), new Color32(100,170,255,255), new Color32(130,255,160,255), new Color32(255,135,220,255),
        new Color32(255,165,110,255), new Color32(100,235,245,255), new Color32(255,130,180,255), new Color32(215,255,140,255)
    };

    private void LateUpdate()
    {
        if (sourceCamera == null) sourceCamera = GetComponent<Camera>() ?? Camera.main;
        if (sourceCamera == null || !sourceCamera.isActiveAndEnabled) { ResetTracking(); return; }
        Capture();
    }

    /// <summary>Advance once per processed camera frame. No image readback or model inference.</summary>
    public void Capture()
    {
        if (sourceCamera == null || sourceCamera.pixelWidth < 2 || sourceCamera.pixelHeight < 2) return;
        if (tracker == null || tracker.MaxMissed != Mathf.Max(0, maxMissed) || tracker.TrailLength != Mathf.Max(1, trailLength))
            tracker = new CameraTrackTracker(Mathf.Max(0, maxMissed), Mathf.Max(1, trailLength));
        if (previousCamera != sourceCamera) tracker.Reset();
        previousCamera = sourceCamera;
        detections.Clear();
        foreach (var target in CameraTrackingTarget.Active)
        {
            if (target == null || !target.isActiveAndEnabled) continue;
            bool found = false; Rect combined = default;
            foreach (var renderer in target.Renderers)
            {
                if (renderer == null || !renderer.enabled || !renderer.gameObject.activeInHierarchy || renderer.forceRenderingOff ||
                    (sourceCamera.cullingMask & (1 << renderer.gameObject.layer)) == 0) continue;
                if (!CameraBoxProjection.TryProject(sourceCamera, renderer.localBounds, renderer.localToWorldMatrix, out Rect box)) continue;
                combined = found ? Rect.MinMaxRect(Mathf.Min(combined.xMin, box.xMin), Mathf.Min(combined.yMin, box.yMin),
                    Mathf.Max(combined.xMax, box.xMax), Mathf.Max(combined.yMax, box.yMax)) : box;
                found = true;
            }
            if (found && combined.width >= minimumBoxPixels && combined.height >= minimumBoxPixels)
                detections.Add(new CameraTrackTracker.Detection(combined, target.kind));
        }
        tracker.Update(detections, sourceCamera.pixelWidth, sourceCamera.pixelHeight);
    }

    [ContextMenu("Reset camera tracking")]
    public void ResetTracking() { tracker?.Reset(); }
    private void OnDisable() { ResetTracking(); }

    private void OnGUI()
    {
        if (!drawBoxes || tracker == null || sourceCamera == null || !sourceCamera.isActiveAndEnabled ||
            sourceCamera.targetTexture != null || sourceCamera.targetDisplay != 0 || Event.current.type != EventType.Repaint) return;
        Rect viewport = sourceCamera.pixelRect;
        viewport.y = Screen.height - viewport.yMax;
        Color previousColor = GUI.color;
        GUI.BeginGroup(viewport);
        foreach (var t in tracker.Tracks)
        {
            bool lost = t.Missed > 0;
            if (lost && !drawLost) continue;
            Color color = Colors[(t.Id - 1) % Colors.Length]; if (lost) color.a = .45f;
            GUI.color = color;
            Rect b = t.Box;
            Line(new Vector2(b.xMin, b.yMin), new Vector2(b.xMax, b.yMin), lost);
            Line(new Vector2(b.xMax, b.yMin), new Vector2(b.xMax, b.yMax), lost);
            Line(new Vector2(b.xMax, b.yMax), new Vector2(b.xMin, b.yMax), lost);
            Line(new Vector2(b.xMin, b.yMax), new Vector2(b.xMin, b.yMin), lost);
            if (drawTrails) for (int i = 1; i < t.Trail.Count; i++) Line(t.Trail[i - 1], t.Trail[i], false);
            GUI.Label(new Rect(Mathf.Clamp(b.xMin, 0, Mathf.Max(0, viewport.width - 260)), Mathf.Max(0, b.yMin - 20), 260, 20),
                "#" + t.Id + " " + t.Kind + " | " + t.State + (lost ? " (" + t.Missed + ")" : ""));
        }
        GUI.EndGroup(); GUI.color = previousColor;
    }

    private static void Line(Vector2 a, Vector2 b, bool dashed)
    {
        Vector2 delta = b - a; float length = delta.magnitude;
        if (length < .01f) return;
        Matrix4x4 old = GUI.matrix;
        GUIUtility.RotateAroundPivot(Mathf.Atan2(delta.y, delta.x) * Mathf.Rad2Deg, a);
        if (dashed) for (float x = 0; x < length; x += 14) GUI.DrawTexture(new Rect(a.x + x, a.y, Mathf.Min(7, length - x), 1), Texture2D.whiteTexture);
        else GUI.DrawTexture(new Rect(a.x, a.y, length, 2), Texture2D.whiteTexture);
        GUI.matrix = old;
    }
}
