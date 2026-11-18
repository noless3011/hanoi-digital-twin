using UnityEngine;

/// <summary>Projects oriented renderer-local bounds, clipped to the camera depth range and image.</summary>
public static class CameraBoxProjection
{
    /// <summary>Returns camera-local pixels, top-left origin. Does not perform occlusion testing.</summary>
    public static bool TryProject(Camera camera, Bounds localBounds, Matrix4x4 localToWorld, out Rect box)
    {
        box = default;
        if (camera == null || camera.pixelWidth < 2 || camera.pixelHeight < 2) return false;
        var points = new Vector3[8]; var depth = new float[8];
        Vector3 min = localBounds.min, max = localBounds.max;
        float near = camera.nearClipPlane, far = camera.farClipPlane;
        float x1 = float.PositiveInfinity, y1 = float.PositiveInfinity, x2 = float.NegativeInfinity, y2 = float.NegativeInfinity;
        for (int i = 0; i < 8; i++)
        {
            points[i] = localToWorld.MultiplyPoint3x4(new Vector3((i & 1) == 0 ? min.x : max.x, (i & 2) == 0 ? min.y : max.y, (i & 4) == 0 ? min.z : max.z));
            depth[i] = Vector3.Dot(points[i] - camera.transform.position, camera.transform.forward);
            if (depth[i] >= near && depth[i] <= far) Include(points[i]);
        }
        // Clip all twelve box edges at both depth planes before perspective division.
        for (int i = 0; i < 8; i++) for (int axis = 1; axis <= 4; axis <<= 1)
        {
            if ((i & axis) != 0) continue;
            int j = i | axis;
            Clip(i, j, near); Clip(i, j, far);
        }
        if (float.IsInfinity(x1)) return false;
        x1 = Mathf.Clamp01(x1); x2 = Mathf.Clamp01(x2); y1 = Mathf.Clamp01(y1); y2 = Mathf.Clamp01(y2);
        if (x2 <= x1 || y2 <= y1) return false;
        box = Rect.MinMaxRect(x1 * (camera.pixelWidth - 1), (1 - y2) * (camera.pixelHeight - 1),
            x2 * (camera.pixelWidth - 1), (1 - y1) * (camera.pixelHeight - 1));
        return true;

        void Clip(int i, int j, float plane)
        {
            if ((depth[i] < plane && depth[j] > plane) || (depth[j] < plane && depth[i] > plane))
                Include(Vector3.LerpUnclamped(points[i], points[j], (plane - depth[i]) / (depth[j] - depth[i])));
        }
        void Include(Vector3 point)
        {
            Vector3 p = camera.WorldToViewportPoint(point);
            if (float.IsNaN(p.x) || float.IsNaN(p.y) || float.IsInfinity(p.x) || float.IsInfinity(p.y)) return;
            x1 = Mathf.Min(x1, p.x); y1 = Mathf.Min(y1, p.y); x2 = Mathf.Max(x2, p.x); y2 = Mathf.Max(y2, p.y);
        }
    }
}
