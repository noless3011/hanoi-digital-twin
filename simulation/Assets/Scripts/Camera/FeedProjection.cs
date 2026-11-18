using System;
using UnityEngine;

/// <summary>A camera's view at a video feed's resolution, computed without changing the camera.
/// Matches Camera.WorldToViewportPoint when the camera's aspect equals the feed's aspect.
/// Setting Camera.aspect instead would leave the camera's aspect fixed until ResetAspect.</summary>
public sealed class FeedProjection
{
    private readonly Matrix4x4 worldToClip, clipToWorld;
    public readonly Vector3 position, forward;
    public readonly float near, far;
    public readonly int width, height;
    /// <summary>World to clip space (OpenGL convention): viewport = (clip.xy / clip.w + 1) / 2.</summary>
    public Matrix4x4 WorldToClip => worldToClip;

    public FeedProjection(Camera camera, int width, int height)
    {
        if (camera == null) throw new ArgumentNullException(nameof(camera));
        if (camera.orthographic) throw new ArgumentException("The feed camera must use a perspective projection.");
        if (width <= 0 || height <= 0) throw new ArgumentException("The feed size must be positive.");
        float aspect = (float)width / height;
        Matrix4x4 projection;
        if (camera.usePhysicalProperties)
            Camera.CalculateProjectionMatrixFromPhysicalProperties(out projection, camera.focalLength, camera.sensorSize, camera.lensShift,
                camera.nearClipPlane, camera.farClipPlane, new Camera.GateFitParameters(camera.gateFit, aspect));
        else projection = Matrix4x4.Perspective(camera.fieldOfView, aspect, camera.nearClipPlane, camera.farClipPlane);
        worldToClip = projection * camera.worldToCameraMatrix;
        clipToWorld = worldToClip.inverse;
        position = camera.transform.position; forward = camera.transform.forward;
        near = camera.nearClipPlane; far = camera.farClipPlane;
        this.width = width; this.height = height;
    }

    /// <summary>Unity viewport coordinates (x right, y up; 0..1 inside the image) and depth along the view direction.</summary>
    public Vector3 WorldToViewport(Vector3 world)
    {
        Vector4 clip = worldToClip * new Vector4(world.x, world.y, world.z, 1);
        return new Vector3((clip.x / clip.w + 1) / 2, (clip.y / clip.w + 1) / 2, Vector3.Dot(world - position, forward));
    }

    /// <summary>Normalized feed image point (x right, y down), or null when behind the near plane.</summary>
    public Vector2? WorldToImage(Vector3 world)
    {
        Vector3 viewport = WorldToViewport(world);
        return viewport.z <= near ? (Vector2?)null : new Vector2(viewport.x, 1 - viewport.y);
    }

    /// <summary>Ray from the camera through a viewport point (x right, y up).</summary>
    public Ray ViewportToRay(float x, float y)
    {
        Vector4 p = clipToWorld * new Vector4(2 * x - 1, 2 * y - 1, 0, 1);
        return new Ray(position, (new Vector3(p.x, p.y, p.z) / p.w - position).normalized);
    }

    /// <summary>Normalized image box (x right, y down) of oriented local bounds, clipped at the near and far
    /// planes and to the image. Same rules as CameraBoxProjection.TryProject, at the feed's aspect.</summary>
    public bool TryProjectBox(Bounds localBounds, Matrix4x4 localToWorld, out Rect box)
    {
        box = default;
        var points = new Vector3[8]; var depth = new float[8];
        Vector3 min = localBounds.min, max = localBounds.max;
        float x1 = float.PositiveInfinity, y1 = float.PositiveInfinity, x2 = float.NegativeInfinity, y2 = float.NegativeInfinity;
        for (int i = 0; i < 8; i++)
        {
            points[i] = localToWorld.MultiplyPoint3x4(new Vector3((i & 1) == 0 ? min.x : max.x, (i & 2) == 0 ? min.y : max.y, (i & 4) == 0 ? min.z : max.z));
            depth[i] = Vector3.Dot(points[i] - position, forward);
            if (depth[i] >= near && depth[i] <= far) Include(points[i]);
        }
        for (int i = 0; i < 8; i++) for (int axis = 1; axis <= 4; axis <<= 1)
        {
            if ((i & axis) != 0) continue;
            int j = i | axis;
            Clip(i, j, near); Clip(i, j, far);
        }
        if (float.IsInfinity(x1)) return false;
        x1 = Mathf.Clamp01(x1); x2 = Mathf.Clamp01(x2); y1 = Mathf.Clamp01(y1); y2 = Mathf.Clamp01(y2);
        if (x2 <= x1 || y2 <= y1) return false;
        box = Rect.MinMaxRect(x1, 1 - y2, x2, 1 - y1);
        return true;

        void Clip(int i, int j, float plane)
        {
            if ((depth[i] < plane && depth[j] > plane) || (depth[j] < plane && depth[i] > plane))
                Include(Vector3.LerpUnclamped(points[i], points[j], (plane - depth[i]) / (depth[j] - depth[i])));
        }
        void Include(Vector3 point)
        {
            Vector3 p = WorldToViewport(point);
            if (float.IsNaN(p.x) || float.IsNaN(p.y) || float.IsInfinity(p.x) || float.IsInfinity(p.y)) return;
            x1 = Mathf.Min(x1, p.x); y1 = Mathf.Min(y1, p.y); x2 = Mathf.Max(x2, p.x); y2 = Mathf.Max(y2, p.y);
        }
    }
}
