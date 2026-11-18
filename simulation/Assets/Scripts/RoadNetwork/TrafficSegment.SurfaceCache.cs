#if LEGACY_SUBLANE_RUNTIME // Old Unity simulation map; unused since the sublane system was removed.
using System.Collections.Concurrent;
using UnityEngine;

public sealed partial class TrafficSegment
{
    private volatile SurfaceGeometry surfaceGeometry;
    private readonly ConcurrentDictionary<(float, float, int), bool> feasibilityCache = new ConcurrentDictionary<(float, float, int), bool>();
    private readonly object surfaceGeometryLock = new object();
    private sealed class SurfaceGeometry
    {
        public SurfaceTriangle[] triangles;
        public readonly ConcurrentDictionary<float, SurfaceTriangle[]> start = new ConcurrentDictionary<float, SurfaceTriangle[]>();
        public readonly ConcurrentDictionary<float, SurfaceTriangle[]> end = new ConcurrentDictionary<float, SurfaceTriangle[]>();
    }
    internal readonly struct SurfaceTriangle
    {
        private readonly Vector3 a;
        private readonly float abX, abZ, acX, acZ, abY, acY, determinant;
        private readonly float minX, maxX, minZ, maxZ;
        public SurfaceTriangle(Vector3 a, Vector3 b, Vector3 c)
        {
            this.a = a;
            abX = b.x - a.x; abZ = b.z - a.z; acX = c.x - a.x; acZ = c.z - a.z;
            abY = b.y - a.y; acY = c.y - a.y;
            determinant = abX * acZ - abZ * acX;
            // Conservative bounds include the original barycentric edge tolerance.
            float paddingX = (Mathf.Abs(abX) + Mathf.Abs(acX) + 1) * 0.00001f;
            float paddingZ = (Mathf.Abs(abZ) + Mathf.Abs(acZ) + 1) * 0.00001f;
            minX = Mathf.Min(a.x, b.x, c.x) - paddingX; maxX = Mathf.Max(a.x, b.x, c.x) + paddingX;
            minZ = Mathf.Min(a.z, b.z, c.z) - paddingZ; maxZ = Mathf.Max(a.z, b.z, c.z) + paddingZ;
        }
        public bool Contains(Vector3 point, float verticalTolerance)
        {
            if (point.x < minX || point.x > maxX || point.z < minZ || point.z > maxZ || Mathf.Abs(determinant) < 0.00000001f) return false;
            float apX = point.x - a.x, apZ = point.z - a.z;
            float u = (apX * acZ - apZ * acX) / determinant, v = (abX * apZ - abZ * apX) / determinant;
            if (u < -0.000001f || v < -0.000001f || u + v > 1 + 0.000001f) return false;
            return Mathf.Abs(point.y - (a.y + u * abY + v * acY)) <= verticalTolerance;
        }
    }
    internal readonly struct SurfaceQuery
    {
        private readonly SurfaceTriangle[] body, start, end;
        internal SurfaceQuery(SurfaceTriangle[] body, SurfaceTriangle[] start, SurfaceTriangle[] end)
        { this.body = body; this.start = start; this.end = end; }
        public bool Contains(Vector3 point)
        {
            for (int i = 0; i < body.Length; i++) if (body[i].Contains(point, 0.75f)) return true;
            for (int i = 0; i < start.Length; i++) if (start[i].Contains(point, 0.75f)) return true;
            for (int i = 0; i < end.Length; i++) if (end[i].Contains(point, 0.75f)) return true;
            return false;
        }
    }
    internal SurfaceQuery PrepareSurfaceQuery(float startOverhang, float endOverhang)
    {
        var surface = GetSurfaceGeometry();
        return new SurfaceQuery(surface.triangles,
            startOverhang > 0 ? GetEndpointTriangles(surface, true, startOverhang) : System.Array.Empty<SurfaceTriangle>(),
            endOverhang > 0 ? GetEndpointTriangles(surface, false, endOverhang) : System.Array.Empty<SurfaceTriangle>());
    }
    private SurfaceTriangle[] GetEndpointTriangles(SurfaceGeometry surface, bool first, float amount)
    {
        var cache = first ? surface.start : surface.end;
        if (cache.TryGetValue(amount, out var triangles)) return triangles;
        triangles = EndpointTriangles(first, amount);
        if (cache.Count < 256) cache.TryAdd(amount, triangles);
        return triangles;
    }
    private SurfaceGeometry GetSurfaceGeometry()
    {
        if (surfaceGeometry != null) return surfaceGeometry;
        lock (surfaceGeometryLock)
        {
            if (surfaceGeometry != null) return surfaceGeometry;
            var result = new SurfaceGeometry { triangles = new SurfaceTriangle[(points.Length - 1) * 2] };
            for (int i = 1; i < points.Length; i++)
            {
                result.triangles[(i - 1) * 2] = new SurfaceTriangle(left[i - 1], left[i], right[i - 1]);
                result.triangles[(i - 1) * 2 + 1] = new SurfaceTriangle(right[i - 1], left[i], right[i]);
            }
            surfaceGeometry = result;
            return result;
        }
    }
    private SurfaceTriangle[] EndpointTriangles(bool first, float amount)
    {
        int index = first ? 0 : points.Length - 1, next = first ? 1 : index - 1;
        Vector3 tangent = first ? points[next] - points[index] : points[index] - points[next];
        float distance = TrafficNetwork.HorizontalDistance(points[index], points[next]);
        if (distance <= 0.0001f) return new SurfaceTriangle[0];
        Vector3 extension = tangent * (amount / distance);
        Vector3 a = first ? left[index] - extension : left[index];
        Vector3 b = first ? right[index] - extension : right[index];
        Vector3 c = first ? left[index] : left[index] + extension;
        Vector3 d = first ? right[index] : right[index] + extension;
        return new[] { new SurfaceTriangle(a, c, b), new SurfaceTriangle(b, c, d) };
    }
}
#endif
