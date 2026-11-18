using System.Collections.Generic;
using UnityEngine;

/// <summary>Lane cross-section frames, lane fill strips, and shared junction patches.</summary>
public static class RoadMeshGeometry
{
    public struct Frame
    {
        public Vector3 center, side, up, left, right;
        public Quaternion rotation, automaticRotation;
    }

    private const float DirectionEpsilon = 0.000001f;

    public static Frame[] Frames(LaneEditor editor, LaneEditor.LaneStrip lane)
    {
        if (editor == null || lane == null || lane.sections == null) return new Frame[0];
        Frame[] frames = new Frame[lane.sections.Count];
        Vector3 fallbackCenter = Finite(editor.transform.position) ? editor.transform.position : Vector3.zero;
        for (int i = 0; i < frames.Length; i++)
        {
            LaneEditor.Section section = lane.sections[i];
            LaneEditor.LaneNode node = section != null ? editor.FindNode(section.nodeId) : null;
            Vector3 center = node != null ? editor.transform.TransformPoint(node.position) : fallbackCenter;
            frames[i].center = Finite(center) ? center : fallbackCenter;
            fallbackCenter = frames[i].center;
        }

        Vector3 fallbackForward = SafeDirection(editor.transform.forward, Vector3.forward);
        Vector3 previousSide = SafeDirection(editor.transform.right, Vector3.right);
        for (int i = 0; i < frames.Length; i++)
        {
            // Look past coincident nodes so dragging a section over its neighbour stays stable.
            Vector3 before = NeighbourDirection(frames, i, -1);
            Vector3 after = NeighbourDirection(frames, i, 1);
            if (before.sqrMagnitude < DirectionEpsilon) before = after;
            if (after.sqrMagnitude < DirectionEpsilon) after = before;
            before = SafeDirection(before, fallbackForward);
            after = SafeDirection(after, before);
            Vector3 tangent = SafeDirection(before + after, after);

            Vector3 automaticSide = SideFor(tangent, previousSide);
            Vector3 automaticUp = SafeDirection(Vector3.Cross(tangent, automaticSide), Vector3.up);
            Quaternion automaticRotation = Quaternion.LookRotation(tangent, automaticUp);
            LaneEditor.Section section = lane.sections[i];
            Vector3 offset = section != null && Finite(section.rotation) ? section.rotation : Vector3.zero;
            Quaternion rotation = automaticRotation * Quaternion.Euler(offset);

            // A capped miter closes ordinary bends without producing very long spikes at hairpins.
            Vector3 outgoingSide = SideFor(after, automaticSide);
            float miterDot = Mathf.Abs(Vector3.Dot(automaticSide, outgoingSide));
            float extension = Mathf.Min(2f, 1f / Mathf.Max(0.25f, miterDot));
            float width = Finite(lane.width) ? Mathf.Max(0.01f, lane.width) : 3.2f;
            float scale = section != null && Finite(section.widthScale) ? Mathf.Max(0.01f, section.widthScale) : 1f;
            Vector3 side = rotation * Vector3.right;
            float halfWidth = width * scale * 0.5f * extension;
            if (!Finite(halfWidth)) halfWidth = width * 0.5f;
            Vector3 half = side * halfWidth;

            frames[i].automaticRotation = automaticRotation;
            frames[i].rotation = rotation;
            frames[i].side = side;
            frames[i].up = rotation * Vector3.up;
            frames[i].left = frames[i].center - half;
            frames[i].right = frames[i].center + half;
            previousSide = automaticSide;
        }
        return frames;
    }

    private static Vector3 NeighbourDirection(Frame[] frames, int index, int step)
    {
        for (int other = index + step; other >= 0 && other < frames.Length; other += step)
        {
            Vector3 direction = step < 0 ? frames[index].center - frames[other].center : frames[other].center - frames[index].center;
            if (Finite(direction) && Finite(direction.sqrMagnitude) && direction.sqrMagnitude >= DirectionEpsilon) return direction.normalized;
        }
        return Vector3.zero;
    }

    private static Vector3 SideFor(Vector3 forward, Vector3 fallbackSide)
    {
        Vector3 side = Vector3.Cross(Vector3.up, forward);
        if (side.sqrMagnitude < DirectionEpsilon) side = Vector3.ProjectOnPlane(fallbackSide, forward);
        if (side.sqrMagnitude < DirectionEpsilon) side = Vector3.Cross(Vector3.forward, forward);
        return SafeDirection(side, Vector3.right);
    }

    private static Vector3 SafeDirection(Vector3 direction, Vector3 fallback)
    {
        return Finite(direction) && Finite(direction.sqrMagnitude) && direction.sqrMagnitude >= DirectionEpsilon
            ? direction.normalized : fallback;
    }

    private static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
    private static bool Finite(Vector3 value) => Finite(value.x) && Finite(value.y) && Finite(value.z);

    /// <summary>A lane's surface: one quad between each pair of cross-sections.</summary>
    public static Mesh Strip(Transform owner, Frame[] frames)
    {
        if (owner == null || frames == null) return null;
        const int cells = 1;
        List<Vector3> vertices = new List<Vector3>();
        List<Vector3> worldPoints = new List<Vector3>();
        List<Vector3> worldNormals = new List<Vector3>();
        List<Vector2> uv = new List<Vector2>();
        List<int> triangles = new List<int>();
        float length = 0;
        for (int row = 0; row < frames.Length; row++)
        {
            if (row > 0) length += Vector3.Distance(frames[row - 1].center, frames[row].center);
            for (int column = 0; column <= cells; column++)
            {
                float across = (float)column / cells;
                Vector3 point = Vector3.Lerp(frames[row].left, frames[row].right, across);
                worldPoints.Add(point);
                vertices.Add(owner.InverseTransformPoint(point));
                worldNormals.Add(Vector3.zero);
                uv.Add(new Vector2(across, length));
            }
            if (row == 0) continue;
            Vector3 up = SafeDirection(frames[row - 1].up + frames[row].up, frames[row - 1].up);
            for (int column = 0; column < cells; column++)
            {
                int a = (row - 1) * (cells + 1) + column, b = a + 1;
                int c = row * (cells + 1) + column, d = c + 1;
                AddOrientedTriangle(worldPoints, triangles, a, c, b, up, worldNormals);
                AddOrientedTriangle(worldPoints, triangles, b, c, d, up, worldNormals);
            }
        }

        Mesh mesh = MakeMesh("Lane quad strip", vertices, uv, triangles);
        List<Vector3> normals = new List<Vector3>(worldNormals.Count);
        Matrix4x4 worldNormalToLocal = owner.localToWorldMatrix.transpose;
        for (int i = 0; i < worldNormals.Count; i++)
        {
            Vector3 worldNormal = SafeDirection(worldNormals[i], frames[i / (cells + 1)].up);
            // The transpose is the inverse of Unity's normal transform, including nonuniform scaling.
            normals.Add(SafeDirection(worldNormalToLocal.MultiplyVector(worldNormal), Vector3.up));
        }
        mesh.SetNormals(normals);
        return mesh;
    }

    private static void AddOrientedTriangle(List<Vector3> points, List<int> triangles, int a, int b, int c,
        Vector3 referenceNormal, List<Vector3> normals = null)
    {
        Vector3 normal = Vector3.Cross(points[b] - points[a], points[c] - points[a]);
        if (!Finite(normal) || normal.sqrMagnitude < 0.0000000001f) return;
        if (Vector3.Dot(normal, referenceNormal) < 0)
        {
            int swap = b; b = c; c = swap;
            normal = -normal;
        }
        triangles.Add(a); triangles.Add(b); triangles.Add(c);
        if (normals == null) return;
        normals[a] += normal; normals[b] += normal; normals[c] += normal;
    }

    private static Mesh MakeMesh(string name, List<Vector3> vertices, List<Vector2> uv, List<int> triangles)
    {
        Mesh mesh = new Mesh { name = name, hideFlags = HideFlags.HideAndDontSave };
        if (vertices.Count > 65535) mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        mesh.SetVertices(vertices); mesh.SetUVs(0, uv); mesh.SetTriangles(triangles, 0);
        mesh.RecalculateNormals(); mesh.RecalculateBounds();
        return mesh;
    }

    public static Mesh Junction(Transform owner, List<Vector3> edges)
    {
        if (owner == null || edges == null || edges.Count < 4) return null;
        List<Vector3> hull = Hull(edges);
        if (hull.Count < 3) return null;
        List<Vector3> vertices = new List<Vector3>();
        List<Vector2> uv = new List<Vector2>();
        List<int> triangles = new List<int>();
        foreach (Vector3 point in hull)
        {
            vertices.Add(owner.InverseTransformPoint(point));
            uv.Add(new Vector2(point.x, point.z));
        }
        for (int i = 1; i < hull.Count - 1; i++) AddOrientedTriangle(hull, triangles, 0, i, i + 1, Vector3.up);
        return MakeMesh("Shared lane junction patch", vertices, uv, triangles);
    }

    private static List<Vector3> Hull(List<Vector3> points)
    {
        List<Vector3> sorted = new List<Vector3>();
        foreach (Vector3 point in points) if (Finite(point)) sorted.Add(point);
        sorted.Sort((a, b) => a.x == b.x ? a.z.CompareTo(b.z) : a.x.CompareTo(b.x));
        List<Vector3> distinct = new List<Vector3>();
        foreach (Vector3 point in sorted)
        {
            if (distinct.Count == 0) { distinct.Add(point); continue; }
            Vector3 previous = distinct[distinct.Count - 1];
            if (Mathf.Abs(point.x - previous.x) > 0.0001f || Mathf.Abs(point.z - previous.z) > 0.0001f) distinct.Add(point);
        }
        if (distinct.Count < 3) return distinct;
        List<Vector3> hull = new List<Vector3>();
        foreach (Vector3 point in distinct)
        {
            while (hull.Count >= 2 && Cross(hull[hull.Count - 2], hull[hull.Count - 1], point) <= 0) hull.RemoveAt(hull.Count - 1);
            hull.Add(point);
        }
        int lower = hull.Count;
        for (int i = distinct.Count - 2; i >= 0; i--)
        {
            while (hull.Count > lower && Cross(hull[hull.Count - 2], hull[hull.Count - 1], distinct[i]) <= 0) hull.RemoveAt(hull.Count - 1);
            hull.Add(distinct[i]);
        }
        hull.RemoveAt(hull.Count - 1);
        return hull;
    }

    private static float Cross(Vector3 a, Vector3 b, Vector3 c) => (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x);
}
