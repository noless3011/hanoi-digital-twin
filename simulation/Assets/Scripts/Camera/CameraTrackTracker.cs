using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

/// <summary>
/// Box-only TrackTrack, matching perception/tracktrack.yaml and vehicle_tracking.py.
/// Fixed camera: no ReID, GMC, detector, or Unity object identity in association.
/// Algorithm reference: Ultralytics 8.4.171 track_tracker.py (AGPL-3.0).
/// Coordinates are camera-local pixels with the origin at the top left.
/// </summary>
public sealed class CameraTrackTracker
{
    public readonly struct Detection
    {
        public readonly Rect box;
        public readonly string kind;
        public readonly float confidence;
        public Detection(Rect box, string kind, float confidence = 1) { this.box = box; this.kind = kind; this.confidence = confidence; }
    }

    public sealed class Track
    {
        public int Id { get; internal set; }
        public string Kind { get; internal set; }
        public float Confidence { get; internal set; }
        public Rect Box { get; internal set; }
        public string State { get; internal set; }
        public int Hits { get; internal set; }
        public int Missed { get; internal set; }
        public IReadOnlyList<Vector2> Trail => trail;
        internal readonly List<Vector2> trail = new List<Vector2>();
    }

    private enum State { New, Tracked, Lost, Removed }
    private sealed class Entry
    {
        public readonly Track view = new Track();
        // The XYWH Kalman covariance consists of four independent position/velocity blocks.
        public readonly double[] mean = new double[8], pp = new double[4], pv = new double[4], vv = new double[4];
        public readonly List<(int frame, Rect box)> history = new List<(int, Rect)>();
        public readonly Vector2[] velocity = new Vector2[4];
        public State state;
        public int start, last, updates;
        public float previousScore;
        public Rect Predicted => new Rect((float)(mean[0] - mean[2] / 2), (float)(mean[1] - mean[3] / 2), (float)mean[2], (float)mean[3]);
        public Rect History(int frame)
        {
            foreach (var h in history) if (h.frame == frame) return h.box;
            return history.Count == 0 ? Predicted : history[history.Count - 1].box;
        }
        public void Remember(int frame, Rect box)
        {
            history.Add((frame, box));
            if (history.Count > 4) history.RemoveAt(0);
        }
        public void Predict()
        {
            if (state != State.Tracked) mean[6] = mean[7] = 0;
            for (int i = 0; i < 4; i++)
            {
                double size = mean[2 + i % 2], q = size / 20, v = size / 160;
                mean[i] += mean[i + 4];
                pp[i] += 2 * pv[i] + vv[i] + q * q;
                pv[i] += vv[i]; vv[i] += v * v;
            }
        }
    }

    private readonly List<Entry> entries = new List<Entry>();
    private readonly List<Track> views = new List<Track>();
    private int frame, width, height, totalIds;
    public int MaxMissed { get; }
    public int TrailLength { get; }
    public int TotalIds => totalIds;
    public IReadOnlyList<Track> Tracks => views;
    public int Visible => views.Count(t => t.Missed == 0);
    public int Lost => views.Count(t => t.Missed > 0);

    public CameraTrackTracker(int maxMissed = 30, int trailLength = 40)
    {
        if (maxMissed < 0 || trailLength < 1) throw new ArgumentOutOfRangeException();
        MaxMissed = maxMissed; TrailLength = trailLength;
    }

    /// <summary>Retire associations; IDs keep increasing unless a new session is requested.</summary>
    public void Reset(bool newSession = false)
    {
        entries.Clear(); views.Clear(); frame = width = height = 0;
        if (newSession) totalIds = 0;
    }

    public IReadOnlyList<Track> Update(IReadOnlyList<Detection> detections, int imageWidth, int imageHeight)
    {
        if (detections == null) throw new ArgumentNullException(nameof(detections));
        if (imageWidth < 2 || imageHeight < 2) throw new ArgumentOutOfRangeException();
        var high = new List<Detection>(); var low = new List<Detection>();
        foreach (var d in detections)
        {
            if (!Finite(d.box.xMin) || !Finite(d.box.yMin) || !Finite(d.box.xMax) || !Finite(d.box.yMax) ||
                !Finite(d.confidence) || d.confidence < 0 || d.confidence > 1 || d.box.width < 0 || d.box.height < 0)
                throw new ArgumentException("Detections require finite ordered boxes and confidence in [0,1].");
            var b = Rect.MinMaxRect(Mathf.Clamp(d.box.xMin, 0, imageWidth - 1), Mathf.Clamp(d.box.yMin, 0, imageHeight - 1),
                Mathf.Clamp(d.box.xMax, 0, imageWidth - 1), Mathf.Clamp(d.box.yMax, 0, imageHeight - 1));
            if (b.width <= 0 || b.height <= 0) continue;
            var clipped = new Detection(b, d.kind, d.confidence);
            if (d.confidence >= .35f) high.Add(clipped);
            else if (d.confidence > .10f) low.Add(clipped);
        }
        if (width != 0 && (width != imageWidth || height != imageHeight)) Reset();
        width = imageWidth; height = imageHeight; frame++;
        var tentative = entries.Where(t => t.state == State.New).ToList();
        var pool = entries.Where(t => t.state == State.Tracked).Concat(entries.Where(t => t.state == State.Lost)).ToList();
        foreach (var t in entries) t.Predict();
        var all = high.Concat(low).ToList();
        var matched = Associate(pool, all, high.Count);
        var used = new HashSet<int>();
        var activated = new List<Entry>();
        foreach (var pair in matched)
        {
            Observe(pool[pair.track], all[pair.det], pool[pair.track].state == State.Lost);
            used.Add(pair.det); activated.Add(pool[pair.track]);
        }
        var matchedTracks = new HashSet<int>(matched.Select(p => p.track));
        for (int i = 0; i < pool.Count; i++) if (!matchedTracks.Contains(i)) pool[i].state = State.Lost;
        var leftover = high.Where((d, i) => !used.Contains(i)).ToList();
        var confirmed = Associate(tentative, leftover, leftover.Count);
        used.Clear(); matchedTracks.Clear();
        foreach (var pair in confirmed)
        {
            Observe(tentative[pair.track], leftover[pair.det], false);
            activated.Add(tentative[pair.track]); used.Add(pair.det); matchedTracks.Add(pair.track);
        }
        for (int i = 0; i < tentative.Count; i++) if (!matchedTracks.Contains(i)) tentative[i].state = State.Removed;
        leftover = leftover.Where((d, i) => !used.Contains(i)).ToList();
        // Track-aware initialization: suppress against active tracks and stronger leftover detections.
        var active = entries.Where(t => t.state == State.Tracked).Concat(activated).ToList();
        var allow = leftover.Select(d => d.confidence > .40f && !active.Any(t => IoU(t.Predicted, d.box) > .55f)).ToArray();
        foreach (int i in Enumerable.Range(0, leftover.Count).OrderByDescending(i => leftover[i].confidence).ThenByDescending(i => i))
        {
            if (!allow[i]) continue;
            for (int j = 0; j < leftover.Count; j++) if (j != i && IoU(leftover[i].box, leftover[j].box) > .55f) allow[j] = false;
        }
        for (int i = 0; i < leftover.Count; i++) if (allow[i]) entries.Add(Create(leftover[i]));
        foreach (var t in entries) if (t.state == State.Lost && frame - t.last > MaxMissed) t.state = State.Removed;
        // Standard TrackTrack active/lost duplicate removal keeps the longer-lived track.
        var duplicates = new HashSet<Entry>();
        foreach (var a in entries.Where(t => t.state == State.New || t.state == State.Tracked))
            foreach (var b in entries.Where(t => t.state == State.Lost))
                if (IoU(a.Predicted, b.Predicted) > .85f) duplicates.Add(a.last - a.start > b.last - b.start ? b : a);
        entries.RemoveAll(t => t.state == State.Removed || duplicates.Contains(t));
        views.Clear();
        foreach (var t in entries)
        {
            t.view.Missed = frame - t.last;
            if (t.state == State.Lost) t.view.State = "LOST";
            views.Add(t.view);
        }
        views.Sort((a, b) => a.Id.CompareTo(b.Id));
        return views;
    }

    private Entry Create(Detection d)
    {
        var t = new Entry { state = State.New, start = frame, last = frame, previousScore = d.confidence };
        var m = Measurement(d.box);
        for (int i = 0; i < 4; i++)
        {
            t.mean[i] = m[i]; double size = m[2 + i % 2];
            t.pp[i] = Math.Pow(size / 10, 2); t.vv[i] = Math.Pow(size / 16, 2);
        }
        t.Remember(frame, t.Predicted);
        t.view.Id = ++totalIds; SetView(t, d, "NEW"); return t;
    }

    private void Observe(Entry t, Detection d, bool recovered)
    {
        var m = Measurement(d.box);
        for (int i = 0; i < 4; i++)
        {
            double noise = Math.Pow(t.mean[2 + i % 2] / 20, 2) * Math.Max(1 - d.confidence, .05);
            double s = t.pp[i] + noise, k = t.pp[i] / s, kv = t.pv[i] / s, error = m[i] - t.mean[i];
            t.mean[i] += k * error; t.mean[i + 4] += kv * error;
            double oldPP = t.pp[i], oldPV = t.pv[i];
            t.pp[i] -= k * oldPP; t.pv[i] -= k * oldPV; t.vv[i] -= kv * oldPV;
        }
        t.Remember(frame, recovered ? t.Predicted : d.box);
        if (recovered) { t.updates = 0; t.state = State.Tracked; }
        else
        {
            t.updates++;
            for (int c = 0; c < 4; c++)
            {
                Vector2 v = Vector2.zero;
                for (int dt = 1; dt <= 3; dt++)
                {
                    Vector2 delta = Corner(d.box, c) - Corner(t.History(frame - dt), c);
                    v += delta / (delta.magnitude + .00001f) / dt;
                }
                t.velocity[c] = v / 3;
            }
            if (t.state == State.Tracked || t.updates + 1 >= 3) t.state = State.Tracked;
        }
        t.last = frame; t.previousScore = t.view.Confidence;
        SetView(t, d, recovered ? "RECOVERED" : t.state == State.New ? "NEW" : "TRACKED");
    }

    private void SetView(Entry t, Detection d, string state)
    {
        t.view.Box = d.box; t.view.Kind = d.kind; t.view.Confidence = d.confidence; t.view.State = state; t.view.Hits++;
        t.view.trail.Add(new Vector2(d.box.center.x, d.box.yMax));
        if (t.view.trail.Count > TrailLength) t.view.trail.RemoveAt(0);
    }

    private List<(int track, int det)> Associate(List<Entry> tracks, List<Detection> dets, int highCount)
    {
        var cost = new float[tracks.Count, dets.Count];
        for (int i = 0; i < tracks.Count; i++) for (int j = 0; j < dets.Count; j++)
        {
            var t = tracks[i]; var d = dets[j]; var b = t.Predicted;
            float iou = IoU(b, d.box), hUnion = Math.Max(b.yMax, d.box.yMax) - Math.Min(b.yMin, d.box.yMin);
            float hIoU = Math.Max(0, Math.Min(b.yMax, d.box.yMax) - Math.Max(b.yMin, d.box.yMin)) / (hUnion + 1e-9f);
            float value = 1 - hIoU * iou + .1f * Math.Abs(2 * t.view.Confidence - t.previousScore - d.confidence);
            if (iou > .10f)
            {
                float angle = 0; var history = t.History(frame - 3);
                for (int c = 0; c < 4; c++)
                {
                    Vector2 delta = Corner(d.box, c) - Corner(history, c);
                    float dot = Vector2.Dot(t.velocity[c], delta / (delta.magnitude + .00001f));
                    angle += (float)(Math.Acos(Math.Max(-1, Math.Min(1, dot))) / Math.PI) / 4;
                }
                value += .05f * angle * d.confidence;
            }
            else value = 1;
            cost[i, j] = Math.Min(1, Math.Max(0, value) + (j >= highCount ? .2f : 0));
        }
        var result = new List<(int, int)>(); float threshold = .70f;
        while (tracks.Count > 0 && dets.Count > 0)
        {
            var nearestDet = new int[tracks.Count]; var nearestTrack = new int[dets.Count];
            for (int i = 0; i < tracks.Count; i++) for (int j = 1; j < dets.Count; j++) if (cost[i, j] < cost[i, nearestDet[i]]) nearestDet[i] = j;
            for (int j = 0; j < dets.Count; j++) for (int i = 1; i < tracks.Count; i++) if (cost[i, j] < cost[nearestTrack[j], j]) nearestTrack[j] = i;
            var pairs = new List<(int track, int det)>();
            for (int i = 0; i < tracks.Count; i++) if (nearestTrack[nearestDet[i]] == i && cost[i, nearestDet[i]] < threshold) pairs.Add((i, nearestDet[i]));
            if (pairs.Count == 0) break;
            result.AddRange(pairs);
            foreach (var p in pairs)
            {
                for (int j = 0; j < dets.Count; j++) cost[p.track, j] = float.PositiveInfinity;
                for (int i = 0; i < tracks.Count; i++) cost[i, p.det] = float.PositiveInfinity;
            }
            threshold -= .05f;
        }
        return result;
    }

    private static bool Finite(float v) => !float.IsNaN(v) && !float.IsInfinity(v);
    private static double[] Measurement(Rect b) => new[] { (double)b.center.x, b.center.y, b.width, b.height };
    private static Vector2 Corner(Rect b, int c) => new Vector2(c < 2 ? b.xMin : b.xMax, c % 2 == 0 ? b.yMin : b.yMax);
    private static float IoU(Rect a, Rect b)
    {
        float area = Math.Max(0, Math.Min(a.xMax, b.xMax) - Math.Max(a.xMin, b.xMin)) * Math.Max(0, Math.Min(a.yMax, b.yMax) - Math.Max(a.yMin, b.yMin));
        return area / (a.width * a.height + b.width * b.height - area + 1e-7f);
    }
}
