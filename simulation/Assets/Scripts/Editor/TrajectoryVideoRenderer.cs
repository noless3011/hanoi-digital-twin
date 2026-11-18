using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Debug = UnityEngine.Debug;

/// <summary>Renders a vehicle-trajectories file through the scene's feed camera into an MP4, frame by frame at a
/// fixed rate (not real time), so the video can be run through the same RF-DETR + tracker as real footage.
/// The camera is the Lane Editor's export camera at its export image size, the same view the layout and the
/// image-to-ground calibration were made from. Overlays (OnGUI) and the lane gizmo lines are not in the video.
/// Frames are piped raw to ffmpeg.
///
/// Batch mode (Unity must not have the project open):
///   Unity.exe -batchmode -projectPath simulation -executeMethod TrajectoryVideoRenderer.RenderFromCommandLine
///     -scene Assets/Scenes/buoi-hht.unity -trajectories PATH.json -out PATH.mp4 [-fps 25] [-start 0] [-duration S]
///     [-crf 23] [-ffmpeg ffmpeg] -logFile -
/// </summary>
public static class TrajectoryVideoRenderer
{
    public sealed class Options
    {
        public string trajectories, output, ffmpeg = "ffmpeg";
        public float fps = 25, start, duration = -1;
        public int crf = 23;
        public bool hideLaneLines = true;
    }

    [MenuItem("Tools/Traffic/Render trajectories to video...")]
    private static void RenderFromMenu()
    {
        string input = EditorUtility.OpenFilePanel("Trajectories", Path.Combine(Application.streamingAssetsPath, "Traffic"), "json");
        if (string.IsNullOrEmpty(input)) return;
        string output = EditorUtility.SaveFilePanel("Video", Path.GetDirectoryName(input), Path.GetFileNameWithoutExtension(input) + ".render", "mp4");
        if (string.IsNullOrEmpty(output)) return;
        try { Render(new Options { trajectories = input, output = output }, true); }
        catch (Exception e) { Debug.LogException(e); EditorUtility.DisplayDialog("Render failed", e.Message, "OK"); }
    }

    public static void RenderFromCommandLine()
    {
        int code = 0;
        try
        {
            string[] args = Environment.GetCommandLineArgs();
            string Arg(string name, string fallback = null)
            {
                int i = Array.IndexOf(args, "-" + name);
                return i >= 0 && i + 1 < args.Length ? args[i + 1] : fallback;
            }
            float Num(string name, float fallback) => float.Parse(Arg(name, fallback.ToString(CultureInfo.InvariantCulture)), CultureInfo.InvariantCulture);
            string scene = Arg("scene") ?? throw new ArgumentException("-scene is required");
            EditorSceneManager.OpenScene(scene, OpenSceneMode.Single);
            Render(new Options {
                trajectories = Path.GetFullPath(Arg("trajectories") ?? throw new ArgumentException("-trajectories is required")),
                output = Path.GetFullPath(Arg("out") ?? throw new ArgumentException("-out is required")),
                fps = Num("fps", 25), start = Num("start", 0), duration = Num("duration", -1),
                crf = (int)Num("crf", 23), ffmpeg = Arg("ffmpeg", "ffmpeg"),
            }, false);
        }
        catch (Exception e) { Debug.LogException(e); code = 1; }
        EditorApplication.Exit(code);
    }

    public static void Render(Options o, bool progressBar)
    {
        var laneEditor = UnityEngine.Object.FindObjectsByType<LaneEditor>(FindObjectsSortMode.None).FirstOrDefault();
        Camera camera = laneEditor != null ? CameraLaneExport.ResolveCamera(laneEditor) : Camera.main;
        if (camera == null) throw new InvalidOperationException("No feed camera in the open scene.");
        Vector2Int size = laneEditor != null ? laneEditor.exportImageSize : new Vector2Int(384, 216);
        if (o.fps <= 0) throw new ArgumentException("fps must be positive");

        var hidden = o.hideLaneLines ? UnityEngine.Object.FindObjectsByType<LineRenderer>(FindObjectsSortMode.None).Where(r => r.enabled).ToArray() : new LineRenderer[0];
        foreach (var r in hidden) r.enabled = false;
        var host = new GameObject("Trajectory video render") { hideFlags = HideFlags.DontSave };
        var rt = new RenderTexture(size.x, size.y, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB) { antiAliasing = 4 };
        var frame = new Texture2D(size.x, size.y, TextureFormat.RGB24, false);
        var previousTarget = camera.targetTexture;
        Process ffmpeg = null;
        try
        {
            var player = host.AddComponent<TrajectoryPlayer>();
            player.playOnStart = false;
            player.trajectoryPath = o.trajectories;
            player.Load();
            double end = o.duration > 0 ? Math.Min(player.Duration, o.start + o.duration) : player.Duration;
            int frames = Math.Max(1, (int)Math.Floor((end - o.start) * o.fps) + 1);
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(o.output)));

            string fps = o.fps.ToString(CultureInfo.InvariantCulture);
            ffmpeg = Process.Start(new ProcessStartInfo {
                FileName = o.ffmpeg,
                // ReadPixels rows are bottom-up, hence vflip.
                Arguments = $"-y -loglevel error -f rawvideo -pix_fmt rgb24 -s {size.x}x{size.y} -r {fps} -i - " +
                            $"-vf vflip -c:v libx264 -preset medium -crf {o.crf} -pix_fmt yuv420p -r {fps} \"{o.output}\"",
                UseShellExecute = false, RedirectStandardInput = true, CreateNoWindow = true,
            });
            if (ffmpeg == null) throw new InvalidOperationException("Could not start ffmpeg (" + o.ffmpeg + ")");
            var stdin = ffmpeg.StandardInput.BaseStream;

            camera.targetTexture = rt;
            for (int i = 0; i < frames; i++)
            {
                player.Seek(o.start + i / (double)o.fps);
                camera.Render();
                var active = RenderTexture.active;
                RenderTexture.active = rt;
                frame.ReadPixels(new Rect(0, 0, size.x, size.y), 0, 0, false);
                RenderTexture.active = active;
                var bytes = frame.GetRawTextureData<byte>();
                stdin.Write(bytes.ToArray(), 0, bytes.Length);
                if (progressBar && i % 25 == 0 &&
                    EditorUtility.DisplayCancelableProgressBar("Rendering trajectories", $"Frame {i}/{frames}", i / (float)frames))
                    throw new OperationCanceledException("Cancelled");
                if (!progressBar && i % 250 == 0) Debug.Log($"Rendered {i}/{frames} frames");
            }
            stdin.Close();
            ffmpeg.WaitForExit();
            if (ffmpeg.ExitCode != 0) throw new InvalidOperationException("ffmpeg exited with " + ffmpeg.ExitCode);
            Debug.Log($"Rendered {frames} frames ({size.x}x{size.y} @ {fps} fps, {player.VehicleCount} vehicles) to {o.output}");
        }
        finally
        {
            if (ffmpeg != null && !ffmpeg.HasExited) { try { ffmpeg.Kill(); } catch { } }
            camera.targetTexture = previousTarget;
            foreach (var r in hidden) r.enabled = true;
            UnityEngine.Object.DestroyImmediate(host);
            UnityEngine.Object.DestroyImmediate(frame);
            rt.Release();
            UnityEngine.Object.DestroyImmediate(rt);
            if (progressBar) EditorUtility.ClearProgressBar();
        }
    }
}
