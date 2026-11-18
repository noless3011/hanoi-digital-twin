using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

/// <summary>Scene view authoring for LaneEditor: select and reshape lanes, draw new ones, and place traffic
/// light stop lines. The Inspector shows the same selection with exact numbers.</summary>
[CustomEditor(typeof(LaneEditor))]
public class LaneEditorInspector : Editor
{
    private enum Mode { Select, Draw, Lights }
    // What a click would act on: an existing node, a point on another lane, or the road surface.
    private struct Target
    {
        public bool valid, onSurface;
        public string nodeId;
        public int lane, segment;
        public float t;
        public Vector3 point;
    }

    private static readonly string[] ModeNames = { "Select", "Draw lane", "Traffic lights" };
    private static readonly float[] Sides = { -1f, 1f };
    private static readonly Color CenterColor = new Color(0.25f, 0.8f, 1), SelectedColor = Color.yellow, JoinColor = new Color(0.3f, 1, 0.4f);
    private const float PanelWidth = 340, SegmentPickRadius = 10;

    // Editing state is shared by every Lane Editor and lasts until scripts reload.
    private static Mode mode;
    private static bool snapToSurface = true, showLabels = true, showLaneList = true, showSurface, showDisplay, showExport;
    private static float newLaneWidth = 3.2f;

    private bool drawing;            // Draw mode: clicks add points to the selected lane.
    private bool joining, joinStart; // Select mode: the next click picks where one end of the selected lane joins.
    private Action pending;          // A button's action, run after the GUI pass so its layout stays consistent.
    private Rect panelRect;
    private float panelHeight = 150;
    private Vector2 laneScroll;

    private void OnEnable() { Undo.undoRedoPerformed += OnUndo; }
    private void OnDisable()
    {
        Undo.undoRedoPerformed -= OnUndo;
        Tools.hidden = false;
        if (target != null) EndGesture((LaneEditor)target);
    }
    private void OnUndo()
    {
        if (target == null) return;
        LaneEditor editor = (LaneEditor)target;
        joining = false; editor.CancelTrafficLightDrawing();
        if (editor.CurrentLane == null) { editor.SelectedLane = editor.SelectedSection = -1; drawing = false; }
        editor.RebuildVisuals();
        Repaint(); SceneView.RepaintAll();
    }

    // ---------------------------------------------------------------- modes and gestures

    private void SetMode(LaneEditor editor, Mode next)
    {
        EndGesture(editor);
        mode = next;
        if (next == Mode.Draw) { Select(editor, -1, -1); PrepareColliders(editor); }
        if (next == Mode.Lights) PrepareColliders(editor);
        Repaint(); SceneView.RepaintAll();
    }
    private void Select(LaneEditor editor, int lane, int section)
    {
        editor.SelectedLane = lane; editor.SelectedSection = section;
        Repaint(); SceneView.RepaintAll();
    }
    // Stop whatever is half done: an unfinished lane, a join pick, or a stop line with one end placed.
    private void EndGesture(LaneEditor editor)
    {
        editor.CancelTrafficLightDrawing();
        joining = false;
        FinishDrawing(editor);
    }
    private void FinishDrawing(LaneEditor editor)
    {
        if (!drawing) return;
        drawing = false;
        LaneEditor.LaneStrip lane = editor.CurrentLane;
        if (lane == null) return;
        if (lane.sections.Count < 2) Change(editor, "Discard unfinished lane", editor.DeleteSelectedLane);
        else { newLaneWidth = lane.width; editor.SelectedSection = -1; }
    }
    private void BeginExtend(LaneEditor editor)
    {
        EndGesture(editor);
        if (editor.CurrentLane == null) return;
        mode = Mode.Draw; drawing = true;
        editor.SelectedSection = editor.CurrentLane.sections.Count - 1;
        PrepareColliders(editor);
        Repaint(); SceneView.RepaintAll();
    }
    private void BeginBranch(LaneEditor editor, string nodeId)
    {
        EndGesture(editor);
        LaneEditor.LaneStrip started = null; float width = newLaneWidth;
        Change(editor, "Branch lane", () => started = editor.StartLaneAt(nodeId, width));
        if (started == null) return;
        mode = Mode.Draw; drawing = true;
        PrepareColliders(editor);
        Repaint();
    }
    private void BeginJoin(LaneEditor editor, bool start)
    {
        bool begin = !joining;
        EndGesture(editor);
        mode = Mode.Select; joining = begin; joinStart = start;
        Repaint(); SceneView.RepaintAll();
    }
    private void ToggleLightDrawing(LaneEditor editor)
    {
        bool begin = !editor.IsDrawingTrafficLight;
        EndGesture(editor);
        if (begin) { PrepareColliders(editor); editor.BeginTrafficLightDrawing(); }
        Repaint(); SceneView.RepaintAll();
    }
    private void DeleteSelection(LaneEditor editor)
    {
        if (mode == Mode.Lights) Change(editor, "Delete traffic light", editor.DeleteSelectedTrafficLight);
        else if (drawing)
        {
            Change(editor, "Remove lane point", editor.RemoveLastSection);
            if (editor.CurrentLane != null && editor.CurrentLane.sections.Count == 0) FinishDrawing(editor);
        }
        else if (editor.SelectedNode != null) Change(editor, "Delete lane node", editor.DeleteSelectedNode);
        else Change(editor, "Delete lane", editor.DeleteSelectedLane);
        Repaint();
    }
    private void RunPending()
    {
        if (pending == null) return;
        Action action = pending; pending = null;
        action();
    }
    private string Hint(LaneEditor editor)
    {
        if (mode == Mode.Lights)
            return editor.IsDrawingTrafficLight
                ? (editor.HasTrafficLightStart ? "Click the end of the stop line." : "Click the start of the stop line.") + " Esc cancels."
                : "Click a stop line to select it, then drag its ends. Delete removes it.";
        if (mode == Mode.Draw)
            return drawing
                ? "Click to add points. Click a node or another lane to join it and finish. Enter finishes, Backspace removes the last point."
                : "Click the road to start a lane. Start on a node or a lane to branch from it. Esc returns to Select.";
        if (joining) return "Click the node or lane that the " + (joinStart ? "start" : "end") + " of this lane joins. Esc cancels.";
        if (editor.CurrentLane == null) return "Click a lane or a node to select it. F frames the selection.";
        return "Drag a node to move it, a green cube to change the width (Shift: whole lane), a white dot to add a node. "
            + "W shows move axes, E rotates the cross-section. Delete removes the node or lane.";
    }

    // ---------------------------------------------------------------- Inspector

    public override void OnInspectorGUI()
    {
        LaneEditor editor = (LaneEditor)target;
        Mode next = (Mode)GUILayout.Toolbar((int)mode, ModeNames, GUILayout.Height(24));
        if (next != mode) pending = () => SetMode(editor, next);
        EditorGUILayout.HelpBox(Hint(editor), MessageType.None);
        if (mode == Mode.Lights) DrawLightsInspector(editor);
        else DrawLanesInspector(editor);
        EditorGUILayout.Space();
        DrawSettings(editor);
        EditorGUILayout.Space();
        if (GUILayout.Button(new GUIContent("Export camera lane layout", "Writes " + CameraLaneExport.DefaultPath(editor)), GUILayout.Height(26)))
            pending = () => ExportCameraLanes(editor);
        if (!string.IsNullOrEmpty(editor.Status)) EditorGUILayout.LabelField(editor.Status, EditorStyles.wordWrappedMiniLabel);
        RunPending();
    }

    private void DrawLanesInspector(LaneEditor editor)
    {
        int unfinished = editor.laneStrips.FindAll(l => l.sections.Count < 2).Count;
        EditorGUILayout.BeginHorizontal();
        showLaneList = EditorGUILayout.Foldout(showLaneList, "Lanes (" + editor.laneStrips.Count + ")", true);
        if (GUILayout.Button(new GUIContent("New lane", "Draw a new lane in the Scene view"), EditorStyles.miniButton, GUILayout.Width(80)))
            pending = () => SetMode(editor, Mode.Draw);
        EditorGUILayout.EndHorizontal();
        if (showLaneList)
        {
            bool scroll = editor.laneStrips.Count > 12;
            if (scroll) laneScroll = EditorGUILayout.BeginScrollView(laneScroll, GUILayout.Height(240));
            for (int i = 0; i < editor.laneStrips.Count; i++)
            {
                LaneEditor.LaneStrip item = editor.laneStrips[i];
                string label = item.name + (item.sections.Count < 2 ? "   (unfinished)" : "   " + item.width.ToString("0.##") + " m");
                if (!GUILayout.Toggle(editor.SelectedLane == i, label, EditorStyles.miniButton) || editor.SelectedLane == i) continue;
                int index = i;
                pending = () => { EndGesture(editor); Select(editor, index, -1); };
            }
            if (scroll) EditorGUILayout.EndScrollView();
        }
        int spawns = editor.nodes.FindAll(n => n.isSpawnPoint).Count, deadEnds = editor.nodes.FindAll(n => n.isDeadEnd).Count;
        EditorGUILayout.LabelField(editor.nodes.Count + " nodes, " + spawns + " spawn points, " + deadEnds + " dead ends", EditorStyles.miniLabel);
        if (unfinished > 0 && !drawing)
            EditorGUILayout.HelpBox(unfinished + " lane(s) have fewer than two nodes and are left out of the export. Extend or delete them.", MessageType.Warning);

        LaneEditor.LaneStrip lane = editor.CurrentLane;
        if (lane == null) return;
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Selected lane", EditorStyles.boldLabel);
        EditorGUI.BeginChangeCheck();
        string name = EditorGUILayout.TextField("Name", lane.name);
        float width = EditorGUILayout.DelayedFloatField("Width (m)", lane.width);
        if (EditorGUI.EndChangeCheck()) Change(editor, "Edit lane", () => { lane.name = name; lane.width = width; lane.Normalize(); });
        EditorGUILayout.LabelField("Length", editor.LaneLength(lane).ToString("F1") + " m, " + lane.sections.Count + " nodes, travel from node 1 to node " + lane.sections.Count);
        DrawLaneButtons(editor, lane);
        if (!drawing) DrawNodeInspector(editor, lane);
    }

    // Shared by the Inspector and the Scene view panel.
    private void DrawLaneButtons(LaneEditor editor, LaneEditor.LaneStrip lane)
    {
        GUILayout.BeginHorizontal();
        if (drawing)
        {
            if (GUILayout.Button("Finish lane (Enter)")) pending = () => { FinishDrawing(editor); Repaint(); SceneView.RepaintAll(); };
            using (new EditorGUI.DisabledScope(lane.sections.Count == 0))
                if (GUILayout.Button("Remove last point")) pending = () => DeleteSelection(editor);
        }
        else
        {
            bool start = editor.SelectedSection == 0;
            if (GUILayout.Button(new GUIContent("Extend", "Keep drawing from the end of this lane"))) pending = () => BeginExtend(editor);
            using (new EditorGUI.DisabledScope(lane.sections.Count < 2))
            {
                if (GUILayout.Button(new GUIContent("Reverse", "Swap the travel direction"))) pending = () => Change(editor, "Reverse lane", editor.ReverseSelectedLane);
                GUIContent join = new GUIContent(start ? "Join start..." : "Join end...", "Then click the node or lane this end connects to. Select the first node to join the start.");
                if (GUILayout.Toggle(joining, join, "Button") != joining) pending = () => BeginJoin(editor, start);
            }
            if (GUILayout.Button("Delete lane")) pending = () => { EndGesture(editor); Change(editor, "Delete lane", editor.DeleteSelectedLane); };
        }
        GUILayout.EndHorizontal();
    }

    private void DrawNodeInspector(LaneEditor editor, LaneEditor.LaneStrip lane)
    {
        string[] names = new string[lane.sections.Count + 1];
        names[0] = "None";
        for (int i = 0; i < lane.sections.Count; i++) names[i + 1] = "Node " + (i + 1) + (i == 0 ? " (start)" : i == lane.sections.Count - 1 ? " (end)" : "");
        EditorGUILayout.Space();
        int selected = EditorGUILayout.Popup("Selected node", Mathf.Clamp(editor.SelectedSection + 1, 0, names.Length - 1), names) - 1;
        if (selected != editor.SelectedSection) pending = () => Select(editor, editor.SelectedLane, selected);
        if (editor.SelectedSection < 0 || editor.SelectedSection >= lane.sections.Count) return;
        LaneEditor.Section section = lane.sections[editor.SelectedSection];
        LaneEditor.LaneNode node = editor.FindNode(section.nodeId);
        if (node == null) return;

        EditorGUI.BeginChangeCheck();
        Vector3 position = EditorGUILayout.Vector3Field("Position (world)", editor.transform.TransformPoint(node.position));
        if (EditorGUI.EndChangeCheck()) Change(editor, "Move lane node", () => editor.MoveNode(node.id, position));
        EditorGUI.BeginChangeCheck();
        float widthHere = EditorGUILayout.DelayedFloatField(new GUIContent("Width here (m)", "The lane tapers between nodes of different width."), lane.width * section.widthScale);
        Vector3 rotation = EditorGUILayout.Vector3Field(new GUIContent("Cross-section rotation", "Pitch, yaw and roll offset from the lane direction. Yaw skews the lane end."), section.rotation);
        if (EditorGUI.EndChangeCheck()) Change(editor, "Edit lane cross-section", () =>
        {
            section.rotation = rotation;
            section.widthScale = Mathf.Clamp(widthHere / Mathf.Max(0.1f, lane.width), 0.1f, 10);
        });
        EditorGUI.BeginChangeCheck();
        bool spawn = EditorGUILayout.Toggle(new GUIContent("Spawn point", "Vehicles enter the network here."), node.isSpawnPoint);
        string spawnName = spawn ? EditorGUILayout.TextField("Spawn point name", node.spawnPointName) : node.spawnPointName;
        bool deadEnd = EditorGUILayout.Toggle(new GUIContent("Dead end", "The last visible point of the lane, where vehicles leave the camera view."), node.isDeadEnd);
        string endName = deadEnd ? EditorGUILayout.TextField("Dead end name", node.deadEndName) : node.deadEndName;
        if (EditorGUI.EndChangeCheck()) Change(editor, "Edit node marks", () =>
        {
            node.isSpawnPoint = spawn; node.spawnPointName = spawnName;
            node.isDeadEnd = deadEnd; node.deadEndName = endName;
        });

        GUILayout.BeginHorizontal();
        int segment = editor.SelectedSection;
        using (new EditorGUI.DisabledScope(segment >= lane.sections.Count - 1))
            if (GUILayout.Button(new GUIContent("Insert node after", "Adds a node halfway to the next one"))) pending = () => Change(editor, "Insert lane node", () => editor.InsertNode(segment));
        if (GUILayout.Button(new GUIContent("Branch", "Draw a new lane from this node"))) pending = () => BeginBranch(editor, node.id);
        if (GUILayout.Button("Delete node")) pending = () => Change(editor, "Delete lane node", editor.DeleteSelectedNode);
        GUILayout.EndHorizontal();
    }

    private void DrawLightsInspector(LaneEditor editor)
    {
        if (GUILayout.Button(editor.IsDrawingTrafficLight ? "Cancel drawing (Esc)" : "Draw stop line")) pending = () => ToggleLightDrawing(editor);
        for (int i = 0; i < editor.trafficLights.Count; i++)
        {
            if (!GUILayout.Toggle(editor.SelectedTrafficLight == i, editor.trafficLights[i].name, EditorStyles.miniButton) || editor.SelectedTrafficLight == i) continue;
            int index = i;
            pending = () => { EndGesture(editor); editor.SelectedTrafficLight = index; SceneView.RepaintAll(); };
        }
        LaneEditor.TrafficLightLine light = editor.CurrentTrafficLight;
        if (light == null) return;
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Selected traffic light", EditorStyles.boldLabel);
        EditorGUI.BeginChangeCheck();
        string name = EditorGUILayout.TextField("Name", light.name);
        if (EditorGUI.EndChangeCheck()) Change(editor, "Rename traffic light", () => light.name = name);
        EditorGUI.BeginChangeCheck();
        float red = EditorGUILayout.FloatField("Red duration (s)", light.redDurationS);
        float green = EditorGUILayout.FloatField("Green duration (s)", light.greenDurationS);
        float offset = EditorGUILayout.FloatField(new GUIContent("Cycle offset (s)", "Red then green, repeating. A positive offset advances the cycle."), light.cycleOffsetS);
        if (EditorGUI.EndChangeCheck() && LaneEditor.TrafficLightLine.ValidTiming(red, green, offset))
            Change(editor, "Change traffic light timing", () => { light.redDurationS = red; light.greenDurationS = green; light.cycleOffsetS = offset; });
        EditorGUILayout.LabelField("Signal now", light.AspectAt(Application.isPlaying ? editor.TrafficLightDisplayTime : 0));
        int selected = editor.SelectedTrafficLight;
        EditorGUI.BeginChangeCheck();
        Vector3 start = EditorGUILayout.Vector3Field("Start (world)", editor.transform.TransformPoint(light.start));
        if (EditorGUI.EndChangeCheck()) Change(editor, "Move traffic light start", () => editor.MoveTrafficLightEndpoint(selected, true, start));
        EditorGUI.BeginChangeCheck();
        Vector3 end = EditorGUILayout.Vector3Field("End (world)", editor.transform.TransformPoint(light.end));
        if (EditorGUI.EndChangeCheck()) Change(editor, "Move traffic light end", () => editor.MoveTrafficLightEndpoint(selected, false, end));
        EditorGUILayout.LabelField("ID", light.id, EditorStyles.miniLabel);
        if (GUILayout.Button("Delete traffic light")) pending = () => Change(editor, "Delete traffic light", editor.DeleteSelectedTrafficLight);
    }

    private void DrawSettings(LaneEditor editor)
    {
        serializedObject.Update();
        EditorGUI.BeginChangeCheck();
        showSurface = EditorGUILayout.Foldout(showSurface, "Road surface", true);
        if (showSurface)
        {
            EditorGUI.indentLevel++;
            Property("surfaceRoot"); Property("surfaceLayers"); Property("prepareMeshColliders");
            snapToSurface = EditorGUILayout.Toggle(new GUIContent("Snap To Surface", "Dragged nodes and stop line ends follow the road model. Off: they move freely, including in height."), snapToSurface);
            EditorGUI.indentLevel--;
            if (GUILayout.Button("Add mesh colliders to the road model now")) pending = () => PrepareColliders(editor, true);
        }
        showDisplay = EditorGUILayout.Foldout(showDisplay, "Display", true);
        if (showDisplay)
        {
            EditorGUI.indentLevel++;
            Property("showFill"); Property("stripColor"); Property("stripMaterial"); Property("laneMaterial");
            Property("drawAboveSurface"); Property("showLanesInGame"); Property("lineWidth"); Property("arrowSize");
            Property("nodeMarkerSize"); Property("nodePickRadius"); Property("surfaceOffset");
            showLabels = EditorGUILayout.Toggle(new GUIContent("Show Labels", "Lane names and node marks in the Scene view"), showLabels);
            EditorGUI.indentLevel--;
        }
        showExport = EditorGUILayout.Foldout(showExport, "Export and backup", true);
        if (showExport)
        {
            EditorGUI.indentLevel++;
            Property("exportCamera"); Property("exportImageSize"); Property("exportFileName");
            EditorGUI.indentLevel--;
            GUILayout.BeginHorizontal();
            if (GUILayout.Button(new GUIContent("Save lanes to JSON...", "A backup of the lanes, node marks and traffic lights"))) pending = () => SaveJson(editor);
            if (GUILayout.Button(new GUIContent("Load lanes from JSON...", "Replaces the lanes in this scene"))) pending = () => LoadJson(editor);
            GUILayout.EndHorizontal();
        }
        bool changed = EditorGUI.EndChangeCheck();
        serializedObject.ApplyModifiedProperties();
        if (changed) { editor.RebuildVisuals(); SceneView.RepaintAll(); }
    }
    private void Property(string name) => EditorGUILayout.PropertyField(serializedObject.FindProperty(name));

    // ---------------------------------------------------------------- Scene view

    private void OnSceneGUI()
    {
        LaneEditor editor = (LaneEditor)target;
        SceneView scene = SceneView.currentDrawingSceneView;
        if (scene == null || scene.camera == null) return;
        Event e = Event.current;
        Tools.hidden = true; // The object's own move handle would shift the whole network.
        DrawPanel(editor, e);
        RunPending();
        // Clicks on the road edit lanes instead of selecting other scene objects.
        int passive = GUIUtility.GetControlID(FocusType.Passive);
        if (e.type == EventType.Layout) HandleUtility.AddDefaultControl(passive);
        if (e.type == EventType.MouseMove) scene.Repaint();
        HandleKeys(editor, e);
        bool click = e.type == EventType.MouseDown && e.button == 0 && !e.alt && !Tools.viewToolActive;
        DrawLanes(editor);
        if (mode == Mode.Lights) LightsSceneGUI(editor, scene, e, click, passive);
        else if (mode == Mode.Draw) DrawSceneGUI(editor, scene, e, click);
        else SelectSceneGUI(editor, scene, e, click, passive);
    }

    private void DrawPanel(LaneEditor editor, Event e)
    {
        panelRect = new Rect(10, 10, PanelWidth, panelHeight);
        Handles.BeginGUI();
        GUILayout.BeginArea(panelRect, GUI.skin.box);
        GUILayout.BeginVertical();
        Mode next = (Mode)GUILayout.Toolbar((int)mode, ModeNames, GUILayout.Height(24));
        if (next != mode) pending = () => SetMode(editor, next);
        GUILayout.Label(Hint(editor), EditorStyles.wordWrappedLabel);
        if (mode == Mode.Lights)
        {
            if (GUILayout.Button(editor.IsDrawingTrafficLight ? "Cancel drawing (Esc)" : "Draw stop line")) pending = () => ToggleLightDrawing(editor);
            if (editor.CurrentTrafficLight != null) GUILayout.Label(editor.CurrentTrafficLight.name, EditorStyles.boldLabel);
        }
        else DrawLanePanel(editor);
        GUILayout.BeginHorizontal();
        bool fill = GUILayout.Toggle(editor.showFill, "Fill");
        if (fill != editor.showFill) pending = () => Change(editor, "Toggle lane fill", () => editor.showFill = fill);
        showLabels = GUILayout.Toggle(showLabels, "Labels");
        snapToSurface = GUILayout.Toggle(snapToSurface, new GUIContent("Snap to surface", "Dragged nodes follow the road model"));
        GUILayout.EndHorizontal();
        GUILayout.EndVertical();
        if (e.type == EventType.Repaint)
        {
            // The panel is as tall as its content, measured on the previous repaint.
            float height = GUILayoutUtility.GetLastRect().yMax + 10;
            if (Mathf.Abs(height - panelHeight) > 1) { panelHeight = height; SceneView.RepaintAll(); }
        }
        GUILayout.EndArea();
        Handles.EndGUI();
        if (e.type == EventType.MouseDown && panelRect.Contains(e.mousePosition)) e.Use();
    }

    private void DrawLanePanel(LaneEditor editor)
    {
        LaneEditor.LaneStrip lane = editor.CurrentLane;
        if (lane == null) return;
        GUILayout.Label(lane.name + ":  " + lane.width.ToString("0.##") + " m wide, " + editor.LaneLength(lane).ToString("F0") + " m long", EditorStyles.boldLabel);
        DrawLaneButtons(editor, lane);
        LaneEditor.LaneNode node = drawing ? null : editor.SelectedNode;
        if (node == null) return;
        GUILayout.BeginHorizontal();
        GUILayout.Label("Node " + (editor.SelectedSection + 1) + "/" + lane.sections.Count, GUILayout.Width(62));
        bool spawn = GUILayout.Toggle(node.isSpawnPoint, new GUIContent("Spawn", "Vehicles enter here"), "Button");
        if (spawn != node.isSpawnPoint) pending = () => Change(editor, "Mark spawn point", () => node.isSpawnPoint = spawn);
        bool deadEnd = GUILayout.Toggle(node.isDeadEnd, new GUIContent("Dead end", "Vehicles leave the camera view here"), "Button");
        if (deadEnd != node.isDeadEnd) pending = () => Change(editor, "Mark dead end", () => node.isDeadEnd = deadEnd);
        if (GUILayout.Button(new GUIContent("Branch", "Draw a new lane from this node"))) pending = () => BeginBranch(editor, node.id);
        if (GUILayout.Button("Delete node")) pending = () => Change(editor, "Delete lane node", editor.DeleteSelectedNode);
        GUILayout.EndHorizontal();
    }

    private void HandleKeys(LaneEditor editor, Event e)
    {
        bool hasSelection = mode == Mode.Lights ? editor.CurrentTrafficLight != null : editor.CurrentLane != null;
        // Delete would otherwise remove the Lane Editor object itself.
        if ((e.type == EventType.ValidateCommand || e.type == EventType.ExecuteCommand) && (e.commandName == "SoftDelete" || e.commandName == "Delete"))
        {
            if (!hasSelection) return;
            if (e.type == EventType.ExecuteCommand) DeleteSelection(editor);
            e.Use(); return;
        }
        if (e.type != EventType.KeyDown) return;
        switch (e.keyCode)
        {
            case KeyCode.Escape:
                if (joining || editor.IsDrawingTrafficLight || drawing) EndGesture(editor);
                else if (mode != Mode.Select) SetMode(editor, Mode.Select);
                else if (editor.SelectedSection >= 0) editor.SelectedSection = -1;
                else if (editor.SelectedLane >= 0) editor.SelectedLane = -1;
                else return;
                break;
            case KeyCode.Return:
            case KeyCode.KeypadEnter:
                if (!drawing) return;
                FinishDrawing(editor);
                break;
            case KeyCode.Backspace:
            case KeyCode.Delete:
                if (!hasSelection) return;
                DeleteSelection(editor);
                break;
            default: return;
        }
        e.Use(); Repaint(); SceneView.RepaintAll();
    }

    // Centre lines of every lane, and the outline of the selected one.
    private void DrawLanes(LaneEditor editor)
    {
        if (Event.current.type != EventType.Repaint) return;
        for (int i = 0; i < editor.laneStrips.Count; i++)
        {
            LaneEditor.LaneStrip lane = editor.laneStrips[i];
            RoadMeshGeometry.Frame[] frames = RoadMeshGeometry.Frames(editor, lane);
            if (frames.Length < 2) continue;
            bool selected = i == editor.SelectedLane && mode != Mode.Lights;
            Vector3[] centers = new Vector3[frames.Length], left = new Vector3[frames.Length], right = new Vector3[frames.Length];
            for (int k = 0; k < frames.Length; k++) { centers[k] = frames[k].center; left[k] = frames[k].left; right[k] = frames[k].right; }
            Handles.color = selected ? SelectedColor : CenterColor;
            Handles.DrawAAPolyLine(selected ? 4 : 2, centers);
            if (selected)
            {
                Handles.DrawAAPolyLine(3, left); Handles.DrawAAPolyLine(3, right);
                for (int k = 0; k < frames.Length; k++) Handles.DrawLine(left[k], right[k]);
            }
            // Clear of the arrow and the insert dot at the middle of the segment.
            if (showLabels) Handles.Label(Vector3.Lerp(centers[0], centers[1], 0.3f), lane.name, selected ? EditorStyles.whiteBoldLabel : EditorStyles.whiteLabel);
        }
    }

    private void DrawNodes(LaneEditor editor, SceneView scene, bool draggable)
    {
        Event e = Event.current;
        bool repaint = e.type == EventType.Repaint;
        Dictionary<string, int> uses = new Dictionary<string, int>();
        if (repaint)
            foreach (LaneEditor.LaneStrip lane in editor.laneStrips)
                foreach (LaneEditor.Section section in lane.sections)
                {
                    if (section.nodeId == null) continue;
                    uses.TryGetValue(section.nodeId, out int count); uses[section.nodeId] = count + 1;
                }
        Handles.CapFunction cap = (controlId, position, rotation, size, eventType) =>
        {
            if (eventType == EventType.Layout || eventType == EventType.MouseMove)
            {
                if (scene.camera.WorldToScreenPoint(position).z <= 0) return;
                float distance = Vector2.Distance(Event.current.mousePosition, HandleUtility.WorldToGUIPoint(position));
                // Keep distance ordering for overlapping nodes; Unity picks controls within five GUI points.
                if (distance <= editor.NodePickRadius) HandleUtility.AddControl(controlId, distance * 4.9f / editor.NodePickRadius);
            }
            else if (eventType == EventType.Repaint) Handles.SphereHandleCap(controlId, position, rotation, size, eventType);
        };
        LaneEditor.LaneNode selectedNode = editor.SelectedNode;
        Vector3 toCamera = scene.camera.transform.forward;
        foreach (LaneEditor.LaneNode node in editor.nodes)
        {
            Vector3 point = editor.transform.TransformPoint(node.position);
            float size = HandleUtility.GetHandleSize(point) * 0.075f * editor.NodeMarkerSize;
            Handles.color = node.isDeadEnd ? Color.red : node.isSpawnPoint ? Color.green : editor.IsNodeInSelectedLane(node.id) ? SelectedColor : Color.cyan;
            if (draggable)
            {
                // Every node allocates its control in the same order, even when a drag changes the selection.
                int hot = GUIUtility.hotControl;
                EditorGUI.BeginChangeCheck();
                Vector3 moved = Handles.FreeMoveHandle(point, size, Vector3.zero, cap);
                bool changed = EditorGUI.EndChangeCheck();
                if ((GUIUtility.hotControl != hot && GUIUtility.hotControl != 0) || changed) { editor.SelectNode(node.id); Repaint(); }
                if (changed) MoveNode(editor, node, moved, scene, snapToSurface);
            }
            else if (repaint) Handles.SphereHandleCap(0, point, Quaternion.identity, size, EventType.Repaint);
            if (!repaint) continue;

            if (node.isDeadEnd)
            {
                Vector3 diagonal = (scene.camera.transform.right + scene.camera.transform.up) * size, other = (scene.camera.transform.right - scene.camera.transform.up) * size;
                Handles.color = Color.red;
                Handles.DrawLine(point - diagonal, point + diagonal); Handles.DrawLine(point - other, point + other);
            }
            if (node.isDeadEnd && node.isSpawnPoint) { Handles.color = Color.green; Handles.DrawWireDisc(point, toCamera, size * 1.5f); }
            // A white ring marks a node where lanes connect.
            else if (uses.TryGetValue(node.id, out int lanes) && lanes > 1) { Handles.color = Color.white; Handles.DrawWireDisc(point, toCamera, size * 1.1f); }
            if (node == selectedNode) { Handles.color = SelectedColor; Handles.DrawWireDisc(point, toCamera, size * 1.8f, 2); }
            if (!showLabels || (!node.isSpawnPoint && !node.isDeadEnd)) continue;
            string label = node.isSpawnPoint ? Mark("Spawn", node.spawnPointName) : "";
            if (node.isDeadEnd) label += (label.Length > 0 ? " / " : "") + Mark("Dead end", node.deadEndName);
            Handles.Label(point + scene.camera.transform.up * size * 2.5f, label, EditorStyles.whiteLabel);
        }
    }
    private static string Mark(string kind, string name) => string.IsNullOrWhiteSpace(name) ? kind : kind + ": " + name;

    private void SelectSceneGUI(LaneEditor editor, SceneView scene, Event e, bool click, int passive)
    {
        if (joining)
        {
            DrawNodes(editor, scene, false);
            Target target = FindTarget(editor, e.mousePosition, editor.CurrentLane, false);
            if (e.type == EventType.Repaint) DrawTargetPreview(editor, scene, target, joinStart);
            if (!click) return;
            e.Use();
            if (!target.valid) { scene.ShowNotification(new GUIContent("Click a node or the centre line of another lane.")); return; }
            bool start = joinStart, joined = false;
            Change(editor, "Join lanes", () => joined = editor.JoinEndpoint(start, ResolveNode(editor, target)));
            if (joined) joining = false;
            Repaint();
            return;
        }
        DrawNodes(editor, scene, true);
        DrawSelectedLaneHandles(editor, scene);
        // No handle under the cursor: the click selects the lane there, or clears the selection.
        if (!click || HandleUtility.nearestControl != passive) return;
        Select(editor, PickLane(editor, e.mousePosition), -1);
        e.Use();
    }

    // Width cubes and insert dots along the selected lane, and the move or rotate handle of its selected node.
    private void DrawSelectedLaneHandles(LaneEditor editor, SceneView scene)
    {
        LaneEditor.LaneStrip lane = editor.CurrentLane;
        if (lane == null) return;
        RoadMeshGeometry.Frame[] frames = RoadMeshGeometry.Frames(editor, lane);
        for (int i = 0; i < frames.Length; i++)
        {
            LaneEditor.Section section = lane.sections[i];
            RoadMeshGeometry.Frame frame = frames[i];
            Handles.color = i == editor.SelectedSection ? Color.green : new Color(0.35f, 0.85f, 0.45f);
            foreach (float sign in Sides)
            {
                Vector3 edge = sign < 0 ? frame.left : frame.right;
                EditorGUI.BeginChangeCheck();
                Vector3 moved = Handles.Slider(edge, frame.side * sign, HandleUtility.GetHandleSize(edge) * 0.07f, Handles.CubeHandleCap, 0);
                if (!EditorGUI.EndChangeCheck()) continue;
                float halfWidth = Mathf.Max(0.05f, Vector3.Dot(moved - frame.center, frame.side * sign));
                float ratio = halfWidth / Mathf.Max(0.01f, Vector3.Distance(edge, frame.center));
                int index = i;
                if (Event.current.shift) Change(editor, "Resize lane", () => { lane.width = Mathf.Max(0.1f, lane.width * ratio); editor.SelectedSection = index; });
                else Change(editor, "Resize lane cross-section", () => { section.widthScale = Mathf.Clamp(section.widthScale * ratio, 0.1f, 10); editor.SelectedSection = index; });
                Repaint();
            }
        }
        for (int i = 0; i < frames.Length - 1; i++)
        {
            Vector3 middle = (frames[i].center + frames[i + 1].center) * 0.5f;
            float size = HandleUtility.GetHandleSize(middle) * 0.045f;
            Handles.color = Color.white;
            if (!Handles.Button(middle, Quaternion.identity, size, size * 2, Handles.DotHandleCap)) continue;
            int segment = i;
            Change(editor, "Insert lane node", () => editor.InsertNode(segment));
            Repaint();
            return; // The lane changed; its handles are redrawn on the next event.
        }

        LaneEditor.LaneNode node = editor.SelectedNode;
        if (node == null || editor.SelectedSection >= frames.Length) return;
        if (Tools.current == Tool.Move)
        {
            // Axis handles move the node exactly as dragged, including in height.
            EditorGUI.BeginChangeCheck();
            Vector3 moved = Handles.PositionHandle(editor.transform.TransformPoint(node.position), Quaternion.identity);
            if (EditorGUI.EndChangeCheck()) MoveNode(editor, node, moved, scene, false);
        }
        else if (Tools.current == Tool.Rotate)
        {
            RoadMeshGeometry.Frame frame = frames[editor.SelectedSection];
            LaneEditor.Section section = lane.sections[editor.SelectedSection];
            EditorGUI.BeginChangeCheck();
            Quaternion rotation = Handles.RotationHandle(frame.rotation, frame.center);
            if (EditorGUI.EndChangeCheck()) Change(editor, "Rotate lane cross-section", () =>
                section.rotation = (Quaternion.Inverse(frame.automaticRotation) * rotation).eulerAngles);
        }
    }

    private void DrawSceneGUI(LaneEditor editor, SceneView scene, Event e, bool click)
    {
        if (drawing && editor.CurrentLane == null) drawing = false;
        DrawNodes(editor, scene, false);
        Target target = FindTarget(editor, e.mousePosition, drawing ? editor.CurrentLane : null, true);
        if (e.type == EventType.Repaint && !panelRect.Contains(e.mousePosition)) DrawTargetPreview(editor, scene, target, false);
        if (!click) return;
        e.Use();
        if (!target.valid) { scene.ShowNotification(new GUIContent("Point at the road to place a lane point.")); return; }
        if (!drawing)
        {
            LaneEditor.LaneStrip started = null; float width = newLaneWidth;
            Change(editor, "Draw lane", () =>
            {
                if (!target.onSurface) started = editor.StartLaneAt(ResolveNode(editor, target), width);
                else { started = editor.AddLane(width); editor.AppendPoint(target.point); }
            });
            drawing = started != null;
        }
        else
        {
            bool joined = false;
            Change(editor, target.onSurface ? "Add lane point" : "Join lanes", () =>
            {
                if (target.onSurface) editor.AppendPoint(target.point);
                else joined = editor.AppendExistingNode(ResolveNode(editor, target));
            });
            if (joined) FinishDrawing(editor);
        }
        Repaint();
    }

    // Shows what a click would do: a small ring adds a point, a green ring joins an existing node or lane.
    private void DrawTargetPreview(LaneEditor editor, SceneView scene, Target target, bool fromStart)
    {
        if (!target.valid) return;
        float size = HandleUtility.GetHandleSize(target.point) * 0.12f;
        Handles.color = target.onSurface ? SelectedColor : JoinColor;
        Handles.DrawWireDisc(target.point, scene.camera.transform.forward, target.onSurface ? size * 0.5f : size, 2);
        if (!target.onSurface) Handles.Label(target.point + scene.camera.transform.up * size * 2, target.nodeId != null ? "join node" : "join lane", EditorStyles.whiteLabel);
        LaneEditor.LaneStrip lane = drawing || joining ? editor.CurrentLane : null;
        if (lane == null || lane.sections.Count == 0) return;
        LaneEditor.LaneNode from = editor.FindNode(lane.sections[fromStart ? 0 : lane.sections.Count - 1].nodeId);
        if (from != null) Handles.DrawDottedLine(editor.transform.TransformPoint(from.position), target.point, 4);
    }

    private void LightsSceneGUI(LaneEditor editor, SceneView scene, Event e, bool click, int passive)
    {
        if (editor.IsDrawingTrafficLight)
        {
            bool hit = SurfacePoint(editor, HandleUtility.GUIPointToWorldRay(e.mousePosition), out Vector3 point);
            if (e.type == EventType.Repaint && hit && !panelRect.Contains(e.mousePosition))
            {
                Handles.color = Color.red;
                if (editor.HasTrafficLightStart) Handles.DrawAAPolyLine(5, editor.TrafficLightStartWorld, point);
                Handles.DrawWireDisc(point, scene.camera.transform.forward, HandleUtility.GetHandleSize(point) * 0.06f, 2);
            }
            if (click)
            {
                if (hit) Change(editor, "Draw traffic light line", () => editor.PlaceTrafficLightPoint(point));
                else scene.ShowNotification(new GUIContent("Point at the road to place the stop line."));
                e.Use(); Repaint();
            }
        }
        int nearest = -1; float nearestDistance = SegmentPickRadius;
        for (int i = 0; i < editor.trafficLights.Count; i++)
        {
            LaneEditor.TrafficLightLine light = editor.trafficLights[i];
            Vector3 start = editor.transform.TransformPoint(light.start), end = editor.transform.TransformPoint(light.end);
            bool selected = editor.SelectedTrafficLight == i;
            if (e.type == EventType.Repaint)
            {
                Handles.color = light.AspectAt(Application.isPlaying ? editor.TrafficLightDisplayTime : 0) == "red" ? Color.red : Color.green;
                Handles.DrawAAPolyLine(selected ? 6 : 3, start, end);
                if (showLabels) Handles.Label((start + end) * 0.5f, light.name, selected ? EditorStyles.whiteBoldLabel : EditorStyles.whiteLabel);
            }
            float distance = HandleUtility.DistanceToLine(start, end);
            if (distance < nearestDistance) { nearestDistance = distance; nearest = i; }
            if (editor.IsDrawingTrafficLight || !selected) continue;
            DrawLightEndpoint(editor, scene, i, true, start);
            DrawLightEndpoint(editor, scene, i, false, end);
        }
        if (!click || editor.IsDrawingTrafficLight || HandleUtility.nearestControl != passive) return;
        editor.SelectedTrafficLight = nearest;
        e.Use(); Repaint();
    }

    private void DrawLightEndpoint(LaneEditor editor, SceneView scene, int index, bool start, Vector3 point)
    {
        Handles.color = SelectedColor;
        EditorGUI.BeginChangeCheck();
        Vector3 moved = Handles.FreeMoveHandle(point, HandleUtility.GetHandleSize(point) * 0.1f, Vector3.zero, Handles.SphereHandleCap);
        if (!EditorGUI.EndChangeCheck()) return;
        if (snapToSurface && !Snap(editor, scene, ref moved)) return;
        Change(editor, "Move traffic light endpoint", () => editor.MoveTrafficLightEndpoint(index, start, moved));
        Repaint();
    }

    // ---------------------------------------------------------------- picking

    private Target FindTarget(LaneEditor editor, Vector2 mouse, LaneEditor.LaneStrip own, bool allowSurface)
    {
        Target target = new Target { lane = -1, segment = -1 };
        target.nodeId = NearestNode(editor, mouse, own);
        if (target.nodeId != null)
        {
            target.point = editor.transform.TransformPoint(editor.FindNode(target.nodeId).position);
            target.valid = true;
        }
        else if (NearestSegment(editor, mouse, own != null ? editor.SelectedLane : -1, out target.lane, out target.segment, out target.t))
        {
            LaneEditor.LaneStrip lane = editor.laneStrips[target.lane];
            Vector3 a = editor.FindNode(lane.sections[target.segment].nodeId).position, b = editor.FindNode(lane.sections[target.segment + 1].nodeId).position;
            target.point = editor.transform.TransformPoint(Vector3.Lerp(a, b, target.t));
            target.valid = true;
        }
        else if (allowSurface && SurfacePoint(editor, HandleUtility.GUIPointToWorldRay(mouse), out target.point))
            target.valid = target.onSurface = true;
        return target;
    }
    private static string ResolveNode(LaneEditor editor, Target target) =>
        target.nodeId ?? (target.lane >= 0 ? editor.NodeOnSegment(target.lane, target.segment, target.t) : null);

    private static string NearestNode(LaneEditor editor, Vector2 mouse, LaneEditor.LaneStrip exclude)
    {
        string closest = null;
        float distance = editor.NodePickRadius;
        Camera camera = SceneView.currentDrawingSceneView.camera;
        foreach (LaneEditor.LaneNode node in editor.nodes)
        {
            if (exclude != null && exclude.sections.Exists(s => s.nodeId == node.id)) continue;
            Vector3 world = editor.transform.TransformPoint(node.position);
            if (camera.WorldToScreenPoint(world).z <= 0) continue;
            float candidate = Vector2.Distance(mouse, HandleUtility.WorldToGUIPoint(world));
            if (candidate < distance) { closest = node.id; distance = candidate; }
        }
        return closest;
    }

    private static bool NearestSegment(LaneEditor editor, Vector2 mouse, int excludeLane, out int laneIndex, out int segmentIndex, out float position)
    {
        laneIndex = segmentIndex = -1;
        position = 0;
        float distance = SegmentPickRadius;
        Camera camera = SceneView.currentDrawingSceneView.camera;
        for (int lane = 0; lane < editor.laneStrips.Count; lane++)
        {
            if (lane == excludeLane) continue;
            LaneEditor.LaneStrip strip = editor.laneStrips[lane];
            for (int segment = 0; segment < strip.sections.Count - 1; segment++)
            {
                LaneEditor.LaneNode first = editor.FindNode(strip.sections[segment].nodeId);
                LaneEditor.LaneNode second = editor.FindNode(strip.sections[segment + 1].nodeId);
                if (first == null || second == null) continue;
                Vector3 start = editor.transform.TransformPoint(first.position), end = editor.transform.TransformPoint(second.position);
                if (camera.WorldToScreenPoint(start).z <= 0 || camera.WorldToScreenPoint(end).z <= 0) continue;
                Vector2 a = HandleUtility.WorldToGUIPoint(start), b = HandleUtility.WorldToGUIPoint(end);
                Vector2 delta = b - a;
                float projected = delta.sqrMagnitude < 0.0001f ? 0 : Mathf.Clamp01(Vector2.Dot(mouse - a, delta) / delta.sqrMagnitude);
                float candidate = Vector2.Distance(mouse, a + delta * projected);
                if (candidate >= distance) continue;
                // Recover the world parameter because GUI projection is perspective-distorted.
                Ray ray = HandleUtility.GUIPointToWorldRay(a + delta * projected);
                Vector3 segmentDelta = end - start, relative = start - ray.origin;
                float lengthSquared = Vector3.Dot(segmentDelta, segmentDelta);
                float parallel = Vector3.Dot(segmentDelta, ray.direction);
                float denominator = lengthSquared - parallel * parallel;
                float worldParameter = denominator > 0.000001f
                    ? (parallel * Vector3.Dot(ray.direction, relative) - Vector3.Dot(segmentDelta, relative)) / denominator
                    : projected;
                distance = candidate;
                laneIndex = lane;
                segmentIndex = segment;
                position = Mathf.Clamp01(worldParameter);
            }
        }
        return laneIndex >= 0;
    }

    // The lane whose centre line is at the cursor, otherwise the lane whose area is under it.
    private static int PickLane(LaneEditor editor, Vector2 mouse)
    {
        if (NearestSegment(editor, mouse, -1, out int byLine, out _, out _)) return byLine;
        Ray ray = HandleUtility.GUIPointToWorldRay(mouse);
        int picked = -1; float nearest = float.PositiveInfinity;
        for (int i = 0; i < editor.laneStrips.Count; i++)
        {
            RoadMeshGeometry.Frame[] frames = RoadMeshGeometry.Frames(editor, editor.laneStrips[i]);
            for (int k = 0; k < frames.Length - 1; k++)
            {
                if (!RayHitsTriangle(ray, frames[k].left, frames[k].right, frames[k + 1].right, out float distance) &&
                    !RayHitsTriangle(ray, frames[k].left, frames[k + 1].right, frames[k + 1].left, out distance)) continue;
                if (distance < nearest) { nearest = distance; picked = i; }
            }
        }
        return picked;
    }
    private static bool RayHitsTriangle(Ray ray, Vector3 a, Vector3 b, Vector3 c, out float distance)
    {
        distance = 0;
        Vector3 ab = b - a, ac = c - a, p = Vector3.Cross(ray.direction, ac);
        float determinant = Vector3.Dot(ab, p);
        if (Mathf.Abs(determinant) < 1e-8f) return false;
        Vector3 offset = ray.origin - a;
        float u = Vector3.Dot(offset, p) / determinant;
        if (u < 0 || u > 1) return false;
        Vector3 q = Vector3.Cross(offset, ab);
        float v = Vector3.Dot(ray.direction, q) / determinant;
        if (v < 0 || u + v > 1) return false;
        distance = Vector3.Dot(ac, q) / determinant;
        return distance > 0;
    }

    // The road model under a ray. Where the model has no collider, level ground at the network's height.
    private static bool SurfacePoint(LaneEditor editor, Ray ray, out Vector3 point)
    {
        if (editor.TryGetSurfacePoint(ray, out point)) return true;
        LaneEditor.LaneNode reference = editor.SelectedNode ?? (editor.nodes.Count > 0 ? editor.nodes[editor.nodes.Count - 1] : null);
        float height = reference != null ? editor.transform.TransformPoint(reference.position).y : editor.transform.position.y;
        if (!new Plane(Vector3.up, new Vector3(0, height, 0)).Raycast(ray, out float distance) || distance > 5000) return false;
        point = ray.GetPoint(distance);
        return true;
    }
    // Move a dragged point onto the road under where it appears on screen.
    private static bool Snap(LaneEditor editor, SceneView scene, ref Vector3 point)
    {
        Vector3 pixel = scene.camera.WorldToScreenPoint(point);
        if (pixel.z <= 0 || !SurfacePoint(editor, scene.camera.ScreenPointToRay(pixel), out Vector3 snapped)) return false;
        point = snapped;
        return true;
    }
    private void MoveNode(LaneEditor editor, LaneEditor.LaneNode node, Vector3 moved, SceneView scene, bool snap)
    {
        if (snap && !Snap(editor, scene, ref moved)) return;
        Change(editor, "Move lane node", () => editor.MoveNode(node.id, moved));
    }

    // F frames the selected node or lane instead of the object's pivot.
    private bool HasFrameBounds() => ((LaneEditor)target).nodes.Count > 0;
    private Bounds OnGetFrameBounds()
    {
        LaneEditor editor = (LaneEditor)target;
        LaneEditor.LaneNode selected = editor.SelectedNode;
        if (selected != null) return new Bounds(editor.transform.TransformPoint(selected.position), Vector3.one * 12);
        List<LaneEditor.LaneNode> framed = new List<LaneEditor.LaneNode>();
        if (editor.CurrentLane != null)
            foreach (LaneEditor.Section section in editor.CurrentLane.sections)
            {
                LaneEditor.LaneNode node = editor.FindNode(section.nodeId);
                if (node != null) framed.Add(node);
            }
        if (framed.Count == 0) framed = editor.nodes;
        Bounds bounds = new Bounds(editor.transform.TransformPoint(framed[0].position), Vector3.one * 4);
        foreach (LaneEditor.LaneNode node in framed) bounds.Encapsulate(editor.transform.TransformPoint(node.position));
        return bounds;
    }

    // ---------------------------------------------------------------- actions

    private static void Change(LaneEditor editor, string label, Action action)
    {
        Undo.RecordObject(editor, label);
        action();
        editor.RebuildVisuals();
        EditorUtility.SetDirty(editor);
        PrefabUtility.RecordPrefabInstancePropertyModifications(editor);
        SceneView.RepaintAll();
    }

    private static void PrepareColliders(LaneEditor editor, bool always = false)
    {
        if (!always && !editor.prepareMeshColliders) return;
        Undo.IncrementCurrentGroup();
        int group = Undo.GetCurrentGroup();
        Undo.SetCurrentGroupName("Add road mesh colliders");
        int added = editor.PrepareSurfaceColliders(go => Undo.AddComponent<MeshCollider>(go));
        Undo.CollapseUndoOperations(group);
        if (always) Debug.Log("Added " + added + " mesh colliders to the road model.", editor);
    }

    private static void ExportCameraLanes(LaneEditor editor)
    {
        try
        {
            Camera camera = CameraLaneExport.ResolveCamera(editor);
            var layout = CameraLaneExport.Build(editor, camera, editor.exportImageSize.x, editor.exportImageSize.y);
            string path = CameraLaneExport.DefaultPath(editor);
            CameraLaneExport.Write(path, layout);
            AssetDatabase.Refresh();
            Debug.Log("Exported camera lane layout through " + camera.name + ": " + path, editor);
            EditorUtility.DisplayDialog("Camera lane layout", "Exported through " + camera.name + " to:\n" + path, "OK");
        }
        catch (Exception error)
        {
            Debug.LogError("Camera lane export failed: " + error.Message, editor);
            EditorUtility.DisplayDialog("Camera lane layout", "Export failed:\n" + error.Message, "OK");
        }
        GUIUtility.ExitGUI();
    }

    private static void SaveJson(LaneEditor editor)
    {
        string path = EditorUtility.SaveFilePanel("Save lane network", "", editor.gameObject.scene.name + ".lanes.json", "json");
        if (!string.IsNullOrEmpty(path))
            try
            {
                File.WriteAllText(path, editor.ToJson());
                Debug.Log("Saved lane network: " + path, editor);
            }
            catch (Exception error) { EditorUtility.DisplayDialog("Save lane network", "Save failed:\n" + error.Message, "OK"); }
        GUIUtility.ExitGUI();
    }

    private void LoadJson(LaneEditor editor)
    {
        string path = EditorUtility.OpenFilePanel("Load lane network", "", "json");
        if (!string.IsNullOrEmpty(path))
            try
            {
                string json = File.ReadAllText(path);
                EndGesture(editor);
                Change(editor, "Load lane network", () => editor.LoadJson(json));
            }
            catch (Exception error) { EditorUtility.DisplayDialog("Load lane network", "Load failed:\n" + error.Message, "OK"); }
        GUIUtility.ExitGUI();
    }
}
