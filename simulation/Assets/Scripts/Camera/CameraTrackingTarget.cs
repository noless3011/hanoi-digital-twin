using System.Collections.Generic;
using UnityEngine;

/// <summary>Marks one object as a projection source; its identity is never sent to the tracker.</summary>
[DisallowMultipleComponent]
public sealed class CameraTrackingTarget : MonoBehaviour
{
    public string kind = "car";
    internal static readonly HashSet<CameraTrackingTarget> Active = new HashSet<CameraTrackingTarget>();
    internal Renderer[] Renderers { get; private set; } = new Renderer[0];
    private void OnEnable() { RefreshRenderers(); Active.Add(this); }
    private void OnDisable() { Active.Remove(this); }
    /// <summary>Call after changing the object's mesh hierarchy.</summary>
    public void RefreshRenderers() { Renderers = GetComponentsInChildren<Renderer>(true); }
}
