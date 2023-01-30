using System;
using System.Collections.ObjectModel;
using UnityEngine;
using UnityEngine.Assertions;
using Microsoft.MixedReality.Toolkit.Utilities.Solvers;

public class WidgetDocks : MonoBehaviour
{
    [SerializeField]
    [Tooltip("A read-only list of possible positions in this dock.")]
    private ReadOnlyCollection<WidgetDock> dockPositions;

    /// <summary>
    /// A read-only list of possible positions in this dock.
    /// </summary>
    public ReadOnlyCollection<WidgetDock> DockPositions => dockPositions;

    /// <summary>
    /// Initializes the list of positions in this dock.
    /// </summary>
    private void OnEnable()
    {
        UpdatePositions();
    }

    /// <summary>
    /// Updates the list of positions in this dock when its children change.
    /// </summary>
    private void OnTransformChildrenChanged()
    {
        UpdatePositions();
    }

    /// <summary>
    /// Updates the list of positions in this dock.
    /// </summary>
    private void UpdatePositions()
    {
        // dockPositions = gameObject.GetComponentsInChildren<WidgetDock>().ToReadOnlyCollection();
    }

    /// <summary>
    /// Moves elements near the desired position to make space for a new element,
    /// if possible.
    /// </summary>
    /// <param name="position">The desired position where an object wants to be docked.</param>
    /// <returns>Returns true if the desired position is now available, false otherwise.</returns>
    public bool TryMoveToFreeSpace(WidgetDock position)
    {
        if (dockPositions == null)
        {
            UpdatePositions();
        }

        if (!dockPositions.Contains(position))
        {
            Debug.LogError("Looking for a DockPosition in the wrong Dock.");
            return false;
        }

        var index = dockPositions.IndexOf(position);

        if (!dockPositions[index].IsOccupied)
        {
            // Already free
            return true;
        }

        // Where is the closest free space? (on a tie, favor left)
        int? closestFreeSpace = null;
        int distanceToClosestFreeSpace = int.MaxValue;
        for (int i = 0; i < dockPositions.Count; i++)
        {
            var distance = Math.Abs(index - i);
            if (!dockPositions[i].IsOccupied && distance < distanceToClosestFreeSpace)
            {
                closestFreeSpace = i;
                distanceToClosestFreeSpace = distance;
            }
        }

        if (closestFreeSpace == null)
        {
            // No free space
            return false;
        }

        if (closestFreeSpace < index)
        {
            // Move left

            // Check if we can undock all of them
            for (int i = closestFreeSpace.Value + 1; i <= index; i++)
            {
                if (!dockPositions[i].DockedObject.CanUndock)
                {
                    return false;
                }
            }

            for (int i = closestFreeSpace.Value + 1; i <= index; i++)
            {
                MoveDockedObject(i, i - 1);
            }
        }
        else
        {
            // Move right

            // Check if we can undock all of them
            for (int i = closestFreeSpace.Value - 1; i >= index; i--)
            {
                if (!dockPositions[i].DockedObject.CanUndock)
                {
                    return false;
                }
            }

            for (int i = closestFreeSpace.Value - 1; i >= index; i--)
            {
                MoveDockedObject(i, i + 1);
            }
        }

        return true;
    }

    /// <summary>
    /// Moves a docked object from a position to another, by undocking it 
    /// and docking it in the new position.
    /// </summary>
    /// <param name="from">The position we're moving the object from.</param>
    /// <param name="to">The position we're moving the object to.</param>
    private void MoveDockedObject(int from, int to)
    {
        var objectToMove = dockPositions[from].DockedObject;
        objectToMove.Undock();
        objectToMove.Dock(dockPositions[to]);
        Assert.AreEqual(dockPositions[to].DockedObject, objectToMove, "The object we just moved needs to match the object docked in the new position.");
    }


    public float holdTime;
    float holdTimer;
    RadialView radialView;

    void Start()
    {
        radialView = gameObject.GetComponent<RadialView>();
    }

    void Update()
    {
        if (holdTimer > 0)
        {
            holdTimer -= Time.deltaTime;
        }
        else
        {
            OnHoldTimerEnd();
        }

    }

    public void OnHoldWidgets()
    {
        holdTimer = holdTime;
        radialView.MoveLerpTime = 9999;
        radialView.RotateLerpTime = 9999;
    }

    public void OnHoldTimerEnd()
    {
        radialView.MoveLerpTime = 1f;
        radialView.RotateLerpTime = 1f;
    }
}
