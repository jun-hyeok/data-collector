using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Text;

public static class LogTracker
{
    public static Dictionary<string, DockingState> widget_list = new Dictionary<string, DockingState>();

    public static void add_widget(string name)
    {
        widget_list.Add(name, DockingState.Undocked);
    }

    public static void remove_widget(string name)
    {
        widget_list.Remove(name);
    }

    public static void update_widget(string name, DockingState isDocked)
    {
        widget_list[name] = isDocked;
    }

    public static DockingState get_value(string name)
    {
        return widget_list[name];
    }

    public static void check_dict()
    {
        Debug.Log("==============");
        foreach(var kvp in widget_list)
        {
            Debug.Log(string.Format("Key : {0}, Value : {1}", kvp.Key, kvp.Value));
        }
        Debug.Log("==============");
    }
}
