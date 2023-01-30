using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class StateParam
{
    public static State expState = State.Ready;

    public static string ip = "0.0.0.0";
}

public enum State
{
    Ready,
    labelInput,
    Cognitive,
    cognitiveReady,
    imuReady,
    faceReady,
    gatheringReady,
    gathering,
    End, 
}