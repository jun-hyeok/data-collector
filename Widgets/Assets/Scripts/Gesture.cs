using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Microsoft;
using Microsoft.MixedReality.Toolkit.Utilities;
using Microsoft.MixedReality.Toolkit.Input;

public class Gesture : MonoBehaviour
{
    private Handedness rightHand = Handedness.Right;
    private Handedness leftHand = Handedness.Left;
    Single rightpinchValue;
    Single leftpinchValue;
    private const float PinchThreshold = 0.7f;
    public Text StateText = null;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (HandJointUtils.FindHand(rightHand) != null)
        {
            rightpinchValue = HandPoseUtils.CalculateIndexPinch(rightHand);
            Debug.Log(rightpinchValue);
            if (rightpinchValue > PinchThreshold && rightpinchValue != 1)
            {
                Debug.Log("rigth pinched");
                if (StateText.text == "Ready")
                {
                    Debug.Log("Ready");
                    StateText.text = "Recording...";
                }
            }
            Debug.Log("Right HAND DETECTED");
        }

        if (HandJointUtils.FindHand(leftHand) != null)
        {
            leftpinchValue = HandPoseUtils.CalculateIndexPinch(leftHand);
            Debug.Log(leftpinchValue);
            if (leftpinchValue > PinchThreshold && leftpinchValue != 1)
            {
                Debug.Log("left pinched");
                if (StateText.text == "Recording...")
                {
                    Debug.Log("Recording...");
                    StateText.text = "Ready";
                }
            }
            Debug.Log("Left HAND DETECTED");
        }
    }
}
