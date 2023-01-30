using System;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using UnityEngine.Networking;

public class CognitiveStatus : MonoBehaviour
{
    public GameObject CongnitiveEmotionalStatus;
    private string url = "https://script.google.com/macros/s/AKfycbwN8eSbtq9myXTFcw8bL2i2N6R21Yr-2m41S9rdZFCMpi1tltsl1zblvpeTWNmHlKNy/exec";
    [SerializeField] int rxPort; // port to receive data from Python on
    [SerializeField] int txPort; // port to send data to Python on
    private string cognitiveLoad = "None";
    private string emotion = "None";

    // Create necessary UdpClient objects
    UdpClient client;
    IPEndPoint remoteEndPoint;

    private void Update()
    {
        if (StateParam.expState == State.Ready)
        {
            CongnitiveEmotionalStatus.SetActive(true);
        }
    }

    private void Awake()
    {
        Debug.Log("Awake");
        UnityWebRequest www = UnityWebRequest.Get(url);
        www.SendWebRequest();
        while (!www.isDone)
        {
            string data = www.downloadHandler.text;
            StateParam.ip = data;
        }
    }

    public void IPCheck()
    {
        Debug.Log("Current server IP is " + StateParam.ip);
        UnityWebRequest www = UnityWebRequest.Get(url);
        www.SendWebRequest();
        while (!www.isDone)
        {
            string data = www.downloadHandler.text;
            StateParam.ip = data;
        }
        Debug.Log("IP is changed to " + StateParam.ip);
        remoteEndPoint = new IPEndPoint(IPAddress.Parse(StateParam.ip), txPort);
        Debug.Log("Congnitive" + StateParam.ip);
        client = new UdpClient(rxPort);
        Debug.Log("CognitiveStatus Comms Initialised");
    }

    private void OnDisable()
    {
        client.Close();
    }

    public void SetCognitiveLoad(string selected)
    {
        cognitiveLoad = selected;
    }

    public void SetEmotion(string selected)
    {
        emotion = selected;
    }

    public void SaveStatus()
    {
        Debug.Log(cognitiveLoad + "_" + emotion);
        SendData(cognitiveLoad + "_" + emotion);
        StateParam.expState = State.cognitiveReady;
    }

    public void SendData(string message) // Use to send data to Python
    {
        try
        {
            Debug.Log(message);
            byte[] data = Encoding.UTF8.GetBytes(message);
            client.Send(data, data.Length, remoteEndPoint);
        }
        catch (Exception err)
        {
            print(err.ToString());
        }
    }


}
