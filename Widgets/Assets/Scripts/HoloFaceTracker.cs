using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using System;
using System.Runtime.InteropServices;
using Shared;

using System.Net;
using System.Net.Sockets;
using System.Threading;

#if ENABLE_WINMD_SUPPORT
using UnityEngine.XR.WSA;
using Windows.Storage;
#endif

#if ENABLE_WINMD_SUPPORT || NETFX_CORE
using Windows.Graphics.Imaging;
using Windows.Media;
using Windows.Media.MediaProperties;
using Windows.Foundation;
using Windows.Media.Capture.Frames;
using Windows.Media.Devices.Core;
using Windows.Perception.Spatial;
using Windows.Graphics.Holographic;
using Windows.Perception;
using Windows.UI.Input.Spatial;
using Debug = Shared.Debug;
#endif

/*
 * Ref : https://github.com/cookieofcode/hololens2-unity-uwp-starter
 */

public class HoloFaceTracker : MonoBehaviour
{
    private VideoFrameProcessor _videoFrameProcessor;
    private bool _isReadyToRender = false;
    private TimeSpan _previousFrameTimestamp;
    private FaceTrackerProcessor _faceTrackerProcessor;
    private bool _isTrackingFaces;
    private int faceNum = 0;

    private int seconds = 0;
    private const int interval = 33;
    private bool isInit = true;

    [HideInInspector] public bool isTxStarted = false;
    [SerializeField] int txPort; // port to send data to Python on
    [SerializeField] int rxPort; // port to receive data from Python on
    // Create necessary UdpClient objects
    UdpClient client;
    IPEndPoint remoteEndPoint;
    Thread sendThread;

    private string sendRow = "";

    // Start is called before the first frame update

    async void Ready()
    {
#if ENABLE_WINMD_SUPPORT
        _videoFrameProcessor = await VideoFrameProcessor.CreateAsync();
        _faceTrackerProcessor = await FaceTrackerProcessor.CreateAsync(_videoFrameProcessor);
#endif
    }

    // Update is called once per frame
    void Update()
    {
        if(StateParam.expState == State.faceReady)
        {
            // Create remote endpoint
            remoteEndPoint = new IPEndPoint(IPAddress.Parse(StateParam.ip), txPort);
            UnityEngine.Debug.Log("Face" + StateParam.ip);

            // Create local client
            client = new UdpClient(rxPort);

            // Initialize (seen in comments window)
            UnityEngine.Debug.Log("[Face]UDP Comms Initialised");
            StateParam.expState = State.gatheringReady;
        }

        if (StateParam.expState == State.gatheringReady && isInit)
        {
            SendData("Ready_hololense_face");
            UnityEngine.Debug.Log("Ready_hololense_face");
            Ready();
            _isReadyToRender = true;
            isInit = false;
            sendThread = new Thread(new ThreadStart(CMakeRowBodys));
            sendThread.IsBackground = true;
            sendThread.Start();
            //StartCoroutine(CMakeRowBodys());
        }

#if ENABLE_WINMD_SUPPORT
        if (_isReadyToRender)
        {
            _isTrackingFaces = _faceTrackerProcessor.IsTrackingFaces();

            if (_isTrackingFaces)
            {
                MediaFrameReference frame = _videoFrameProcessor.GetLatestFrame();
                if (frame == null)
                {
                    return;
                }
                var faces = _faceTrackerProcessor.GetLatestFaces();
                faceNum = faces.Count;
            }
            else
            {
                faceNum = 0;
            }
        }
#endif
    }

    private void CMakeRowBodys()
    {
        while (true)
        {
            if (_isReadyToRender)
            {
                int nRows = 2;
                string[] rowDatas = new string[nRows];

                seconds += interval;
                rowDatas[0] = seconds.ToString();
                rowDatas[1] = faceNum.ToString();
                sendRow = rowDatas[0] + "," + rowDatas[1];
                SendData(sendRow);
            }
            
            if (StateParam.expState == State.End)
            {
                if (sendThread != null)
                    sendThread.Abort();
                client.Close();
                break;
            }

            Thread.Sleep(interval);
        }
    }

    public void SendData(string message) // Use to send data to Python
    {
        try
        {
            UnityEngine.Debug.Log("Send Face" + message);
            byte[] data = Encoding.UTF8.GetBytes(message);
            client.Send(data, data.Length, remoteEndPoint);
        }
        catch (Exception err)
        {
            print(err.ToString());
        }
    }

    void CsvAddRow(string[] rows, List<string[]> rowData)
    {
        string[] rowDataTemp = new string[rows.Length];
        for (int i = 0; i < rows.Length; i++)
            rowDataTemp[i] = rows[i];
        rowData.Add(rowDataTemp);
    }

    public void WriteCsv(List<string[]> rowData, string filePath)
    {
        string[][] output = new string[rowData.Count][];

        for (int i = 0; i < output.Length; i++)
        {
            output[i] = rowData[i];
        }

        int length = output.GetLength(0);
        string delimiter = ",";

        StringBuilder stringBuilder = new StringBuilder();

        for (int index = 0; index < length; index++)
            stringBuilder.AppendLine(string.Join(delimiter, output[index]));

        Stream fileStream = new FileStream(filePath, FileMode.CreateNew, FileAccess.Write);
        StreamWriter outStream = new StreamWriter(fileStream, Encoding.UTF8);
        outStream.WriteLine(stringBuilder);
        outStream.Close();
    }

    private void OnDisable()
    {
        if (sendThread != null)
            sendThread.Abort();
    }
}
