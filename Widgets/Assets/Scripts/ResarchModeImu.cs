using System;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

using System.Net;
using System.Net.Sockets;
using System.Threading;

#if ENABLE_WINMD_SUPPORT
using HL2UnityPlugin;
#endif

/*
 * Ref : https://github.com/IkbeomJeon/HoloLens2-ResearchMode-Unity
 * !중요 : Package.appxmanifest 파일 설정 필견
 */

public class ResarchModeImu : MonoBehaviour
{
#if ENABLE_WINMD_SUPPORT
    HL2ResearchMode researchMode;
#endif

    private float[] accelSampleData = null;
    private float[] gyroSampleData = null;
    private float[] magSampleData = null;
    private bool isInit = true;
    public string[] m_ColumnHeadings = { "Time", "AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ", "MagX", "MagY", "MagZ" };
    private int seconds = 0;
    private const int interval = 33;

    private string sendRow = "";
    [HideInInspector] public bool isTxStarted = false;
    [SerializeField] int txPort; // port to send data to Python on
    [SerializeField] int rxPort; // port to receive data from Python on
    // Create necessary UdpClient objects
    UdpClient client;
    IPEndPoint remoteEndPoint;
    Thread sendThread;
    // Start is called before the first frame update
    void Start()
    {
#if ENABLE_WINMD_SUPPORT
        Debug.Log("Start Intilize");
        researchMode = new HL2ResearchMode();
        researchMode.InitializeAccelSensor();
        researchMode.InitializeGyroSensor();
        researchMode.InitializeMagSensor();

        researchMode.StartAccelSensorLoop();
        researchMode.StartGyroSensorLoop();
        researchMode.StartMagSensorLoop();
        Debug.Log("Start Loop");
#endif
    }

    // Update is called once per frame
    void Update()
    {
        if(StateParam.expState == State.imuReady)
        {
            // Create remote endpoint (to Matlab) 
            remoteEndPoint = new IPEndPoint(IPAddress.Parse(StateParam.ip), txPort);
            Debug.Log("imu" + StateParam.ip);

            // Create local client
            client = new UdpClient(rxPort);

            // Initialize (seen in comments window)
            Debug.Log("[IMU]UDP Comms Initialised");
            StateParam.expState = State.faceReady;
        }

        if (StateParam.expState == State.gatheringReady && isInit)
        {
            SendData("Ready_hololense_imu");
            Debug.Log("Ready_hololense_imu");
            isInit = false;
            //StartCoroutine(CMakeRowBodys(m_ColumnHeadings.Length));
            sendThread = new Thread(new ThreadStart(CMakeRowBodys));
            sendThread.IsBackground = true;
            sendThread.Start();
        }
    }

    private void CMakeRowBodys()
    {
        while (true)
        {
            int nRows = m_ColumnHeadings.Length;
            string[] rowDatas = new string[nRows];

            seconds += interval;
            sendRow = "";
            rowDatas[0] = seconds.ToString();

#if ENABLE_WINMD_SUPPORT
            // update Accel Sample
            if (researchMode.AccelSampleUpdated())
            {
                Debug.Log("Accel updated");
                accelSampleData = researchMode.GetAccelSample();
                if (accelSampleData.Length == 3)
                {
                    Debug.Log("Get accel");
                    rowDatas[1] = accelSampleData[0].ToString();
                    rowDatas[2] = accelSampleData[1].ToString();
                    rowDatas[3] = accelSampleData[2].ToString();
                }
            }

            if (researchMode.GyroSampleUpdated())
            {
                Debug.Log("Gyro updated");
                gyroSampleData = researchMode.GetGyroSample();
                if (gyroSampleData.Length == 3)
                {
                    Debug.Log("Get gyro");
                    rowDatas[4] = gyroSampleData[0].ToString();
                    rowDatas[5] = gyroSampleData[1].ToString();
                    rowDatas[6] = gyroSampleData[2].ToString();
                }
            }

            if (researchMode.MagSampleUpdated())
            {
                Debug.Log("Mag updated");
                magSampleData = researchMode.GetMagSample();
                if (magSampleData.Length == 3)
                {
                    Debug.Log("Get mag");
                    rowDatas[7] = magSampleData[0].ToString();
                    rowDatas[8] = magSampleData[1].ToString();
                    rowDatas[9] = magSampleData[2].ToString();
                }
            }
#endif
            foreach (var data in rowDatas)
            {
                sendRow += data + ",";
            }
            SendData(sendRow);

            if (StateParam.expState == State.End)
            {
                if (sendThread != null)
                    sendThread.Abort();
                client.Close();
                break;
            }

            //yield return new WaitForSeconds(interval);
            Thread.Sleep(interval);
        }
    }

    public void SendData(string message) // Use to send data to Python
    {
        try
        {
            UnityEngine.Debug.Log("Send IMU" + message);
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
        Debug.Log("++++++++++++save path++++++++++");
        Debug.Log(filePath);
        StreamWriter outStream = new StreamWriter(fileStream, Encoding.UTF8);
        outStream.WriteLine(stringBuilder);
        outStream.Close();
    }

    public void StopSensorsEvent()
    {
#if ENABLE_WINMD_SUPPORT
        researchMode.StopAllSensorDevice();
#endif
    }

    private void OnApplicationFocus(bool focus)
    {
        Debug.Log("Out focus");
        if (!focus) StopSensorsEvent();
    }

    private void OnDisable()
    {
        if (sendThread != null)
            sendThread.Abort();
    }
}
