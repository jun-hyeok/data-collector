using System.Collections;
using System;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Text;

using System.Net;
using System.Net.Sockets;
using System.Threading;

public class ExpManager : MonoBehaviour
{
    private int seconds = 0;
    private const int interval = 33;
    private bool isInit = true;
    public GameObject timer;

    [HideInInspector] public bool isTxStarted = false;
    [SerializeField] int txPort; // port to send data to Python on
    [SerializeField] int rxPort; // port to receive data from Python on
    // Create necessary UdpClient objects
    UdpClient client;
    IPEndPoint remoteEndPoint;
    Thread receiveThread; // Receiving Thread
    Thread sendThread;
    private string sendRow = "";

    private void ReceiveData()
    {
        while (true)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = client.Receive(ref anyIP);
                string text = Encoding.UTF8.GetString(data);
                if (text == "Start" && StateParam.expState == State.gatheringReady)
                {
                    StateParam.expState = State.gathering;
                }

                if (text == "End" && StateParam.expState == State.gathering)
                {
                    StateParam.expState = State.End;
                }
            }
            catch (Exception err)
            {
                print(err.ToString());
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        if(StateParam.expState == State.cognitiveReady)
        {
            // Create remote endpoint (to Matlab) 
            remoteEndPoint = new IPEndPoint(IPAddress.Parse(StateParam.ip), txPort);

            // Create local client
            client = new UdpClient(rxPort);

            receiveThread = new Thread(new ThreadStart(ReceiveData));
            receiveThread.IsBackground = true;
            receiveThread.Start();

            // Initialize (seen in comments window)
            Debug.Log("[Widget]UDP Comms Initialised");
            StateParam.expState = State.imuReady;
        }

        if (StateParam.expState == State.gatheringReady && isInit)
        {
            SendData("Ready_hololense_widget");
            Debug.Log("Ready_hololense_widget");
            //StartCoroutine(CMakeRowBodys());
            sendThread = new Thread(new ThreadStart(CMakeRowBodys));
            sendThread.IsBackground = true;
            sendThread.Start();
            isInit = false;
        }

        if (StateParam.expState == State.gathering && !timer.activeSelf)
        {
            Debug.Log("Start_gathering");
            timer.SetActive(true);
        }
    }

    private void CMakeRowBodys()
    {
        while (true)
        {
            Dictionary<string, DockingState> widget_list = LogTracker.widget_list;
            int nRows = widget_list.Keys.Count*2 + 1;
            string[] rowDatas = new string[nRows];

            seconds += interval;
            sendRow = seconds.ToString() + ",";
            rowDatas[0] = seconds.ToString();
            int i = 1;
            foreach(var kvp in widget_list)
            {
                rowDatas[i] = kvp.Key;
                sendRow += kvp.Key + ",";
                rowDatas[i + 1] = kvp.Value.ToString();
                sendRow += kvp.Value.ToString() + ",";
                i += 2;
            }
            SendData(sendRow);

            if (StateParam.expState == State.End)
            {
                if (receiveThread != null)
                    receiveThread.Abort();
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
            UnityEngine.Debug.Log("Send Widget" + message);
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

    private void OnDisable()
    {
        if (receiveThread != null)
            receiveThread.Abort();
        if (sendThread != null)
            sendThread.Abort();
    }
}
