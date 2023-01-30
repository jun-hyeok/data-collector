using System;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Threading;
using UnityEngine;

using System.Net;
using System.Net.Sockets;


public class AudioStream : MonoBehaviour
{
    public AudioClip voiceClip;
    private int byteWindow = 512;
    public string microphoneName;

    [HideInInspector] public bool isTxStarted = false;
    [SerializeField] int txPort=9001; // port to send data to Python on
    [SerializeField] int rxPort=9000; // port to receive data from Python on
    // Create necessary UdpClient objects
    UdpClient client;
    IPEndPoint remoteEndPoint;
    Thread receiveThread; // Receiving Thread

    private int prevPos = 0;
    private int curPos = 0;

    private ConcurrentQueue<byte> buffer = null;
    private List<byte> sendBuffer = null;
    Thread bufferSendThread;
    private bool isInit = true;

    private void Awake()
    {
        // Create remote endpoint (to Matlab) 
        remoteEndPoint = new IPEndPoint(IPAddress.Parse(StateParam.ip), txPort);

        // Create local client
        client = new UdpClient(rxPort);

        // Initialize (seen in comments window)
        Debug.Log("[Audio]UDP Comms Initialised");

        buffer = new ConcurrentQueue<byte>();
        // Create buffer to save Audio Bytes
        sendBuffer = new List<byte>();
        // UDP send thread
        bufferSendThread = new Thread(ByteBuffer);
        bufferSendThread.IsBackground = true;
    }

    // Start is called before the first frame update
    void Start()
    {
        MicroPhoneToVoiceClip();
    }

    public void MicroPhoneToVoiceClip()
    {
        microphoneName = Microphone.devices[0];
        Debug.Log(microphoneName);
        for (int i = 0; i < Microphone.devices.Length; i++)
        {
            Debug.Log(Microphone.devices[i]);
        }
        // Recording Start
        voiceClip = Microphone.Start(microphoneName, false, 300, AudioSettings.outputSampleRate);
    }

    // Update is called once per frame
    void Update()
    {
        if (StateParam.expState == State.gatheringReady && isInit)
        {
            SendReady("Ready_hololense_Audio");
            Debug.Log("Ready_hololense_Audio");
            StartCoroutine(StreamByte());
            bufferSendThread.Start();
            isInit = false;
        }
    }

    IEnumerator StreamByte()
    {
        /*
         Trim voiceClip -> convert to bytes -> Enqueue to buffer queue
         */
        Debug.Log("StreamByte");
        while (true)
        {
            curPos = Microphone.GetPosition(microphoneName);
            if (curPos == prevPos)
                continue;
            Debug.Log(curPos / voiceClip.frequency);
            byte[] byteRows = Convert(voiceClip, curPos, prevPos);
            prevPos = curPos;
            ArrayEnqueue(buffer, byteRows);
            if (curPos == voiceClip.samples)
                break;
            if (StateParam.expState == State.End)
            {
                bufferSendThread.Abort();
                client.Close();
                break;
            }
                
            yield return null;
        }
    }

    private void OnDestroy()
    {
        StopCoroutine(StreamByte());
        bufferSendThread.Abort();
    }

    private void ByteBuffer()
    {
        /*
         create bytes packet from buffer queue
        send packet through UDP when packet size is 1024
         */
        while (true)
        {
            try
            {
                byte result;
                if (buffer.TryDequeue(out result))
                {
                    sendBuffer.Add(result);
                    if (sendBuffer.Count >= byteWindow)
                    {
                        Debug.Log("Send Audio bytes");
                        SendData(sendBuffer.ToArray());
                        sendBuffer.Clear();
                    }
                }
                else
                {
                    Debug.Log("Wait");
                    continue;
                }
            }
            catch (Exception err)
            {
                print(err.ToString());
            }
        }
    }

    public void ArrayEnqueue(ConcurrentQueue<byte> queue, IEnumerable<byte> enu)
    {
        foreach (byte obj in enu)
            queue.Enqueue(obj);
    }

    private byte[] Convert(AudioClip clip, int pos, int prev)
    {
        int window = pos - prev;
        var samples = new float[window];

        clip.GetData(samples, prev);

        Int16[] intData = new Int16[window];
        //converting in 2 float[] steps to Int16[], //then Int16[] to Byte[]

        Byte[] bytesData = new Byte[window * 2];
        //bytesData array is twice the size of
        //dataSource array because a float converted in Int16 is 2 bytes.

        int rescaleFactor = 32767; //to convert float to Int16

        for (int i = 0; i < window; i++)
        {
            intData[i] = (short)(samples[i] * rescaleFactor);
            Byte[] byteArr = new Byte[2];
            byteArr = BitConverter.GetBytes(intData[i]);
            byteArr.CopyTo(bytesData, i * 2);
        }

        return bytesData;
    }

    public void SendData(byte[] bytedata) // Use to send data to Python
    {
        Debug.Log("SendData");
        try
        {
            client.Send(bytedata, bytedata.Length, remoteEndPoint);
        }
        catch (Exception err)
        {
            print(err.ToString());
        }
    }

    public void SendReady(string message) // Use to send data to Python
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
