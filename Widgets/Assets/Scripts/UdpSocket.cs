using UnityEngine;
using System.Collections;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine.Networking;

public class UdpSocket : MonoBehaviour
    /*
     홀로렌즈 IP Address 확인 테스트용.
     */
{
    [HideInInspector] public bool isTxStarted = false;

    string IP;
    [SerializeField] int rxPort = 3000; // port to receive data from Python on
    [SerializeField] int txPort = 3001; // port to send data to Python on
    // Google Spread Sheet에서 ip 주소 데이터를 받아와서 설정. 
    private string url = "https://script.google.com/macros/s/AKfycbwN8eSbtq9myXTFcw8bL2i2N6R21Yr-2m41S9rdZFCMpi1tltsl1zblvpeTWNmHlKNy/exec";

    private string message = "Test_hololens";

    // Create necessary UdpClient objects
    UdpClient client;
    IPEndPoint remoteEndPoint;
    Thread receiveThread; // Receiving Thread

    IEnumerator SendDataCoroutine()
    {
        while (true)
        {
            SendData(message);
            yield return new WaitForSeconds(0.5f);
        }
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

    private void Awake()
    {
        Debug.Log("Awake");
        UnityWebRequest www = UnityWebRequest.Get(url);
        www.SendWebRequest();
        while (!www.isDone)
        {
            // Google Spread Sheet에서 데이터를 받아온다. 못 받아오는 경우 있음 revision 필요. 
            IP = www.downloadHandler.text;
        }
        Debug.Log(IP);
    }

    void Start()
    {
        // Create remote endpoint (to Matlab) 
        remoteEndPoint = new IPEndPoint(IPAddress.Parse(IP), txPort);

        // Initialize (seen in comments window)
        print("UDP Comms Initialised");
    }

    public void GetIP()
    {
        //메세지를 파이썬 서버에 전송
        Debug.Log("press");
        Debug.Log(IP);
        receiveThread.Start();
        StartCoroutine(SendDataCoroutine());
    }
}
