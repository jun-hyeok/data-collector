using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if ENABLE_WINMD_SUPPORT
using Windows.Storage;
#endif

using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Text;

public class AudioHandler : MonoBehaviour
{
    public AudioClip voiceClip;
    public int sampleWindow = 128;
    public string microphoneName;
#if ENABLE_WINMD_SUPPORT
    private StorageFolder archivedSourceFolder;
#endif
    private SavWav savwav = new SavWav();
    private string m_Path;
    private string m_FilePath;
    public string m_FilePrefix = "Audios_";
    public float loudnessSensibility = 100f;
    public float threshold = 0.1f;
    public Vector3 minScale;
    public Vector3 maxScale;
    private bool isWriting = false;

    private float seconds = 0;
    private const float interval = 0.0333333f;
    [HideInInspector] public bool isTxStarted = false;
    [SerializeField] string IP = "192.168.1.141"; // local host
    [SerializeField] int txPort = 8004; // port to send data to Python on
    [SerializeField] int rxPort = 8000; // port to receive data from Python on
    // Create necessary UdpClient objects
    UdpClient client;
    IPEndPoint remoteEndPoint;
    Thread receiveThread; // Receiving Thread
    private string sendRow = "";
    const int HEADER_SIZE = 44;
    // Start is called before the first frame update
    void Start()
    {
#if ENABLE_WINMD_SUPPORT
        archivedSourceFolder = ApplicationData.Current.LocalFolder;
        m_Path = archivedSourceFolder.Path;
#endif
        Debug.Log(m_Path);
        // Create remote endpoint (to Matlab) 
        remoteEndPoint = new IPEndPoint(IPAddress.Parse(IP), txPort);

        // Create local client
        client = new UdpClient(txPort);
    }

    public void MicroPhoneToVoiceClip()
    {
        microphoneName = Microphone.devices[0];
        for (int i = 0; i < Microphone.devices.Length; i++)
        {
            Debug.Log(Microphone.devices[i]);
        }

        voiceClip = Microphone.Start(microphoneName, false, 660, AudioSettings.outputSampleRate);
    }

    // Update is called once per frame
    void Update()
    {
        if (StateParam.expState == State.gatheringReady && Input.GetKeyDown(KeyCode.Space) && !isWriting)
        {
            MicroPhoneToVoiceClip();
            isWriting = true;
            m_FilePath = m_Path + @"\" + m_FilePrefix + ".wav";
            StartCoroutine(StreamByte());
        }

        if ((StateParam.expState == State.gathering || StateParam.expState == State.End) && Input.GetKeyDown(KeyCode.Space))
        {
            isWriting = false;
            Microphone.End("");
            //m_FilePath = m_Path + @"\" + m_FilePrefix + DateTime.Now.ToString("yyyyMMddHHmmss") + ".wav";
            savwav.Save(m_FilePath, voiceClip);
        }
    }

    IEnumerator StreamByte()
    {
        Debug.Log("StreamByte");
        while(true)
        {
            if (!isWriting)
            {
                break;
            }

            byte[] byteRows = GetByteFileStream(m_FilePath, TrimWav(voiceClip, Microphone.GetPosition(microphoneName)));
            var hz = voiceClip.frequency;
            SendData(byteRows);
            yield return new WaitForSeconds(sampleWindow / hz);

        }
    }

    public void SendLoudnessFromVoiceClip(int clipPosition, AudioClip clip) //Microphone.GetPosition(null)
    {
        int startPosition = clipPosition - sampleWindow;
        if (startPosition < 0) return;
        seconds += interval;
        sendRow = seconds.ToString() + ",";
        Debug.Log("startPosition : " + startPosition);
        float[] waveData = new float[sampleWindow];
        clip.GetData(waveData, startPosition);
        for (int i = 0; i < sampleWindow; i++)
        {
            sendRow += waveData[i].ToString("F3") + ",";
        }
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

    public byte[] GetByteFileStream(string filename, AudioClip clip)
    {
        if (!filename.ToLower().EndsWith(".wav"))
        {
            filename += ".wav";
        }

        var filepath = filename;
        byte[] ImageData;

        using (var fileStream = CreateEmpty(filepath))
        {
            savwav.ConvertAndWrite(fileStream, clip);
            savwav.WriteHeader(fileStream, clip);
            ImageData = new byte[fileStream.Length];
            fileStream.Read(ImageData, 0, System.Convert.ToInt32(fileStream.Length));
            fileStream.Close();
        }

        return ImageData;
    }

    private FileStream CreateEmpty(string filepath)
    {
        var fileStream = new FileStream(filepath, FileMode.Create);
        byte emptyByte = new byte();

        for (int i = 0; i < HEADER_SIZE; i++) //preparing the header
        {
            fileStream.WriteByte(emptyByte);
        }
        return fileStream;
    }

    public AudioClip TrimWav(AudioClip clip, int clipPosition)
    {
        int startPosition = clipPosition - sampleWindow;
        if (startPosition < 0) return null;
        var samples = new float[sampleWindow];
        clip.GetData(samples, startPosition);

        return TrimWav(new List<float>(samples), clip.channels, clip.frequency);
    }

    public AudioClip TrimWav(List<float> samples, int channels, int hz)
    {
        return TrimWav(samples, channels, hz, false);
    }

    public AudioClip TrimWav(List<float> samples, int channels, int hz, bool stream)
    {
        var clip = AudioClip.Create("TempClip", samples.Count, channels, hz, stream);
        clip.SetData(samples.ToArray(), 0);
        return clip;
    }
}
