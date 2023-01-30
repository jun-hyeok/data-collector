using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class SetIPAddress : MonoBehaviour
{
    public TMP_Text ipAddress;
    public GameObject inputField;
    public GameObject congnitiveEmotionalStatus;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (inputField.activeSelf && Input.GetKeyDown(KeyCode.Return))
        {
            StateParam.ip = ipAddress.text;
            inputField.SetActive(false);
        }

        if (congnitiveEmotionalStatus.activeSelf && !inputField.activeSelf && Input.GetKeyDown(KeyCode.Escape))
        {
            inputField.SetActive(true);
            congnitiveEmotionalStatus.SetActive(false);
        }
    }
}
