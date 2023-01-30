using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class ExperimentReady : MonoBehaviour
{
    public GameObject labelInput;
    //private TMP_InputField labelInputInputFiedld;
    //private TMP_InputField prefixInputFiedld;
    public TMP_Text labelInputInputFiedld;
    public TMP_Text prefixInputFiedld;

    // Start is called before the first frame update
    void Start()
    {
        //labelInputInputFiedld = labelInput.GetComponent<TMP_InputField>();
        //prefixInputFiedld = prefix.GetComponent<TMP_InputField>();
    }

    // Update is called once per frame
    void Update()
    {
        if (StateParam.expState == State.Ready)
        {

            if (labelInput.activeSelf && Input.GetKeyDown(KeyCode.Return))
            {
                labelInput.gameObject.SetActive(false);
                StateParam.expState = State.Cognitive;
            }
        }
        
    }
}
