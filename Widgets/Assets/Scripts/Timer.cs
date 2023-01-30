using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class Timer : MonoBehaviour
{
    public float timeRemaining = 10;
    public bool timerIsRuning = false;
    public TMP_Text timeText;
    public GameObject notice;
    // Start is called before the first frame update
    void Start()
    {
        timerIsRuning = true;
    }

    // Update is called once per frame
    void Update()
    {
        if (StateParam.expState == State.gatheringReady && Input.GetKeyDown(KeyCode.Space))
        {
            this.gameObject.SetActive(true);
        }

        if (timeRemaining > 0)
        {
            timeRemaining -= Time.deltaTime;
            DisplayTime(timeRemaining);
        }
        else
        {
            Debug.Log("Over");
            timeRemaining = 0;
            timerIsRuning = false;
            StateParam.expState = State.End;
            notice.SetActive(true);
        }
    }

    void DisplayTime(float timeToDisplay)
    {
        timeToDisplay += 1f;
        float minutes = Mathf.FloorToInt(timeToDisplay / 60);
        float seconds = Mathf.FloorToInt(timeToDisplay % 60);
        timeText.text = string.Format("{0:00}:{1:00}", minutes, seconds);
    }
}
