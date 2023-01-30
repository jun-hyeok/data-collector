package `in`.rithikjain.stopwatch

import android.app.*
import android.app.ActivityManager.*
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Build
import android.os.IBinder
import android.provider.Settings
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import java.io.*
import java.net.Socket
import java.util.*


var isTurnON = "0"
var applicationList = ArrayList<String?>()


class SimpleThread: Thread() {
    public override fun run() {
        //val client = Socket("192.168.1.141", 8001)
        val client = Socket("192.168.1.129", 8001)
        val writer = client.getOutputStream()
        var message = isTurnON
        /**
        for (i in applicationList.indices)
            message += applicationList[i]
        **/
        writer.write(message.toByteArray())
        client.close()
    }
}

class StopwatchService : Service() {
    companion object {
        // Channel ID for notifications
        const val CHANNEL_ID = "Stopwatch_Notifications"

        // Service Actions
        const val START = "START"
        const val PAUSE = "PAUSE"
        const val RESET = "RESET"
        const val GET_STATUS = "GET_STATUS"
        const val MOVE_TO_FOREGROUND = "MOVE_TO_FOREGROUND"
        const val MOVE_TO_BACKGROUND = "MOVE_TO_BACKGROUND"

        // Intent Extras
        const val STOPWATCH_ACTION = "STOPWATCH_ACTION"
        const val TIME_ELAPSED = "TIME_ELAPSED"
        const val IS_STOPWATCH_RUNNING = "IS_STOPWATCH_RUNNING"

        // Intent Actions
        const val STOPWATCH_TICK = "STOPWATCH_TICK"
        const val STOPWATCH_STATUS = "STOPWATCH_STATUS"
    }

    private var timeElapsed: Int = 0
    private var isStopWatchRunning = false

    private var updateTimer = Timer()
    private var stopwatchTimer = Timer()

    // Getting access to the NotificationManager
    private lateinit var notificationManager: NotificationManager


    // Network communication part

    /*
    * The system calls onBind() method to retrieve the IBinder only when the first client binds.
    * The system then delivers the same IBinder to any additional clients that bind,
    * without calling onBind() again.
    * */

    override fun onBind(p0: Intent?): IBinder? {
        Log.d("Stopwatch", "Stopwatch onBind")
        return null
    }

    /*
    * onStartCommand() is called every time a client starts the service
    * using startService(Intent intent)
    * We will check for what action has this service been called for and then perform the
    * action accordingly. The action is extracted from the intent that is used to start
    * this service.
    * */
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        createChannel()
        getNotificationManager()

        val action = intent?.getStringExtra(STOPWATCH_ACTION)!!

        Log.d("Stopwatch", "onStartCommand Action: $action")

        when (action) {
            START -> startStopwatch()
            PAUSE -> pauseStopwatch()
            RESET -> resetStopwatch()
            GET_STATUS -> sendStatus()
            MOVE_TO_FOREGROUND -> moveToForeground()
            MOVE_TO_BACKGROUND -> moveToBackground()
        }

        return START_STICKY
    }

    /*
    * This function is triggered when the app is not visible to the user anymore
    * It check if the stopwatch is running, if it is then it starts a foreground service
    * with the notification.
    * We run another timer to update the notification every second.
    * */
    private fun moveToForeground() {

        if (isStopWatchRunning) {
            startForeground(1, buildNotification())

            updateTimer = Timer()

            updateTimer.scheduleAtFixedRate(object : TimerTask() {
                override fun run() {
                    updateNotification()

                }
            }, 0, 1000)
        }
    }

    /*
    * This function is triggered when the app is visible again to the user
    * It cancels the timer which was updating the notification every second
    * It also stops the foreground service and removes the notification
    * */
    private fun moveToBackground() {
        updateTimer.cancel()
        stopForeground(true)
    }

    /*
    * This function starts the stopwatch
    * Sets the status of stopwatch running to true
    * We start a Timer and increase the timeElapsed by 1 every second and broadcast the value
    * with the action of STOPWATCH_TICK.
    * We will receive this broadcast in the MainActivity to get access to the time elapsed.
    * */
    private fun startStopwatch() {
        isStopWatchRunning = true

        sendStatus()

        stopwatchTimer = Timer()
        stopwatchTimer.scheduleAtFixedRate(object : TimerTask() {
            override fun run() {
                val stopwatchIntent = Intent()
                stopwatchIntent.action = STOPWATCH_TICK

                timeElapsed++

                stopwatchIntent.putExtra(TIME_ELAPSED, timeElapsed)
                sendBroadcast(stopwatchIntent)
            }
        }, 0, 1000)

        val filter = IntentFilter()
        filter.addAction(Intent.ACTION_SCREEN_ON)
        filter.addAction(Intent.ACTION_SCREEN_OFF)

        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                val action = intent!!.action
                val thread = SimpleThread()
                Log.d("Test", "receive : $action")

                when (action) {
                    Intent.ACTION_SCREEN_ON -> {
                        Log.d("Stopwatch", "screen on")
                        setParam("using")
                        thread.start()
                    }
                    Intent.ACTION_SCREEN_OFF -> {
                        Log.d("Stopwatch", "screen off")
                        setParam("not use")
                        thread.start()
                    }
                }
            }
        }
        registerReceiver(receiver, filter);

    }

    private fun setParam(isturn : String)
    {
        isTurnON = isturn
    }

    /*
    * This function pauses the stopwatch
    * Sends an update of the current state of the stopwatch
    * */
    private fun pauseStopwatch() {
        stopwatchTimer.cancel()
        isStopWatchRunning = false
        sendStatus()
    }

    /*
    * This function resets the stopwatch
    * Sends an update of the current state of the stopwatch
    * */
    private fun resetStopwatch() {
        pauseStopwatch()
        timeElapsed = 0
        val thread = SimpleThread()
        setParam("End")
        thread.start()
        sendStatus()
    }

    /*
    * This function is responsible for broadcasting the status of the stopwatch
    * Broadcasts if the stopwatch is running and also the time elapsed
    * */
    private fun sendStatus() {
        val statusIntent = Intent()
        statusIntent.action = STOPWATCH_STATUS
        statusIntent.putExtra(IS_STOPWATCH_RUNNING, isStopWatchRunning)
        statusIntent.putExtra(TIME_ELAPSED, timeElapsed)
        sendBroadcast(statusIntent)
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val notificationChannel = NotificationChannel(
                CHANNEL_ID,
                "Stopwatch",
                NotificationManager.IMPORTANCE_DEFAULT
            )
            notificationChannel.setSound(null, null)
            notificationChannel.setShowBadge(true)
            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager.createNotificationChannel(notificationChannel)
        }
    }

    private fun getNotificationManager() {
        notificationManager = ContextCompat.getSystemService(
            this,
            NotificationManager::class.java
        ) as NotificationManager
    }

    /*
    * This function is responsible for building and returning a Notification with the current
    * state of the stopwatch along with the timeElapsed
    * */
    private fun buildNotification(): Notification {
        val title = if (isStopWatchRunning) {
            "Stopwatch is running!"
        } else {
            "Stopwatch is paused!"
        }

        val hours: Int = timeElapsed.div(60).div(60)
        val minutes: Int = timeElapsed.div(60)
        val seconds: Int = timeElapsed.rem(60)

        val intent = Intent(this, MainActivity::class.java)
        val pIntent = PendingIntent.getActivity(this, 0, intent, PendingIntent.FLAG_MUTABLE)

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(title)
            .setOngoing(true)
            .setContentText(
                "${"%02d".format(hours)}:${"%02d".format(minutes)}:${
                    "%02d".format(
                        seconds
                    )
                }"
            )
            .setColorized(true)
            .setColor(Color.parseColor("#BEAEE2"))
            .setSmallIcon(R.drawable.ic_clock)
            .setOnlyAlertOnce(true)
            .setContentIntent(pIntent)
            .setAutoCancel(true)
            .build()
    }


    /*
    * This function uses the notificationManager to update the existing notification with the new notification
    * */
    private fun updateNotification() {
        notificationManager.notify(
            1,
            buildNotification()
        )
    }

    fun getProcessImportance(importance: Int): String? {
        if (RunningAppProcessInfo.IMPORTANCE_VISIBLE == importance) {
            return "IMPORTANCE_VISIBLE"
        }
        return if (RunningAppProcessInfo.IMPORTANCE_FOREGROUND == importance) {
            "IMPORTANCE_FOREGROUND"
        } else null
    }


    fun getRunningProcessList() {
        val activityManager = getSystemService(Service.ACTIVITY_SERVICE) as ActivityManager
        val applications = activityManager.runningAppProcesses
        for (i in applications.indices) {
            Log.d(
                "logCheck",
                "[" + getApplicationName(applications[i].processName) + "] processName:" + applications[i].processName + ", importance: " + getProcessImportance(
                    applications[i].importance))
            val name = getApplicationName(applications[i].processName)
            applicationList.add(name)
        }
    }

    fun getProcess(){
        val intent = Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS)
        val topPackageName: String
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val recentTasks: List<RunningTaskInfo> =
                Objects.requireNonNull(activityManager).getRunningTasks(Int.MAX_VALUE)
            val tasks = activityManager.appTasks


            if (tasks != null)
            {
                Log.d("threadCheck", tasks.count().toString())
                /**
                for (i in tasks.indices) {
                    val name = tasks[i]
                    applicationList.add(name)
                }
                **/
            }
            else
            {
                Log.d("threadCheck", "not")
            }
            /**
            if (stats != null) {
                val mySortedMap: SortedMap<Long, UsageStats> = TreeMap()
                for (usageStats in stats) {
                    mySortedMap[usageStats.lastTimeUsed] = usageStats
                }
                if (!mySortedMap.isEmpty()) {
                    topPackageName = mySortedMap[mySortedMap.lastKey()]!!.packageName
                    Log.d("threadCheck", topPackageName)
                }
            }
            **/
        }
    }

    private fun getApplicationName(package_name: String): String? {
        val packageinfo = packageManager.getInstalledPackages(PackageManager.GET_ACTIVITIES)
        for (i in packageinfo.indices) {
            val pi = packageinfo[i]
            if (package_name == pi.packageName == true) {
                return pi.applicationInfo.loadLabel(packageManager).toString()
            }
        }
        return null
    }
    /**
    val logcheck = object : Thread() {
        override fun run() {
            while(isTurnON != "End")
            {
                getProcess()
                Log.d("threadCheck", "done")
            }
        }
    }
    **/

}