using UnityEngine;
using UnityEngine.UI;

public class RightCtrl : MonoBehaviour {

    SteamVR_Controller.Device RightHand;
    SteamVR_TrackedObject TrackedRight;

	// Use this for initialization
	void Start () {
        TrackedRight = GetComponent<SteamVR_TrackedObject>();
	}
	
	// Update is called once per frame
	void Update () {
        RightHand = SteamVR_Controller.Input((int)TrackedRight.index);
        //Debug.Log(this.transform.localPosition + "; " + this.name + ": " + this.transform.position);
        Text debugInfo= GameObject.Find("DebugInfo").GetComponent<Text>();
        debugInfo.text = this.transform.position.ToString() +
            "\n" + this.transform.localPosition.ToString() +
            "\n" + this.transform.parent.name + ": " + this.transform.parent.position;
    }
}
