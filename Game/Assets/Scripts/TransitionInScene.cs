using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class TransitionInScene : MonoBehaviour {

    Image image;

	// Use this for initialization
	void Start () {
        image = GetComponent<Image>();
	}
	
	public void AlphaGuiToZero()
    {
        gameObject.SetActive(false);
    }
}
