/* eslint-env browser */

async function setupRecorder({
  button, // HTMLButtonElement
  sourceSelect, // HTMLSelectElement (speaker language)
  targetSelect, // HTMLSelectElement (translation language)
  transcriptEl, // HTMLElement to display transcription
  translationEl, // HTMLElement to display translation
}) {
  let mediaRecorder = null;
  let chunks = [];

  const reset = () => {
    chunks = [];
    mediaRecorder = null;
    button.textContent = "🎙️ Speak";
  };

  button.addEventListener("click", async () => {
    // Start recording
    if (!mediaRecorder) {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.addEventListener("dataavailable", (e) => {
        chunks.push(e.data);
      });

      mediaRecorder.addEventListener("stop", async () => {
        const blob = new Blob(chunks, { type: "audio/webm" });
        reset();

        const form = new FormData();
        form.append("file", blob, "speech.webm");
        form.append("source_lang", sourceSelect.value);
        form.append("target_lang", targetSelect.value);

        transcriptEl.textContent = "… processing …";
        translationEl.textContent = "";

        try {
          const res = await fetch("/transcribe_translate", {
            method: "POST",
            body: form,
          });
          if (!res.ok) {
            throw new Error(await res.text());
          }
          const { transcription, translation } = await res.json();
          transcriptEl.textContent = transcription;
          translationEl.textContent = translation;
        } catch (err) {
          transcriptEl.textContent = "Error: " + err.message;
          translationEl.textContent = "";
        }
      });

      mediaRecorder.start();
      button.textContent = "⏹️ Stop";
    } else if (mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
  });
}

// Doctor side
setupRecorder({
  button: document.getElementById("doctor-record"),
  sourceSelect: document.getElementById("doctor-lang"),
  targetSelect: document.getElementById("patient-lang"),
  transcriptEl: document.querySelector("#doctor .transcription"),
  translationEl: document.querySelector("#doctor .translation"),
});

// Patient side
setupRecorder({
  button: document.getElementById("patient-record"),
  sourceSelect: document.getElementById("patient-speak-lang"),
  targetSelect: document.getElementById("doctor-target-lang"),
  transcriptEl: document.querySelector("#patient .transcription"),
  translationEl: document.querySelector("#patient .translation"),
}); 