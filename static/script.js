async function enviar() {
    const input = document.getElementById("input");
    const chat = document.getElementById("chat");

    const mensagem = input.value;

    chat.innerHTML += `<p><b>Você:</b> ${mensagem}</p>`;

    const res = await fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ mensagem })
    });

    const data = await res.json();

    chat.innerHTML += `<p><b>IA:</b> ${data.resposta}</p>`;

    // tocar áudio
    const audio = new Audio(data.audio);
    audio.play();

    input.value = "";
}