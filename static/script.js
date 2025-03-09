document.addEventListener("DOMContentLoaded", function() {
    const recordBtn = document.getElementById("record-btn");
    const registerSection = document.getElementById("register-section");
    const registerBtn = document.getElementById("register-btn");

    recordBtn.addEventListener("click", function() {
        recordBtn.innerHTML = "🎙️ Listening...";
        fetch("/predict")
            .then(response => response.json())
            .then(data => {
                recordBtn.innerHTML = "🎙️ Start Recording";
                document.getElementById("greeting").innerText = data.greeting;
                document.getElementById("speaker").innerText = data.speaker;
                document.getElementById("emotion").innerText = data.emotion;
                document.getElementById("age-gender").innerText = data.age_gender;

                if (data.speaker === "Unknown User") {
                    registerSection.style.display = "block";
                } else {
                    registerSection.style.display = "none";
                }
            });
    });

    registerBtn.addEventListener("click", function() {
        const name = document.getElementById("name").value;
        if (name.trim() === "") {
            alert("Please enter a name!");
            return;
        }

        fetch("/register", {
            method: "POST",
            body: new URLSearchParams({ name: name }),
            headers: { "Content-Type": "application/x-www-form-urlencoded" }
        })
        .then(response => response.json())
        .then(data => {
            alert(data.message);
            registerSection.style.display = "none";
        });
    });
});
