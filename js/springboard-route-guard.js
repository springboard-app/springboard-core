firebase.auth().onAuthStateChanged(user => {
    if (user === null) {
        location.replace("index.html");
    }
});