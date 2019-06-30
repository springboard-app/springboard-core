firebase.auth().onAuthStateChanged(user => {
    if (user === null) {
        location.replace("index.html");
    }
});

firebase.auth.authSignIn = new Promise((resolve, reject) => {
    firebase.auth().onAuthStateChanged(user => {
        if (user) {
            resolve(user);
        }
    })
});